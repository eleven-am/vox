from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Coroutine
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal

from vox.server.rtc_media import create_rtc_audio_queue

RtcControlTransport = Literal["pondsocket", "grpc"]


def track_task(tasks: set[asyncio.Task], coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    task = asyncio.create_task(coro)
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


def track_media_task(record: Any, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    media_tasks = getattr(record, "media_tasks", None)
    if media_tasks is not None:
        return track_task(media_tasks, coro)
    task = asyncio.create_task(coro)
    return task


def cancel_media_tasks(record: Any) -> list[asyncio.Task]:
    tasks = list(getattr(record, "media_tasks", ()))
    for task in tasks:
        task.cancel()
    media_tasks = getattr(record, "media_tasks", None)
    if media_tasks is not None:
        media_tasks.clear()
    return tasks


async def cancel_and_drain_media_tasks(record: Any) -> None:
    tasks = cancel_media_tasks(record)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@dataclass
class RtcSessionRecord:
    session_id: str
    created_at: float
    expires_at: float
    expected_control_transport: RtcControlTransport
    control_attached: bool = False
    attached_control_transport: RtcControlTransport | None = None
    browser_attached: bool = False
    browser_disconnect_emitted: bool = False
    closed: bool = False
    rtc_peer: Any | None = None
    audio_output_track: Any | None = None
    data_channel: Any | None = None
    orchestrator: Any | None = None
    media_events: asyncio.Queue[dict | None] | None = None
    control_events: asyncio.Queue[dict | None] | None = None
    audio_output: asyncio.Queue[Any] | None = None
    pending_client_events: list[str] = field(default_factory=list)
    media_tasks: set[asyncio.Task] = field(default_factory=set)
    forward_browser_events: bool = True
    pending_remote_candidates: list[Any] = field(default_factory=list)
    remote_description_set: bool = False
    remote_candidates_complete: bool = False
    ice_restart_in_progress: bool = False


class RtcSessionRegistry:
    """In-memory registry for short-lived local RTC sessions.

    This intentionally stores only ephemeral routing state. Live WebRTC peers,
    control WebSockets, and conversation sessions are process-local objects, so
    persistence would not make active calls survive a restart.
    """

    def __init__(self, *, attach_ttl_s: int = 120) -> None:
        self._attach_ttl_s = attach_ttl_s
        self._sessions: dict[str, RtcSessionRecord] = {}
        self._teardown_tasks: set[asyncio.Task] = set()

    @property
    def attach_ttl_s(self) -> int:
        return self._attach_ttl_s

    def create_session(
        self,
        *,
        control_transport: RtcControlTransport,
        now: float | None = None,
    ) -> RtcSessionRecord:
        now = time.time() if now is None else now
        self._prune_expired(now=now)
        session_id = f"rtc_{secrets.token_urlsafe(18)}"
        record = RtcSessionRecord(
            session_id=session_id,
            created_at=now,
            expires_at=now + self._attach_ttl_s,
            expected_control_transport=control_transport,
            control_events=asyncio.Queue(),
            media_events=asyncio.Queue(),
            audio_output=create_rtc_audio_queue(),
        )
        self._sessions[session_id] = record
        return record

    def get(self, session_id: str, *, now: float | None = None) -> RtcSessionRecord | None:
        now = time.time() if now is None else now
        record = self._sessions.get(session_id)
        if record is None or record.closed:
            return None
        if not record.browser_attached and now >= record.expires_at:
            self.close(session_id)
            return None
        return record

    def attach_browser_session(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> RtcSessionRecord | None:
        """Attach a browser using an already-authenticated server call."""
        record = self.get(session_id, now=now)
        if record is None or record.browser_attached:
            return None
        record.browser_attached = True
        if record.media_events is None:
            record.media_events = asyncio.Queue()
        if record.audio_output is None:
            record.audio_output = create_rtc_audio_queue()
        return record

    def attach_control(
        self,
        session_id: str,
        *,
        transport: RtcControlTransport,
        now: float | None = None,
    ) -> RtcSessionRecord | None:
        record = self.get(session_id, now=now)
        if record is None or record.control_attached or record.expected_control_transport != transport:
            return None
        record.control_attached = True
        record.attached_control_transport = transport
        return record

    def detach_control(self, session_id: str) -> None:
        record = self._sessions.get(session_id)
        if record is not None:
            record.control_attached = False
            record.attached_control_transport = None

    def close(self, session_id: str) -> None:
        record = self._sessions.pop(session_id, None)
        if record is None:
            return
        record.closed = True
        self._release_resources(record)

    async def close_attached(self, record: RtcSessionRecord, *, orchestrator: Any | None) -> None:
        if orchestrator is not None:
            with suppress(Exception):
                await orchestrator.close()
        record.orchestrator = None
        record.data_channel = None
        if record.audio_output is not None:
            if record.audio_output_track is not None:
                record.audio_output_track.clear()
            while True:
                try:
                    record.audio_output.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await record.audio_output.put(None)
        if record.media_events is not None:
            await record.media_events.put(None)
        await cancel_and_drain_media_tasks(record)
        if record.rtc_peer is not None:
            peer = record.rtc_peer
            record.rtc_peer = None
            with suppress(Exception):
                await peer.close()
        self.detach_control(record.session_id)
        self.close(record.session_id)

    def _release_resources(self, record: RtcSessionRecord) -> None:
        media_tasks = cancel_media_tasks(record)
        peer = record.rtc_peer
        record.rtc_peer = None
        orchestrator = record.orchestrator
        record.orchestrator = None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        if media_tasks or peer is not None or orchestrator is not None:
            track_task(
                self._teardown_tasks,
                self._drain_resources(media_tasks, orchestrator, peer),
            )

    @staticmethod
    async def _drain_resources(
        media_tasks: list[asyncio.Task],
        orchestrator: Any | None,
        peer: Any | None,
    ) -> None:
        if media_tasks:
            await asyncio.gather(*media_tasks, return_exceptions=True)
        if orchestrator is not None:
            with suppress(Exception):
                await orchestrator.close()
        if peer is not None:
            with suppress(Exception):
                await peer.close()

    async def drain_teardowns(self) -> None:
        while self._teardown_tasks:
            await asyncio.gather(*tuple(self._teardown_tasks), return_exceptions=True)

    async def close_all(self) -> None:
        for session_id in tuple(self._sessions):
            self.close(session_id)
        await self.drain_teardowns()

    def _prune_expired(self, *, now: float) -> None:
        for session_id, record in list(self._sessions.items()):
            if record.closed or (not record.browser_attached and now >= record.expires_at):
                self.close(session_id)
