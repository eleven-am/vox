from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections import deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

from vox.server.rtc_media import create_rtc_audio_queue

RtcControlTransport = Literal["pondsocket", "grpc"]
RTC_CONTROL_EVENT_MAX_COUNT = 128
RTC_CONTROL_EVENT_MAX_BYTES = 262_144
RTC_MEDIA_EVENT_MAX_COUNT = 256
RTC_MEDIA_EVENT_MAX_BYTES = 524_288
RTC_TEARDOWN_RETRY_DELAY_S = 0.05
RTC_TEARDOWN_MAX_RETRY_DELAY_S = 5.0

logger = logging.getLogger(__name__)


class RtcEventQueue:
    def __init__(self, *, max_count: int, max_bytes: int) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_count)
        self._sizes: deque[int] = deque()
        self._max_bytes = max_bytes
        self._bytes = 0
        self._closed = False

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    @property
    def buffered_bytes(self) -> int:
        return self._bytes

    async def put(self, item: Any) -> None:
        self.put_nowait(item)

    def put_nowait(self, item: Any) -> None:
        if self._closed:
            raise asyncio.QueueFull
        size = self._item_size(item)
        if self._queue.full() or self._bytes + size > self._max_bytes:
            raise asyncio.QueueFull
        self._queue.put_nowait(item)
        self._sizes.append(size)
        self._bytes += size

    def put_terminal_nowait(self, item: Any) -> None:
        size = self._item_size(item)
        while self._queue.full() or self._bytes + size > self._max_bytes:
            self.get_nowait()
        self._queue.put_nowait(item)
        self._sizes.append(size)
        self._bytes += size

    async def get(self) -> Any:
        item = await self._queue.get()
        size = self._sizes.popleft()
        self._bytes -= size
        return item

    def get_nowait(self) -> Any:
        item = self._queue.get_nowait()
        size = self._sizes.popleft()
        self._bytes -= size
        return item

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while not self.empty():
            self.get_nowait()
        self._queue.put_nowait(None)
        self._sizes.append(0)

    @staticmethod
    def _item_size(item: Any) -> int:
        if item is None:
            return 0
        return len(json.dumps(item, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


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


def cancel_media_tasks(record: Any) -> list[asyncio.Task[Any]]:
    tasks: list[asyncio.Task[Any]] = list(getattr(record, "media_tasks", ()))
    for task in tasks:
        task.cancel()
    media_tasks = getattr(record, "media_tasks", None)
    if media_tasks is not None:
        media_tasks.clear()
    return tasks


async def cancel_and_drain_media_tasks(record: Any) -> None:
    tasks = cancel_media_tasks(record)
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result


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
    pending_rtc_attachment: Any | None = None
    retired_rtc_attachments: list[Any] = field(default_factory=list)
    audio_output_track: Any | None = None
    audio_sender_track: Any | None = None
    input_audio_track: Any | None = None
    data_channel: Any | None = None
    orchestrator: Any | None = None
    media_events: RtcEventQueue | None = None
    control_events: RtcEventQueue | None = None
    audio_output: asyncio.Queue[Any] | None = None
    pending_client_events: list[str] = field(default_factory=list)
    media_tasks: set[asyncio.Task] = field(default_factory=set)
    forward_browser_events: bool = True
    pending_remote_candidates: list[Any] = field(default_factory=list)
    remote_description_set: bool = False
    remote_candidates_complete: bool = False
    negotiation_generation: int | None = None


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
        self._teardown_records: dict[asyncio.Task, RtcSessionRecord] = {}
        self._closing_sessions: dict[str, RtcSessionRecord] = {}
        self._teardown_retry_handles: dict[str, asyncio.TimerHandle] = {}
        self._teardown_retry_attempts: dict[str, int] = {}

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
            control_events=RtcEventQueue(
                max_count=RTC_CONTROL_EVENT_MAX_COUNT,
                max_bytes=RTC_CONTROL_EVENT_MAX_BYTES,
            ),
            media_events=RtcEventQueue(
                max_count=RTC_MEDIA_EVENT_MAX_COUNT,
                max_bytes=RTC_MEDIA_EVENT_MAX_BYTES,
            ),
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
            record.media_events = RtcEventQueue(
                max_count=RTC_MEDIA_EVENT_MAX_COUNT,
                max_bytes=RTC_MEDIA_EVENT_MAX_BYTES,
            )
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
        self.close_record(record)

    def close_record(self, record: RtcSessionRecord) -> None:
        if self._sessions.get(record.session_id) is record:
            self._sessions.pop(record.session_id, None)
        record.closed = True
        self._closing_sessions[record.session_id] = record
        self._start_teardown(record)

    async def close_attached(self, record: RtcSessionRecord, *, orchestrator: Any | None) -> None:
        if orchestrator is not None:
            record.orchestrator = orchestrator
        else:
            record.orchestrator = None
        self.close_record(record)
        task = next(
            (task for task, owned_record in self._teardown_records.items() if owned_record is record),
            None,
        )
        if task is not None:
            await asyncio.shield(task)

    def _start_teardown(self, record: RtcSessionRecord) -> None:
        if any(owned_record is record for owned_record in self._teardown_records.values()):
            return
        retry_handle = self._teardown_retry_handles.pop(record.session_id, None)
        if retry_handle is not None:
            retry_handle.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._drain_record_resources(record))
        self._teardown_tasks.add(task)
        self._teardown_records[task] = record
        task.add_done_callback(self._teardown_finished)

    async def _drain_record_resources(self, record: RtcSessionRecord) -> None:
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
            record.media_events.close()
        if record.control_events is not None:
            record.control_events.close()
        errors: list[BaseException] = []
        try:
            await cancel_and_drain_media_tasks(record)
        except BaseException as exc:
            errors.append(exc)
        orchestrator = record.orchestrator
        if orchestrator is not None:
            try:
                await orchestrator.close()
            except BaseException as exc:
                errors.append(exc)
            else:
                if record.orchestrator is orchestrator:
                    record.orchestrator = None
        for owner in self._owned_peers(record):
            try:
                await owner.close()
            except BaseException as exc:
                errors.append(exc)
            else:
                self._release_peer_reference(record, owner)
        if errors:
            raise errors[0]
        record.input_audio_track = None
        record.data_channel = None
        record.audio_sender_track = None
        self.detach_control(record.session_id)

    def _teardown_finished(self, task: asyncio.Task) -> None:
        try:
            error = task.exception()
        except asyncio.CancelledError:
            error = asyncio.CancelledError()
        record = self._teardown_records.pop(task, None)
        self._teardown_tasks.discard(task)
        if error is not None:
            if record is not None and self._closing_sessions.get(record.session_id) is record:
                self._schedule_teardown_retry(record, error)
            return
        if record is not None:
            retry_handle = self._teardown_retry_handles.pop(record.session_id, None)
            if retry_handle is not None:
                retry_handle.cancel()
            self._teardown_retry_attempts.pop(record.session_id, None)
            self._closing_sessions.pop(record.session_id, None)

    def _schedule_teardown_retry(
        self,
        record: RtcSessionRecord,
        error: BaseException,
    ) -> None:
        attempts = self._teardown_retry_attempts.get(record.session_id, 0) + 1
        self._teardown_retry_attempts[record.session_id] = attempts
        delay = min(
            RTC_TEARDOWN_RETRY_DELAY_S * (2 ** min(attempts - 1, 8)),
            RTC_TEARDOWN_MAX_RETRY_DELAY_S,
        )
        logger.warning(
            "RTC session %s teardown attempt %d failed; retrying in %.2fs: %s",
            record.session_id,
            attempts,
            delay,
            error,
        )
        loop = asyncio.get_running_loop()
        retry_handle = self._teardown_retry_handles.pop(record.session_id, None)
        if retry_handle is not None:
            retry_handle.cancel()
        self._teardown_retry_handles[record.session_id] = loop.call_later(
            delay,
            self._retry_teardown,
            record,
        )

    def _retry_teardown(self, record: RtcSessionRecord) -> None:
        self._teardown_retry_handles.pop(record.session_id, None)
        if self._closing_sessions.get(record.session_id) is record:
            self._start_teardown(record)

    @staticmethod
    def _owned_peers(record: RtcSessionRecord) -> tuple[Any, ...]:
        peers: list[Any] = []
        active = record.rtc_peer
        pending = record.pending_rtc_attachment
        for owner in (active, pending, *record.retired_rtc_attachments):
            peer = getattr(owner, "peer", owner)
            if owner is not None and all(peer is not getattr(owned, "peer", owned) for owned in peers):
                peers.append(owner)
        return tuple(peers)

    @staticmethod
    def _release_peer_reference(record: RtcSessionRecord, owner: Any) -> None:
        if record.rtc_peer is getattr(owner, "peer", owner):
            record.rtc_peer = None
        if record.pending_rtc_attachment is owner:
            record.pending_rtc_attachment = None
        record.retired_rtc_attachments[:] = [
            retired for retired in record.retired_rtc_attachments if retired is not owner
        ]

    async def drain_teardowns(self) -> None:
        for record in tuple(self._closing_sessions.values()):
            self._start_teardown(record)
        tasks = tuple(self._teardown_tasks)
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def close_all(self) -> None:
        for session_id in tuple(self._sessions):
            self.close(session_id)
        await self.drain_teardowns()

    def _prune_expired(self, *, now: float) -> None:
        for session_id, record in list(self._sessions.items()):
            if record.closed or (not record.browser_attached and now >= record.expires_at):
                self.close(session_id)
