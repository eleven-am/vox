from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from vox.server.rtc_tasks import cancel_media_tasks, track_task


@dataclass
class RtcSessionRecord:
    session_id: str
    client_token_hash: str
    created_at: float
    expires_at: float
    control_attached: bool = False
    browser_attached: bool = False
    browser_disconnect_emitted: bool = False
    closed: bool = False
    media_token_hash: str = ""
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


class RtcSessionRegistry:
    """In-memory registry for short-lived local RTC sessions.

    This intentionally stores only ephemeral routing state. Live WebRTC peers,
    control WebSockets, and conversation sessions are process-local objects, so
    persistence would not make active calls survive a restart.
    """

    def __init__(self, *, join_token_ttl_s: int = 120) -> None:
        self._join_token_ttl_s = join_token_ttl_s
        self._sessions: dict[str, RtcSessionRecord] = {}
        self._teardown_tasks: set[asyncio.Task] = set()

    @property
    def join_token_ttl_s(self) -> int:
        return self._join_token_ttl_s

    def create_session(self, *, now: float | None = None) -> tuple[RtcSessionRecord, str]:
        now = time.time() if now is None else now
        self._prune_expired(now=now)
        session_id = f"rtc_{secrets.token_urlsafe(18)}"
        client_token = f"rtc_client_{secrets.token_urlsafe(32)}"
        record = RtcSessionRecord(
            session_id=session_id,
            client_token_hash=self._hash_token(client_token),
            created_at=now,
            expires_at=now + self._join_token_ttl_s,
            control_events=asyncio.Queue(),
        )
        self._sessions[session_id] = record
        return record, client_token

    def get(self, session_id: str, *, now: float | None = None) -> RtcSessionRecord | None:
        now = time.time() if now is None else now
        record = self._sessions.get(session_id)
        if record is None or record.closed:
            return None
        if not record.browser_attached and now >= record.expires_at:
            self.close(session_id)
            return None
        return record

    def consume_client_token(self, client_token: str, *, now: float | None = None) -> RtcSessionRecord | None:
        now = time.time() if now is None else now
        token_hash = self._hash_token(client_token)
        for record in list(self._sessions.values()):
            if record.closed or record.browser_attached:
                continue
            if now >= record.expires_at:
                self.close(record.session_id)
                continue
            if secrets.compare_digest(record.client_token_hash, token_hash):
                record.browser_attached = True
                record.client_token_hash = ""
                return record
        return None

    def attach_browser(self, client_token: str, *, now: float | None = None) -> tuple[RtcSessionRecord, str] | None:
        record = self.consume_client_token(client_token, now=now)
        if record is None:
            return None
        media_token = f"rtc_media_{secrets.token_urlsafe(32)}"
        record.media_token_hash = self._hash_token(media_token)
        record.media_events = asyncio.Queue()
        record.audio_output = asyncio.Queue()
        return record, media_token

    def validate_media_token(self, session_id: str, token: str) -> RtcSessionRecord | None:
        record = self._sessions.get(session_id)
        if record is None or record.closed or not record.media_token_hash:
            return None
        if not secrets.compare_digest(record.media_token_hash, self._hash_token(token)):
            return None
        return record

    def attach_control(self, session_id: str, *, now: float | None = None) -> RtcSessionRecord | None:
        record = self.get(session_id, now=now)
        if record is None or record.control_attached:
            return None
        record.control_attached = True
        return record

    def detach_control(self, session_id: str) -> None:
        record = self._sessions.get(session_id)
        if record is not None:
            record.control_attached = False

    def close(self, session_id: str) -> None:
        record = self._sessions.pop(session_id, None)
        if record is None:
            return
        record.closed = True
        self._release_resources(record)

    def _release_resources(self, record: RtcSessionRecord) -> None:
        cancel_media_tasks(record)
        peer = record.rtc_peer
        record.rtc_peer = None
        if peer is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        track_task(self._teardown_tasks, self._close_peer(peer))

    @staticmethod
    async def _close_peer(peer: Any) -> None:
        with suppress(Exception):
            await peer.close()

    def _prune_expired(self, *, now: float) -> None:
        for session_id, record in list(self._sessions.items()):
            if record.closed or (not record.browser_attached and now >= record.expires_at):
                self.close(session_id)

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
