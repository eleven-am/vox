from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

from vox.server.livekit_config import LiveKitConfig, LiveKitIssuedToken, LiveKitTokenIssuer


@dataclass
class LiveKitRtcSessionRecord:
    session_id: str
    room: str
    client_identity: str
    agent_identity: str
    client_token: str
    agent_token: str
    created_at: float
    expires_at: float
    livekit_url: str
    control_attached: bool = False
    closed: bool = False
    conversation: Any | None = None


class LiveKitRtcSessionRegistry:
    """Process-local registry for short-lived LiveKit-backed RTC sessions."""

    def __init__(self, *, config: LiveKitConfig) -> None:
        self._config = config
        self._issuer = LiveKitTokenIssuer(config)
        self._sessions: dict[str, LiveKitRtcSessionRecord] = {}

    @property
    def join_token_ttl_s(self) -> int:
        return self._config.join_token_ttl_s

    def create_session(self, *, now: float | None = None) -> LiveKitRtcSessionRecord:
        now = time.time() if now is None else now
        self._prune_expired(now=now)
        suffix = secrets.token_urlsafe(18)
        session_id = f"rtc_{suffix}"
        room = f"vox-{suffix}"
        client_identity = f"{session_id}-browser"
        agent_identity = f"{session_id}-agent"
        client_token = self._issue(room=room, identity=client_identity, name="Vox browser")
        agent_token = self._issue(room=room, identity=agent_identity, name="Vox agent")
        expires_at = min(client_token.expires_at, agent_token.expires_at)
        record = LiveKitRtcSessionRecord(
            session_id=session_id,
            room=room,
            client_identity=client_identity,
            agent_identity=agent_identity,
            client_token=client_token.token,
            agent_token=agent_token.token,
            created_at=now,
            expires_at=expires_at,
            livekit_url=self._config.url,
        )
        self._sessions[session_id] = record
        return record

    def get(self, session_id: str, *, now: float | None = None) -> LiveKitRtcSessionRecord | None:
        now = time.time() if now is None else now
        record = self._sessions.get(session_id)
        if record is None or record.closed:
            return None
        if now >= record.expires_at:
            self.close(session_id)
            return None
        return record

    def attach_control(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> LiveKitRtcSessionRecord | None:
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
        record = self._sessions.get(session_id)
        if record is not None:
            record.closed = True
        self._sessions.pop(session_id, None)

    def _issue(self, *, room: str, identity: str, name: str) -> LiveKitIssuedToken:
        return self._issuer.issue_join_token(
            room=room,
            identity=identity,
            name=name,
            ttl_s=self._config.join_token_ttl_s,
        )

    def _prune_expired(self, *, now: float) -> None:
        for session_id, record in list(self._sessions.items()):
            if record.closed or now >= record.expires_at:
                self.close(session_id)
