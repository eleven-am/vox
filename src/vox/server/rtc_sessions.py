from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from vox.server.rtc_ice import ice_servers_from_env


class RtcSessionRegistryProtocol(Protocol):
    join_token_ttl_s: int

    def create_session(self) -> tuple[Any, str]: ...


@dataclass(frozen=True)
class RtcSessionBootstrapRequest:
    forward_browser_events: bool | None = None


def parse_rtc_session_bootstrap_request(body: Any) -> RtcSessionBootstrapRequest:
    if isinstance(body, dict) and "browser_events" in body:
        return RtcSessionBootstrapRequest(
            forward_browser_events=bool(body["browser_events"]),
        )
    return RtcSessionBootstrapRequest()


def create_rtc_session_bootstrap(
    *,
    registry: RtcSessionRegistryProtocol,
    request: RtcSessionBootstrapRequest,
) -> dict[str, Any]:
    record, client_token = registry.create_session()
    if request.forward_browser_events is not None:
        record.forward_browser_events = request.forward_browser_events
    return {
        "session_id": record.session_id,
        "client_token": client_token,
        "expires_at": datetime.fromtimestamp(record.expires_at, tz=UTC).isoformat(),
        "join_token_ttl_seconds": registry.join_token_ttl_s,
        "ice_servers": ice_servers_from_env(now=record.created_at),
    }
