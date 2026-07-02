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


RTC_OFFER_TOKEN_FIELDS = ("client_token",)
RTC_CANDIDATE_TOKEN_FIELDS = ("media_token", "token")


def parse_rtc_session_bootstrap_request(body: Any) -> RtcSessionBootstrapRequest:
    if isinstance(body, dict) and "browser_events" in body:
        return RtcSessionBootstrapRequest(
            forward_browser_events=bool(body["browser_events"]),
        )
    return RtcSessionBootstrapRequest()


def rtc_token_from_authorization_or_body(
    authorization_token: str | None,
    body: Any,
    *,
    body_fields: tuple[str, ...],
) -> str:
    if authorization_token:
        return authorization_token
    if not isinstance(body, dict):
        return ""
    for field in body_fields:
        token = str(body.get(field) or "").strip()
        if token:
            return token
    return ""


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
