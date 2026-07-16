"""Canonical RTC session and browser-signaling operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from vox.operations.errors import (
    InvalidRtcCandidateError,
    RtcSessionNotFoundError,
)
from vox.server.rtc_ice import InvalidIceCandidateError, ice_servers_from_env
from vox.server.rtc_registry import (
    RtcControlTransport,
    RtcSessionRecord,
    RtcSessionRegistry,
)
from vox.server.rtc_signaling import add_browser_rtc_candidate, create_browser_rtc_answer


@dataclass(frozen=True)
class RtcSessionBootstrapRequest:
    control_transport: RtcControlTransport
    forward_browser_events: bool | None = None


@dataclass(frozen=True)
class RtcSessionBootstrap:
    session_id: str
    expires_at: str
    attach_ttl_seconds: int
    ice_servers: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RtcOfferRequest:
    session_id: str
    offer_type: str
    sdp: str


@dataclass(frozen=True)
class RtcOfferAnswer:
    session_id: str
    answer_type: str
    sdp: str


@dataclass(frozen=True)
class RtcCandidateRequest:
    session_id: str
    candidate: str | None
    sdp_mid: str | None = None
    sdp_m_line_index: int | None = None
    username_fragment: str | None = None


@dataclass(frozen=True)
class RtcCandidateResult:
    ok: bool


@dataclass(frozen=True)
class RtcCloseResult:
    closed: bool


def create_rtc_session(
    *,
    registry: RtcSessionRegistry,
    request: RtcSessionBootstrapRequest,
) -> RtcSessionBootstrap:
    record = registry.create_session(control_transport=request.control_transport)
    if request.forward_browser_events is not None:
        record.forward_browser_events = request.forward_browser_events
    return RtcSessionBootstrap(
        session_id=record.session_id,
        expires_at=datetime.fromtimestamp(record.expires_at, tz=UTC).isoformat(),
        attach_ttl_seconds=registry.attach_ttl_s,
        ice_servers=tuple(ice_servers_from_env(now=record.created_at)),
    )


async def exchange_server_rtc_offer(
    *,
    registry: RtcSessionRegistry,
    request: RtcOfferRequest,
) -> RtcOfferAnswer:
    record = registry.attach_browser_session(request.session_id)
    if record is None:
        raise RtcSessionNotFoundError(request.session_id)
    return await _exchange_rtc_offer(
        registry=registry,
        record=record,
        request=request,
    )


async def _exchange_rtc_offer(
    *,
    registry: RtcSessionRegistry,
    record: RtcSessionRecord,
    request: RtcOfferRequest,
) -> RtcOfferAnswer:
    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": request.offer_type, "sdp": request.sdp},
    )
    return RtcOfferAnswer(
        session_id=str(result["session_id"]),
        answer_type=str(result["type"]),
        sdp=str(result["sdp"]),
    )


async def add_server_rtc_candidate(
    *,
    registry: RtcSessionRegistry,
    request: RtcCandidateRequest,
) -> RtcCandidateResult:
    record = registry.get(request.session_id)
    if record is None or record.rtc_peer is None:
        raise RtcSessionNotFoundError(request.session_id)
    return await _add_rtc_candidate(record=record, request=request)


async def _add_rtc_candidate(
    *,
    record: RtcSessionRecord,
    request: RtcCandidateRequest,
) -> RtcCandidateResult:
    candidate = {
        "candidate": request.candidate,
        "sdpMid": request.sdp_mid,
        "sdpMLineIndex": request.sdp_m_line_index,
        "usernameFragment": request.username_fragment,
    }
    try:
        result = await add_browser_rtc_candidate(record=record, candidate=candidate)
    except InvalidIceCandidateError as exc:
        raise InvalidRtcCandidateError() from exc
    return RtcCandidateResult(ok=bool(result["ok"]))


def close_rtc_session(
    *,
    registry: RtcSessionRegistry,
    session_id: str,
) -> RtcCloseResult:
    record = registry.get(session_id)
    if record is None:
        return RtcCloseResult(closed=False)
    registry.close(session_id)
    return RtcCloseResult(closed=True)


def rtc_session_bootstrap_payload(result: RtcSessionBootstrap) -> dict[str, Any]:
    return {
        "session_id": result.session_id,
        "expires_at": result.expires_at,
        "attach_ttl_seconds": result.attach_ttl_seconds,
        "ice_servers": list(result.ice_servers),
    }
