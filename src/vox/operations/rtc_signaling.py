"""Canonical RTC session and browser-signaling operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
    restart: bool = False
    generation: int | None = None


@dataclass(frozen=True)
class RtcOfferAnswer:
    session_id: str
    answer_type: str
    sdp: str
    attempt_id: str | None = None
    commit_callback: Callable[[], Awaitable[None]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    rollback_callback: Callable[[], Awaitable[None]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    async def commit(self) -> None:
        if self.commit_callback is not None:
            await self.commit_callback()

    async def rollback(self) -> None:
        if self.rollback_callback is not None:
            await self.rollback_callback()


@dataclass(frozen=True)
class RtcCandidateRequest:
    session_id: str
    candidate: str | None
    sdp_mid: str | None = None
    sdp_m_line_index: int | None = None
    username_fragment: str | None = None
    generation: int | None = None


@dataclass(frozen=True)
class RtcCandidateResult:
    ok: bool


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
    record = registry.get(request.session_id)
    if record is None or (record.browser_attached and not request.restart):
        raise RtcSessionNotFoundError(request.session_id)
    result = await _exchange_rtc_offer(
        registry=registry,
        record=record,
        request=request,
    )
    return result


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
        restart=request.restart,
        generation=request.generation,
    )
    return RtcOfferAnswer(
        session_id=result.session_id,
        answer_type=result.answer_type,
        sdp=result.sdp,
        attempt_id=result.attachment.attempt_id,
        commit_callback=result.commit,
        rollback_callback=result.rollback,
    )


async def add_server_rtc_candidate(
    *,
    registry: RtcSessionRegistry,
    request: RtcCandidateRequest,
) -> RtcCandidateResult:
    record = registry.get(request.session_id)
    if record is None:
        raise RtcSessionNotFoundError(request.session_id)
    pending = record.pending_rtc_attachment
    if pending is not None and getattr(pending, "generation", None) == request.generation:
        peer = pending.peer
    else:
        peer = record.rtc_peer
    if peer is None:
        raise RtcSessionNotFoundError(request.session_id)
    return await _add_rtc_candidate(record=record, request=request, peer=peer)


async def _add_rtc_candidate(
    *,
    record: RtcSessionRecord,
    request: RtcCandidateRequest,
    peer: Any,
) -> RtcCandidateResult:
    candidate = {
        "candidate": request.candidate,
        "sdpMid": request.sdp_mid,
        "sdpMLineIndex": request.sdp_m_line_index,
        "usernameFragment": request.username_fragment,
    }
    try:
        result = await add_browser_rtc_candidate(
            record=record,
            candidate=candidate,
            peer=peer,
        )
    except InvalidIceCandidateError as exc:
        raise InvalidRtcCandidateError() from exc
    return RtcCandidateResult(ok=bool(result["ok"]))


def rtc_session_bootstrap_payload(result: RtcSessionBootstrap) -> dict[str, Any]:
    return {
        "session_id": result.session_id,
        "expires_at": result.expires_at,
        "attach_ttl_seconds": result.attach_ttl_seconds,
        "ice_servers": list(result.ice_servers),
    }
