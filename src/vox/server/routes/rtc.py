"""RTC session bootstrap, WebRTC signaling, and developer control channel."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from vox.server.auth import bearer_token_from_http, require_api_key
from vox.server.app_services import app_rtc_registry
from vox.server.rtc_media_events import iter_media_sse
from vox.server.rtc_sessions import (
    RTC_CANDIDATE_TOKEN_FIELDS,
    RTC_OFFER_TOKEN_FIELDS,
    create_rtc_session_bootstrap,
    parse_rtc_session_bootstrap_request,
    rtc_token_from_authorization_or_body,
)
from vox.server.rtc_signaling import (
    RtcSignalingError,
    add_browser_rtc_candidate,
    create_browser_rtc_answer,
)

router = APIRouter()


def rtc_signaling_error_to_http(exc: RtcSignalingError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.post("/v1/rtc/sessions")
async def create_rtc_session(request: Request) -> dict:
    require_api_key(request)
    registry = app_rtc_registry(request)
    try:
        body = await request.json()
    except Exception:
        body = None
    return create_rtc_session_bootstrap(
        registry=registry,
        request=parse_rtc_session_bootstrap_request(body),
    )


@router.post("/v1/rtc/sessions/{session_id}/offer")
async def create_rtc_answer(request: Request, session_id: str) -> dict:
    registry = app_rtc_registry(request)
    body = await request.json()
    client_token = rtc_token_from_authorization_or_body(
        bearer_token_from_http(request),
        body,
        body_fields=RTC_OFFER_TOKEN_FIELDS,
    )
    try:
        return await create_browser_rtc_answer(
            registry=registry,
            session_id=session_id,
            client_token=client_token,
            offer=body,
        )
    except RtcSignalingError as exc:
        raise rtc_signaling_error_to_http(exc) from exc


@router.post("/v1/rtc/sessions/{session_id}/candidates")
async def add_rtc_candidate(request: Request, session_id: str) -> dict:
    registry = app_rtc_registry(request)
    body = await request.json()
    token = rtc_token_from_authorization_or_body(
        bearer_token_from_http(request),
        body,
        body_fields=RTC_CANDIDATE_TOKEN_FIELDS,
    )
    try:
        return await add_browser_rtc_candidate(
            registry=registry,
            session_id=session_id,
            media_token=token,
            candidate=body,
        )
    except RtcSignalingError as exc:
        raise rtc_signaling_error_to_http(exc) from exc


@router.get("/v1/rtc/sessions/{session_id}/events")
async def rtc_media_events(request: Request, session_id: str, token: str) -> StreamingResponse:
    registry = app_rtc_registry(request)
    record = registry.validate_media_token(session_id, token)
    if record is None or record.media_events is None:
        raise HTTPException(status_code=401, detail="invalid RTC media token")

    return StreamingResponse(iter_media_sse(record), media_type="text/event-stream")
