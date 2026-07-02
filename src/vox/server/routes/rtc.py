"""RTC session bootstrap, WebRTC signaling, and developer control channel."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse

from vox.server.auth import require_api_key
from vox.server.rtc_control import handle_rtc_control_ws
from vox.server.rtc_ice import (
    InvalidIceCandidateError,
    parse_browser_ice_candidate,
)
from vox.server.rtc_media_events import iter_media_sse
from vox.server.rtc_registry import RtcSessionRegistry
from vox.server.rtc_sessions import (
    create_rtc_session_bootstrap,
    parse_rtc_session_bootstrap_request,
)
from vox.server.rtc_signaling import RtcSignalingError, create_browser_rtc_answer

router = APIRouter()
legacy_router = APIRouter()


def get_rtc_registry(request_or_ws: Request | WebSocket) -> RtcSessionRegistry:
    registry = getattr(request_or_ws.app.state, "rtc_registry", None)
    if registry is None:
        registry = RtcSessionRegistry()
        request_or_ws.app.state.rtc_registry = registry
    return registry


@router.post("/v1/rtc/sessions")
async def create_rtc_session(request: Request) -> dict:
    require_api_key(request)
    registry = get_rtc_registry(request)
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
    registry = get_rtc_registry(request)
    body = await request.json()
    client_token = _bearer_token(request) or str(body.get("client_token") or "")
    try:
        return await create_browser_rtc_answer(
            registry=registry,
            session_id=session_id,
            client_token=client_token,
            offer=body,
        )
    except RtcSignalingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/v1/rtc/sessions/{session_id}/candidates")
async def add_rtc_candidate(request: Request, session_id: str) -> dict:
    registry = get_rtc_registry(request)
    body = await request.json()
    token = _bearer_token(request) or str(body.get("media_token") or body.get("token") or "")
    record = registry.validate_media_token(session_id, token)
    if record is None or record.rtc_peer is None:
        raise HTTPException(status_code=401, detail="invalid RTC media token")

    try:
        ice = parse_browser_ice_candidate(body)
    except InvalidIceCandidateError as exc:
        raise HTTPException(status_code=400, detail="invalid ICE candidate") from exc
    await record.rtc_peer.addIceCandidate(ice)
    return {"ok": True}


@router.get("/v1/rtc/sessions/{session_id}/events")
async def rtc_media_events(request: Request, session_id: str, token: str) -> StreamingResponse:
    registry = get_rtc_registry(request)
    record = registry.validate_media_token(session_id, token)
    if record is None or record.media_events is None:
        raise HTTPException(status_code=401, detail="invalid RTC media token")

    return StreamingResponse(iter_media_sse(record), media_type="text/event-stream")


@legacy_router.websocket("/v1/rtc/sessions/{session_id}/control")
async def rtc_control_ws(websocket: WebSocket, session_id: str) -> None:
    registry = get_rtc_registry(websocket)
    scheduler = websocket.app.state.scheduler
    await handle_rtc_control_ws(
        websocket,
        session_id,
        registry=registry,
        scheduler=scheduler,
    )


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None
