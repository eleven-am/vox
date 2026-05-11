"""LiveKit-backed RTC session bootstrap and developer control channel."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from vox.operations.errors import OperationError, SessionAlreadyConfiguredError
from vox.operations.livekit_conversation import LiveKitConversation
from vox.server.livekit_config import LiveKitConfig
from vox.server.livekit_registry import LiveKitRtcSessionRegistry
from vox.server.routes.conversation import _event_to_wire, _send_error, parse_session_update

logger = logging.getLogger(__name__)
router = APIRouter()


def get_livekit_rtc_registry(request_or_ws: Request | WebSocket) -> LiveKitRtcSessionRegistry:
    registry = getattr(request_or_ws.app.state, "livekit_rtc_registry", None)
    if registry is not None:
        return registry
    config = getattr(request_or_ws.app.state, "livekit_config", None)
    if config is None:
        config = LiveKitConfig.from_env()
    registry = LiveKitRtcSessionRegistry(config=config)
    request_or_ws.app.state.livekit_rtc_registry = registry
    return registry


@router.post("/v1/rtc/sessions")
async def create_rtc_session(request: Request) -> dict:
    try:
        registry = get_livekit_rtc_registry(request)
    except OperationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    record = registry.create_session()
    return {
        "provider": "livekit",
        "session_id": record.session_id,
        "room": record.room,
        "livekit_url": record.livekit_url,
        "client_token": record.client_token,
        "participant_identity": record.client_identity,
        "expires_at": datetime.fromtimestamp(record.expires_at, tz=UTC).isoformat(),
        "join_token_ttl_seconds": registry.join_token_ttl_s,
        "control_url": f"/v1/rtc/sessions/{record.session_id}/control",
    }


@router.post("/v1/rtc/sessions/{session_id}/offer")
async def create_rtc_answer(session_id: str) -> dict:
    raise HTTPException(
        status_code=410,
        detail=(
            "Vox RTC sessions are LiveKit-backed. Browser clients should connect "
            "to the returned livekit_url with client_token instead of posting SDP offers."
        ),
    )


@router.post("/v1/rtc/sessions/{session_id}/candidates")
async def add_rtc_candidate(session_id: str) -> dict:
    raise HTTPException(
        status_code=410,
        detail="Vox no longer accepts browser ICE candidates; LiveKit handles ICE signaling.",
    )


@router.get("/v1/rtc/sessions/{session_id}/events")
async def rtc_media_events(session_id: str, token: str = "") -> dict:
    raise HTTPException(
        status_code=410,
        detail="Vox no longer exposes RTC trickle/SSE events; LiveKit handles media signaling.",
    )


@router.websocket("/v1/rtc/sessions/{session_id}/control")
async def rtc_control_ws(websocket: WebSocket, session_id: str) -> None:
    try:
        registry = get_livekit_rtc_registry(websocket)
    except OperationError:
        await websocket.close(code=1011, reason="LiveKit is not configured")
        return
    record = registry.attach_control(session_id)
    if record is None:
        await websocket.close(code=1008, reason="unknown, expired, or already attached RTC session")
        return

    await websocket.accept()
    scheduler = websocket.app.state.scheduler
    store = getattr(websocket.app.state, "store", None)
    conversation = LiveKitConversation(
        scheduler=scheduler,
        store=store,
        livekit_url=record.livekit_url,
        room=record.room,
        agent_token=record.agent_token,
    )
    record.conversation = conversation

    async def emit_events() -> None:
        async for event in conversation.events():
            wire = _event_to_wire(event)
            if wire is not None:
                wire.setdefault("session_id", session_id)
                with suppress(Exception):
                    await websocket.send_json(wire)

    emit_task = asyncio.create_task(emit_events())

    try:
        await websocket.send_json({
            "type": "rtc.session.attached",
            "session_id": session_id,
            "provider": "livekit",
            "room": record.room,
        })
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                break
            if "text" not in raw or raw["text"] is None:
                await _send_error(websocket, "only JSON text frames are supported")
                continue

            try:
                msg = json.loads(raw["text"])
            except json.JSONDecodeError as exc:
                await _send_error(websocket, f"invalid JSON: {exc}")
                continue

            msg_type = msg.get("type")
            if not msg_type:
                await _send_error(websocket, "missing 'type' field")
                continue

            if msg_type == "session.update":
                try:
                    config = parse_session_update(msg)
                    await conversation.start_session(config)
                except SessionAlreadyConfiguredError:
                    await _send_error(websocket, "session already configured")
                except (OperationError, RuntimeError) as exc:
                    await _send_error(websocket, str(exc))
                continue

            if conversation.config is None:
                await _send_error(websocket, "send session.update first")
                continue

            if msg_type == "response.start":
                await conversation.start_response()
            elif msg_type == "response.delta":
                response = msg.get("response", {}) or {}
                text = response.get("delta") or msg.get("delta")
                if not text:
                    await _send_error(websocket, "response.delta requires 'delta' text")
                    continue
                await conversation.append_response_text(text)
            elif msg_type == "response.commit":
                await conversation.commit_response()
            elif msg_type == "response.cancel":
                await conversation.cancel_response()
            else:
                await _send_error(websocket, f"unknown control message type: {msg_type!r}")

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("RTC control WS error")
        with suppress(Exception):
            await _send_error(websocket, "internal error; closing")
    finally:
        await conversation.end_of_stream()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(emit_task, timeout=5.0)
        if not emit_task.done():
            emit_task.cancel()
            with suppress(Exception):
                await emit_task
        await conversation.close()
        record.conversation = None
        registry.detach_control(session_id)
        registry.close(session_id)
        with suppress(Exception):
            await websocket.close()
