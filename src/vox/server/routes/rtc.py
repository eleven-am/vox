"""RTC session bootstrap, WebRTC signaling, and developer control channel."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from vox.core.tasks import drain_task
from vox.operations.conversation import (
    ConvDoneEvent,
    execute_conversation_command,
    serialize_conversation_event,
)
from vox.operations.errors import OperationError
from vox.server.auth import require_api_key
from vox.server.rtc_client_events import (
    emit_client_disconnected_to_control,
    flush_pending_client_events,
    handle_browser_data_channel_message,
    send_client_event_to_browser,
)
from vox.server.rtc_conversation import (
    clear_rtc_audio_if_needed,
    create_rtc_orchestrator,
    forward_wire_event_to_browser,
)
from vox.server.rtc_ice import (
    InvalidIceCandidateError,
    candidate_events_from_sdp,
    ice_servers_from_env,
    parse_browser_ice_candidate,
    patch_aioice_turn_error_code_parser,
    rewrite_private_relay_candidates,
    server_ice_servers_from_env,
)
from vox.server.rtc_media import RtcAudioOutputTrack, cancel_and_drain_media_tasks, pump_input_audio
from vox.server.rtc_media_events import emit_media_event, iter_media_sse
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry
from vox.server.rtc_timeline import RtcTurnTimeline, rtc_audio_stats
from vox.server.websocket import safe_send_ws_error, send_ws_error

logger = logging.getLogger(__name__)
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
    record, client_token = registry.create_session()
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict) and "browser_events" in body:
        record.forward_browser_events = bool(body["browser_events"])
    return {
        "session_id": record.session_id,
        "client_token": client_token,
        "expires_at": datetime.fromtimestamp(record.expires_at, tz=UTC).isoformat(),
        "join_token_ttl_seconds": registry.join_token_ttl_s,
        "ice_servers": ice_servers_from_env(now=record.created_at),
    }


@router.post("/v1/rtc/sessions/{session_id}/offer")
async def create_rtc_answer(request: Request, session_id: str) -> dict:
    registry = get_rtc_registry(request)
    body = await request.json()
    client_token = _bearer_token(request) or str(body.get("client_token") or "")
    attached = registry.attach_browser(client_token)
    if attached is None:
        raise HTTPException(status_code=401, detail="invalid or expired client token")
    record, media_token = attached
    if record.session_id != session_id:
        registry.close(record.session_id)
        raise HTTPException(status_code=404, detail="RTC session not found")

    patch_aioice_turn_error_code_parser()
    pc = RTCPeerConnection(configuration=_rtc_configuration(server_ice_servers_from_env()))
    record.rtc_peer = pc
    if record.audio_output is None:
        record.audio_output = asyncio.Queue()
    record.audio_output_track = RtcAudioOutputTrack(record.audio_output)
    pc.addTrack(record.audio_output_track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        await emit_media_event(
            record,
            {
                "type": "rtc.connection_state",
                "state": pc.connectionState,
            },
        )
        if pc.connectionState in {"closed", "failed"}:
            await emit_client_disconnected_to_control(
                record,
                session_id,
                reason=f"peer_connection_{pc.connectionState}",
                connection_state=pc.connectionState,
                ice_connection_state=pc.iceConnectionState,
                data_channel_state=getattr(record.data_channel, "readyState", None),
            )
            registry.close(session_id)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        await emit_media_event(
            record,
            {
                "type": "rtc.ice_connection_state",
                "state": pc.iceConnectionState,
            },
        )
        if pc.iceConnectionState == "failed":
            await emit_client_disconnected_to_control(
                record,
                session_id,
                reason="ice_connection_failed",
                connection_state=pc.connectionState,
                ice_connection_state=pc.iceConnectionState,
                data_channel_state=getattr(record.data_channel, "readyState", None),
            )

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange() -> None:
        await emit_media_event(
            record,
            {
                "type": "rtc.ice_gathering_state",
                "state": pc.iceGatheringState,
            },
        )

    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "audio":
            task = asyncio.create_task(pump_input_audio(track, lambda pcm, sr: _ingest_media_audio(record, pcm, sr)))
            record.media_tasks.add(task)
            task.add_done_callback(record.media_tasks.discard)

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:
        record.data_channel = channel

        @channel.on("open")
        def on_open() -> None:
            flush_pending_client_events(record)

        @channel.on("close")
        def on_close() -> None:
            if record.data_channel is channel:
                record.data_channel = None
            task = asyncio.create_task(
                emit_client_disconnected_to_control(
                    record,
                    session_id,
                    reason="data_channel_closed",
                    connection_state=pc.connectionState,
                    ice_connection_state=pc.iceConnectionState,
                    data_channel_state=getattr(channel, "readyState", None),
                )
            )
            record.media_tasks.add(task)
            task.add_done_callback(record.media_tasks.discard)

        @channel.on("message")
        def on_message(message) -> None:
            task = asyncio.create_task(handle_browser_data_channel_message(record, session_id, message))
            record.media_tasks.add(task)
            task.add_done_callback(record.media_tasks.discard)

        flush_pending_client_events(record)

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=str(body.get("sdp") or ""),
            type=str(body.get("type") or "offer"),
        )
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    answer_sdp = rewrite_private_relay_candidates(pc.localDescription.sdp)
    for event in candidate_events_from_sdp(answer_sdp):
        await emit_media_event(record, event)
    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": None})

    return {
        "session_id": session_id,
        "media_token": media_token,
        "events_url": f"/v1/rtc/sessions/{session_id}/events?token={media_token}",
        "type": pc.localDescription.type,
        "sdp": answer_sdp,
    }


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
    record = registry.attach_control(session_id)
    if record is None:
        await websocket.close(code=1008, reason="unknown, expired, or already attached RTC session")
        return

    await websocket.accept()
    scheduler = websocket.app.state.scheduler

    orchestrator = create_rtc_orchestrator(scheduler=scheduler, record=record)
    timeline = RtcTurnTimeline(session_id=session_id)

    async def emit_events() -> None:
        async for event in orchestrator.events():
            clear_rtc_audio_if_needed(record, event)
            wire = serialize_conversation_event(event)
            if wire is not None:
                wire.setdefault("session_id", session_id)
                forward_wire_event_to_browser(record, wire)
                with suppress(Exception):
                    await websocket.send_json(wire)
                timing = timeline.observe(wire, audio_stats=rtc_audio_stats(record))
                if timing is not None:
                    with suppress(Exception):
                        await websocket.send_json(timing)
            if isinstance(event, ConvDoneEvent):
                return

    async def emit_client_events() -> None:
        if record.control_events is None:
            return
        while True:
            event = await record.control_events.get()
            if event is None:
                return
            with suppress(Exception):
                await websocket.send_json(event)

    emit_task = asyncio.create_task(emit_events())
    client_event_task = asyncio.create_task(emit_client_events())

    try:
        await websocket.send_json(
            {
                "type": "rtc.session.attached",
                "session_id": session_id,
            }
        )
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                break
            if "text" not in raw or raw["text"] is None:
                await send_ws_error(websocket, "only JSON text frames are supported")
                continue

            try:
                msg = json.loads(raw["text"])
            except json.JSONDecodeError as exc:
                await send_ws_error(websocket, f"invalid JSON: {exc}")
                continue

            try:
                await execute_conversation_command(
                    orchestrator,
                    msg,
                    allow_input_audio=False,
                    client_event_handler=lambda event_name, payload: send_client_event_to_browser(
                        record,
                        event_name,
                        payload,
                    ),
                    unknown_message_label="unknown control message type",
                )
            except OperationError as exc:
                await send_ws_error(websocket, str(exc))

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("RTC control WS error")
        await safe_send_ws_error(websocket, "internal error; closing")
    finally:
        await orchestrator.end_of_stream(flush_response=False)
        await drain_task(emit_task)
        if record.control_events is not None:
            await record.control_events.put(None)
        await drain_task(client_event_task)
        await orchestrator.close()
        record.orchestrator = None
        record.data_channel = None
        if record.audio_output is not None:
            await record.audio_output.put(None)
        if record.media_events is not None:
            await record.media_events.put(None)
        await cancel_and_drain_media_tasks(record)
        if record.rtc_peer is not None:
            with suppress(Exception):
                await record.rtc_peer.close()
        registry.detach_control(session_id)
        registry.close(session_id)
        with suppress(Exception):
            await websocket.close()


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _rtc_configuration(ice_servers: list[dict]) -> RTCConfiguration:
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
            for server in ice_servers
        ]
    )


async def _ingest_media_audio(record: RtcSessionRecord, pcm16: bytes, sample_rate: int | None) -> None:
    orchestrator = record.orchestrator
    if orchestrator is not None and orchestrator.config is not None:
        await orchestrator.ingest_pcm16(pcm16, sample_rate=sample_rate)
