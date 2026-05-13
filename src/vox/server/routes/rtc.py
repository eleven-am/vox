"""RTC session bootstrap, WebRTC signaling, and developer control channel."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConversationOrchestrator,
)
from vox.operations.errors import OperationError, SessionAlreadyConfiguredError
from vox.server.auth import require_api_key
from vox.server.routes.conversation import (
    _event_to_wire,
    _parse_allow_interruptions,
    _parse_response_text,
    _send_error,
    parse_session_update,
)
from vox.server.rtc_client_events import (
    emit_client_event_to_control,
    flush_pending_client_events,
    parse_client_event_message,
    send_client_event_to_browser,
)
from vox.server.rtc_ice import (
    ice_servers_from_env,
    patch_aioice_turn_error_code_parser,
    rewrite_private_relay_candidates,
    server_ice_servers_from_env,
)
from vox.server.rtc_media import RtcAudioOutputTrack, pump_input_audio
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry

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
        await _emit_media_event(record, {
            "type": "rtc.connection_state",
            "state": pc.connectionState,
        })
        if pc.connectionState in {"closed", "failed"}:
            registry.close(session_id)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        await _emit_media_event(record, {
            "type": "rtc.ice_connection_state",
            "state": pc.iceConnectionState,
        })

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange() -> None:
        await _emit_media_event(record, {
            "type": "rtc.ice_gathering_state",
            "state": pc.iceGatheringState,
        })

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

        @channel.on("message")
        def on_message(message) -> None:
            task = asyncio.create_task(_handle_data_channel_message(record, session_id, message))
            record.media_tasks.add(task)
            task.add_done_callback(record.media_tasks.discard)

        flush_pending_client_events(record)

    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=str(body.get("sdp") or ""),
        type=str(body.get("type") or "offer"),
    ))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    answer_sdp = rewrite_private_relay_candidates(pc.localDescription.sdp)
    for event in _candidate_events_from_sdp(answer_sdp):
        await _emit_media_event(record, event)
    await _emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": None})

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

    candidate_payload = body.get("candidate")
    if candidate_payload is None:
        await record.rtc_peer.addIceCandidate(None)
        return {"ok": True}

    candidate = candidate_payload.get("candidate") if isinstance(candidate_payload, dict) else str(candidate_payload)
    try:
        ice = candidate_from_sdp(str(candidate).removeprefix("candidate:"))
    except (AssertionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid ICE candidate") from exc
    if isinstance(candidate_payload, dict):
        ice.sdpMid = candidate_payload.get("sdpMid")
        ice.sdpMLineIndex = candidate_payload.get("sdpMLineIndex")
    else:
        ice.sdpMid = body.get("sdpMid")
        ice.sdpMLineIndex = body.get("sdpMLineIndex")
    await record.rtc_peer.addIceCandidate(ice)
    return {"ok": True}


@router.get("/v1/rtc/sessions/{session_id}/events")
async def rtc_media_events(request: Request, session_id: str, token: str) -> StreamingResponse:
    registry = get_rtc_registry(request)
    record = registry.validate_media_token(session_id, token)
    if record is None or record.media_events is None:
        raise HTTPException(status_code=401, detail="invalid RTC media token")

    async def stream():
        while True:
            event = await record.media_events.get()
            if event is None:
                yield "event: close\ndata: {}\n\n"
                return
            yield f"event: {event.get('type', 'message')}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@legacy_router.websocket("/v1/rtc/sessions/{session_id}/control")
async def rtc_control_ws(websocket: WebSocket, session_id: str) -> None:
    registry = get_rtc_registry(websocket)
    record = registry.attach_control(session_id)
    if record is None:
        await websocket.close(code=1008, reason="unknown, expired, or already attached RTC session")
        return

    await websocket.accept()
    scheduler = websocket.app.state.scheduler

    async def send_rtc_audio(event: ConvAudioDeltaEvent) -> None:
        if record.audio_output_track is not None:
            await record.audio_output_track.enqueue(base64.b64decode(event.audio_b64), event.sample_rate)

    async def wait_for_rtc_playout() -> None:
        if record.audio_output_track is not None:
            await record.audio_output_track.wait_until_drained()

    orchestrator = ConversationOrchestrator(
        scheduler=scheduler,
        pace_response_done_to_audio=True,
        audio_sink=send_rtc_audio,
        wait_for_output_playout=wait_for_rtc_playout,
    )
    record.orchestrator = orchestrator

    async def emit_events() -> None:
        async for event in orchestrator.events():
            if isinstance(event, ConvAudioClearEvent) and record.audio_output_track is not None:
                record.audio_output_track.clear()
            wire = _event_to_wire(event)
            if wire is not None:
                wire.setdefault("session_id", session_id)
                with suppress(Exception):
                    await websocket.send_json(wire)
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
        await websocket.send_json({
            "type": "rtc.session.attached",
            "session_id": session_id,
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
                    await orchestrator.start_session(config)
                except SessionAlreadyConfiguredError:
                    await _send_error(websocket, "session already configured")
                except OperationError as exc:
                    await _send_error(websocket, str(exc))
                continue

            if msg_type == "client.event":
                try:
                    event_name, payload = parse_client_event_message(msg)
                except ValueError as exc:
                    await _send_error(websocket, str(exc))
                    continue
                send_client_event_to_browser(record, event_name, payload)
                continue

            if orchestrator.config is None:
                await _send_error(websocket, "send session.update first")
                continue

            if msg_type == "response.start":
                allow_interruptions = _parse_allow_interruptions(msg)
                await orchestrator.start_response(allow_interruptions=allow_interruptions)
            elif msg_type == "response.delta":
                text = _parse_response_text(msg, "delta")
                if not text:
                    await _send_error(websocket, "response.delta requires 'delta' text")
                    continue
                allow_interruptions = _parse_allow_interruptions(msg)
                await orchestrator.append_response_text(text, allow_interruptions=allow_interruptions)
            elif msg_type == "response.commit":
                await orchestrator.commit_response()
            elif msg_type == "response.cancel":
                await orchestrator.cancel_response()
            elif msg_type == "response.replace_text":
                text = _parse_response_text(msg, "text")
                if not text:
                    await _send_error(websocket, "response.replace_text requires 'text'")
                    continue
                allow_interruptions = _parse_allow_interruptions(msg)
                await orchestrator.replace_response_text(text, allow_interruptions=allow_interruptions)
            else:
                await _send_error(websocket, f"unknown control message type: {msg_type!r}")

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("RTC control WS error")
        with suppress(Exception):
            await _send_error(websocket, "internal error; closing")
    finally:
        await orchestrator.end_of_stream(flush_response=False)
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(emit_task, timeout=5.0)
        if not emit_task.done():
            emit_task.cancel()
            with suppress(Exception):
                await emit_task
        if record.control_events is not None:
            await record.control_events.put(None)
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(client_event_task, timeout=5.0)
        if not client_event_task.done():
            client_event_task.cancel()
            with suppress(Exception):
                await client_event_task
        await orchestrator.close()
        record.orchestrator = None
        record.data_channel = None
        if record.audio_output is not None:
            await record.audio_output.put(None)
        if record.media_events is not None:
            await record.media_events.put(None)
        await _cancel_media_tasks(record)
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
    return RTCConfiguration(iceServers=[
        RTCIceServer(
            urls=server["urls"],
            username=server.get("username"),
            credential=server.get("credential"),
        )
        for server in ice_servers
    ])


async def _emit_media_event(record: RtcSessionRecord, event: dict) -> None:
    if record.media_events is not None:
        await record.media_events.put(event)


async def _cancel_media_tasks(record: RtcSessionRecord) -> None:
    tasks = list(record.media_tasks)
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    record.media_tasks.clear()


async def _ingest_media_audio(record: RtcSessionRecord, pcm16: bytes, sample_rate: int | None) -> None:
    orchestrator = record.orchestrator
    if orchestrator is not None and orchestrator.config is not None:
        await orchestrator.ingest_pcm16(pcm16, sample_rate=sample_rate)


async def _handle_data_channel_message(record: RtcSessionRecord, session_id: str, message) -> None:
    if isinstance(message, bytes):
        try:
            text = message.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("dropping non-UTF-8 RTC data channel message for %s", session_id)
            return
    else:
        text = str(message)

    try:
        message_obj = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("dropping non-JSON RTC data channel message for %s", session_id)
        return

    try:
        event_name, payload = parse_client_event_message(message_obj)
    except ValueError:
        logger.warning("dropping malformed RTC client.event payload for %s", session_id)
        return

    await emit_client_event_to_control(record, session_id, event_name, payload)


def _candidate_events_from_sdp(sdp: str) -> list[dict]:
    events: list[dict] = []
    current_mid: str | None = None
    current_mline = -1
    for line in sdp.splitlines():
        if line.startswith("m="):
            current_mline += 1
        elif line.startswith("a=mid:"):
            current_mid = line.removeprefix("a=mid:")
        elif line.startswith("a=candidate:"):
            events.append({
                "type": "rtc.ice_candidate",
                "candidate": {
                    "candidate": line.removeprefix("a="),
                    "sdpMid": current_mid,
                    "sdpMLineIndex": current_mline if current_mline >= 0 else None,
                },
            })
    return events
