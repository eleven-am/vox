"""WebSocket endpoint for agent-facing voice conversations.

Wire shape is OpenAI Realtime-compatible so existing SDKs can point at Vox with
a URL swap. See src/vox/conversation/session.py for the wire event names.

This route is NOT intended for direct browser use. Browsers / phones terminate
WebRTC (or SIP) at an *agent* process, which then speaks this WS on behalf of
the user. That keeps Vox's scope limited to speech inference + turn orchestration.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vox.logging_context import new_request_id, request_id_var
from vox.operations.conversation import (
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationOrchestrator,
    ConvEvent,
    ConvResponseCancelledEvent,
    ConvResponseCommittedEvent,
    ConvResponseCreatedEvent,
    ConvResponseDoneEvent,
    ConvSessionCreatedEvent,
    ConvSpeechStartedEvent,
    ConvSpeechStoppedEvent,
    ConvStateChangedEvent,
    ConvTranscriptDoneEvent,
    parse_session_update,
    serialize_session_config,
)
from vox.operations.errors import (
    OperationError,
    SessionAlreadyConfiguredError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


WIRE_SESSION_CREATED = "session.created"
WIRE_SPEECH_STARTED = "input_audio_buffer.speech_started"
WIRE_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
WIRE_TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"
WIRE_RESPONSE_CREATED = "response.created"
WIRE_AUDIO_DELTA = "response.audio.delta"
WIRE_RESPONSE_DONE = "response.done"
WIRE_RESPONSE_CANCELLED = "response.cancelled"
WIRE_RESPONSE_COMMITTED = "response.committed"
WIRE_STATE_CHANGED = "turn.state_changed"


def _event_to_wire(event: ConvEvent) -> dict | None:
    if isinstance(event, ConvSessionCreatedEvent):
        return {
            "type": WIRE_SESSION_CREATED,
            "session": serialize_session_config(event.config),
        }
    if isinstance(event, ConvSpeechStartedEvent):
        return {"type": WIRE_SPEECH_STARTED, "timestamp_ms": event.timestamp_ms}
    if isinstance(event, ConvSpeechStoppedEvent):
        return {"type": WIRE_SPEECH_STOPPED, "timestamp_ms": event.timestamp_ms}
    if isinstance(event, ConvTranscriptDoneEvent):
        payload: dict = {
            "type": WIRE_TRANSCRIPT_DONE,
            "transcript": event.transcript,
            "language": event.language,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
        }
        if event.eou_probability is not None:
            payload["eou_probability"] = event.eou_probability
        if event.entities:
            payload["entities"] = list(event.entities)
        if event.topics:
            payload["topics"] = list(event.topics)
        if event.words:
            payload["words"] = list(event.words)
        return payload
    if isinstance(event, ConvResponseCreatedEvent):
        return {"type": WIRE_RESPONSE_CREATED}
    if isinstance(event, ConvAudioDeltaEvent):
        return {
            "type": WIRE_AUDIO_DELTA,
            "audio": event.audio_b64,
            "sample_rate": event.sample_rate,
            "audio_format": event.audio_format,
        }
    if isinstance(event, ConvResponseDoneEvent):
        return {"type": WIRE_RESPONSE_DONE}
    if isinstance(event, ConvResponseCancelledEvent):
        return {"type": WIRE_RESPONSE_CANCELLED}
    if isinstance(event, ConvResponseCommittedEvent):
        return {"type": WIRE_RESPONSE_COMMITTED}
    if isinstance(event, ConvStateChangedEvent):
        return {
            "type": WIRE_STATE_CHANGED,
            "state": event.state,
            "previous_state": event.previous_state,
        }
    if isinstance(event, ConvErrorEvent):
        return {"type": "error", "message": event.message}
    return None


@router.websocket("/v1/conversation")
async def conversation_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    scheduler = websocket.app.state.scheduler

    incoming = websocket.headers.get("x-request-id")
    rid = incoming.strip() if incoming and incoming.strip() else new_request_id()
    token = request_id_var.set(rid)

    logger.info("conversation ws connected")

    orchestrator = ConversationOrchestrator(scheduler=scheduler)

    async def emit_events() -> None:
        async for event in orchestrator.events():
            wire = _event_to_wire(event)
            if wire is not None:
                with suppress(Exception):
                    await websocket.send_json(wire)
            if isinstance(event, ConvDoneEvent):
                return

    emit_task = asyncio.create_task(emit_events())

    try:
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

            if orchestrator.config is None:
                await _send_error(websocket, "send session.update first")
                continue

            if msg_type == "input_audio_buffer.append":
                audio_b64 = msg.get("audio")
                if not audio_b64:
                    await _send_error(websocket, "audio field required")
                    continue
                try:
                    pcm = base64.b64decode(audio_b64)
                except Exception as exc:  # noqa: BLE001
                    await _send_error(websocket, f"invalid base64 audio: {exc}")
                    continue
                sample_rate = int(msg.get("sample_rate", 0)) or None
                await orchestrator.ingest_pcm16(pcm, sample_rate=sample_rate)

            elif msg_type == "response.start":
                await orchestrator.start_response()

            elif msg_type == "response.delta":
                response = msg.get("response", {}) or {}
                text = response.get("delta") or msg.get("delta")
                if not text:
                    await _send_error(websocket, "response.delta requires 'delta' text")
                    continue
                await orchestrator.append_response_text(text)

            elif msg_type == "response.commit":
                await orchestrator.commit_response()

            elif msg_type == "response.cancel":
                await orchestrator.cancel_response()

            else:
                await _send_error(websocket, f"unknown message type: {msg_type!r}")

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("conversation WS error")
        with suppress(Exception):
            await _send_error(websocket, "internal error; closing")
    finally:
        await orchestrator.end_of_stream()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(emit_task, timeout=5.0)
        if not emit_task.done():
            emit_task.cancel()
            with suppress(Exception):
                await emit_task
        await orchestrator.close()
        with suppress(Exception):
            await websocket.close()
        logger.info("conversation ws closed")
        request_id_var.reset(token)


async def _send_error(websocket: WebSocket, message: str) -> None:
    with suppress(Exception):
        await websocket.send_json({"type": "error", "message": message})
