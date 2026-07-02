"""WebSocket endpoint for agent-facing voice conversations.

Wire shape is OpenAI Realtime-compatible so existing SDKs can point at Vox with
a URL swap. See src/vox/conversation/session.py for the wire event names.

This route is NOT intended for direct browser use. Browsers / phones terminate
WebRTC (or SIP) at an *agent* process, which then speaks this WS on behalf of
the user. That keeps Vox's scope limited to speech inference + turn orchestration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vox.core.tasks import drain_task
from vox.logging_context import new_request_id, request_id_var
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationOrchestrator,
    ConvEvent,
    ConvInterruptionDetectedEvent,
    ConvInterruptionFalsePositiveEvent,
    ConvResponseCancelledEvent,
    ConvResponseCommittedEvent,
    ConvResponseCreatedEvent,
    ConvResponseDoneEvent,
    ConvSessionCreatedEvent,
    ConvSpeechStartedEvent,
    ConvSpeechStoppedEvent,
    ConvStateChangedEvent,
    ConvTranscriptDeltaEvent,
    ConvTranscriptDoneEvent,
    ConvTurnEouPredictedEvent,
    execute_conversation_command,
    parse_allow_interruptions,
    parse_response_text,
    serialize_session_config,
)
from vox.operations.conversation import (
    parse_session_update as _operation_parse_session_update,
)
from vox.operations.errors import OperationError

logger = logging.getLogger(__name__)
router = APIRouter()
legacy_router = APIRouter()

# Backward-compatible import surface for adjacent transports while command
# dispatch moves into vox.operations.conversation.
parse_session_update = _operation_parse_session_update


WIRE_SESSION_CREATED = "session.created"
WIRE_SPEECH_STARTED = "input_audio_buffer.speech_started"
WIRE_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
WIRE_TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"
WIRE_TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
WIRE_RESPONSE_CREATED = "response.created"
WIRE_AUDIO_DELTA = "response.audio.delta"
WIRE_RESPONSE_DONE = "response.done"
WIRE_RESPONSE_CANCELLED = "response.cancelled"
WIRE_RESPONSE_COMMITTED = "response.committed"
WIRE_AUDIO_CLEAR = "response.audio.clear"
WIRE_INTERRUPTION_DETECTED = "interruption.detected"
WIRE_INTERRUPTION_FALSE_POSITIVE = "interruption.false_positive"
WIRE_TURN_EOU_PREDICTED = "turn.eou.predicted"
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
    if isinstance(event, ConvTranscriptDeltaEvent):
        return {
            "type": WIRE_TRANSCRIPT_DELTA,
            "delta": event.delta,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
        }
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
        return {"type": WIRE_RESPONSE_CREATED, "response_id": event.response_id}
    if isinstance(event, ConvAudioDeltaEvent):
        return {
            "type": WIRE_AUDIO_DELTA,
            "audio": event.audio_b64,
            "sample_rate": event.sample_rate,
            "audio_format": event.audio_format,
            "response_id": event.response_id,
            "sequence": event.sequence,
        }
    if isinstance(event, ConvAudioClearEvent):
        return {"type": WIRE_AUDIO_CLEAR, "response_id": event.response_id}
    if isinstance(event, ConvResponseDoneEvent):
        return {"type": WIRE_RESPONSE_DONE, "response_id": event.response_id}
    if isinstance(event, ConvResponseCancelledEvent):
        return {"type": WIRE_RESPONSE_CANCELLED, "response_id": event.response_id}
    if isinstance(event, ConvResponseCommittedEvent):
        return {"type": WIRE_RESPONSE_COMMITTED, "response_id": event.response_id}
    if isinstance(event, ConvInterruptionDetectedEvent):
        return {
            "type": WIRE_INTERRUPTION_DETECTED,
            "response_id": event.response_id,
            "vad_active_ms": event.vad_active_ms,
            "partial_transcript": event.partial_transcript,
        }
    if isinstance(event, ConvInterruptionFalsePositiveEvent):
        payload_fp: dict = {
            "type": WIRE_INTERRUPTION_FALSE_POSITIVE,
            "response_id": event.response_id,
            "vad_active_ms": event.vad_active_ms,
            "partial_transcript": event.partial_transcript,
        }
        if event.reason:
            payload_fp["reason"] = event.reason
        return payload_fp
    if isinstance(event, ConvTurnEouPredictedEvent):
        return {
            "type": WIRE_TURN_EOU_PREDICTED,
            "probability": event.probability,
            "threshold": event.threshold,
            "decision": event.decision,
            "action": event.action,
            "delay_ms": event.delay_ms,
            "turn_detector": event.turn_detector,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
        }
    if isinstance(event, ConvStateChangedEvent):
        return {
            "type": WIRE_STATE_CHANGED,
            "state": event.state,
            "previous_state": event.previous_state,
        }
    if isinstance(event, ConvErrorEvent):
        return {"type": "error", "message": event.message}
    return None


@legacy_router.websocket("/v1/conversation")
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

            try:
                await execute_conversation_command(orchestrator, msg)
            except OperationError as exc:
                await _send_error(websocket, str(exc))

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("conversation WS error")
        with suppress(Exception):
            await _send_error(websocket, "internal error; closing")
    finally:
        await orchestrator.end_of_stream()
        await drain_task(emit_task)
        await orchestrator.close()
        with suppress(Exception):
            await websocket.close()
        logger.info("conversation ws closed")
        request_id_var.reset(token)


async def _send_error(websocket: WebSocket, message: str) -> None:
    with suppress(Exception):
        await websocket.send_json({"type": "error", "message": message})


def _parse_allow_interruptions(msg: dict) -> bool:
    return parse_allow_interruptions(msg)


def _parse_response_text(msg: dict, preferred_key: str) -> str | None:
    return parse_response_text(msg, preferred_key)
