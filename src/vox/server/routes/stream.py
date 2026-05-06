from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vox.logging_context import new_request_id, request_id_var
from vox.operations.errors import (
    NoDefaultModelError,
    OperationError,
    SessionNotConfiguredError,
    UnknownMessageTypeError,
)
from vox.operations.streaming_transcription import (
    DoneEvent,
    ErrorEvent,
    SessionReadyEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    StreamingTranscriptionConfig,
    StreamingTranscriptionSession,
    TranscriptEvent,
)
from vox.streaming.types import StreamTranscript

logger = logging.getLogger(__name__)

router = APIRouter()


def _operation_error_message(exc: OperationError) -> str:
    if isinstance(exc, NoDefaultModelError):
        return "No STT model specified and no default STT model available"
    return str(exc)


def _config_from_message(data: dict) -> StreamingTranscriptionConfig:
    return StreamingTranscriptionConfig(
        model=data.get("model", "") or "",
        language=data.get("language", "en") or "en",
        sample_rate=int(data.get("sample_rate") or 0) or 16_000,
        partials=bool(data.get("partials", False)),
        partial_window_ms=int(data.get("partial_window_ms") or 1500),
        partial_stride_ms=int(data.get("partial_stride_ms") or 700),
        include_word_timestamps=bool(data.get("include_word_timestamps", False)),
        temperature=float(data.get("temperature", 0.0) or 0.0),
    )


def _transcript_to_payload(transcript: StreamTranscript) -> dict:
    payload = {
        "type": "transcript",
        "text": transcript.text,
        "is_partial": transcript.is_partial,
        "start_ms": transcript.start_ms,
        "end_ms": transcript.end_ms,
        "audio_duration_ms": transcript.audio_duration_ms,
        "processing_duration_ms": transcript.processing_duration_ms,
        "model": transcript.model,
    }
    if transcript.eou_probability is not None:
        payload["eou_probability"] = transcript.eou_probability
    if transcript.entities:
        payload["entities"] = transcript.entities
    if transcript.topics:
        payload["topics"] = transcript.topics
    if transcript.words:
        payload["words"] = transcript.words
    if transcript.segments:
        payload["segments"] = transcript.segments
    return payload


def _event_to_wire(event) -> dict | None:
    if isinstance(event, SessionReadyEvent):
        return {"type": "ready"}
    if isinstance(event, SpeechStartedEvent):
        return {"type": "speech_started", "timestamp_ms": event.timestamp_ms}
    if isinstance(event, SpeechStoppedEvent):
        return {"type": "speech_stopped", "timestamp_ms": event.timestamp_ms}
    if isinstance(event, TranscriptEvent):
        return _transcript_to_payload(event.transcript)
    if isinstance(event, ErrorEvent):
        return {"type": "error", "message": event.message}
    return None


@router.websocket("/v1/audio/stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    incoming = websocket.headers.get("x-request-id")
    rid = incoming.strip() if incoming and incoming.strip() else new_request_id()
    token = request_id_var.set(rid)
    logger.info("realtime STT ws connected")

    scheduler = websocket.app.state.scheduler
    registry = websocket.app.state.registry
    store = websocket.app.state.store

    session = StreamingTranscriptionSession(
        scheduler=scheduler, registry=registry, store=store,
    )
    disconnected = False

    async def emit_events() -> None:
        async for event in session.events():
            wire = _event_to_wire(event)
            if wire is not None:
                with suppress(Exception):
                    await websocket.send_json(wire)
            if isinstance(event, DoneEvent):
                return

    emit_task = asyncio.create_task(emit_events())

    try:
        while True:
            raw = await websocket.receive()

            if "text" in raw:
                data = json.loads(raw["text"])
                msg_type = data.get("type", "")

                if msg_type == "config":
                    try:
                        await session.configure(_config_from_message(data))
                    except OperationError as exc:
                        await session.report_error(_operation_error_message(exc))
                    continue

                if msg_type == "end":
                    break

                await session.report_error(str(UnknownMessageTypeError(msg_type)))
                continue

            if "bytes" in raw and raw["bytes"]:
                try:
                    await session.submit_pcm16(raw["bytes"])
                except SessionNotConfiguredError:
                    await session.report_error("Session not configured")

    except WebSocketDisconnect:
        disconnected = True
        logger.info("WS stream client disconnected")
    except Exception as exc:
        disconnected = True
        logger.exception("WS stream error")
        with suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        if not disconnected:
            await session.end_of_stream()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(emit_task, timeout=5.0)
        if not emit_task.done():
            emit_task.cancel()
            with suppress(Exception):
                await emit_task
        await session.close()
        with suppress(Exception):
            await websocket.close()
        logger.info("realtime STT ws closed")
        request_id_var.reset(token)
