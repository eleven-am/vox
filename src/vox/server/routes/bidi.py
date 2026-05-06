from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vox.conversation.text_buffer import (
    split_by_chars,
    split_by_words,
    split_clauses,
    split_for_tts,
    split_long_sentence,
    split_sentences,
)
from vox.core.errors import VoiceCloningUnsupportedError, VoiceNotFoundError
from vox.logging_context import new_request_id, request_id_var
from vox.operations.errors import (
    OperationError,
    SessionNotConfiguredError,
    UnknownMessageTypeError,
)
from vox.operations.streaming_synthesis_longform import (
    LongformSynthesisSession,
    TtsAudioChunkEvent,
    TtsAudioStartEvent,
    TtsDoneEvent,
    TtsErrorEvent,
    TtsProgressEvent,
    TtsReadyEvent,
    normalize_longform_tts_config,
)
from vox.operations.streaming_transcription_longform import (
    LongformDoneEvent,
    LongformErrorEvent,
    LongformProgressEvent,
    LongformReadyEvent,
    LongformTranscriptionSession,
    normalize_longform_config,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/v1/audio/transcriptions/stream")
async def transcriptions_stream(websocket: WebSocket):
    await websocket.accept()
    incoming = websocket.headers.get("x-request-id")
    rid = incoming.strip() if incoming and incoming.strip() else new_request_id()
    token = request_id_var.set(rid)
    logger.info("long-form STT ws connected")

    try:
        config_msg = await _receive_config(websocket, "audio")
        if config_msg is None:
            return
        try:
            config = normalize_longform_config(
                model=config_msg.get("model", "") or "",
                sample_rate=int(config_msg.get("sample_rate") or 0) or None,
                input_format=config_msg.get("input_format"),
                language=config_msg.get("language"),
                word_timestamps=bool(config_msg.get("word_timestamps", False)),
                temperature=float(config_msg.get("temperature", 0.0) or 0.0),
                chunk_ms=config_msg.get("chunk_ms"),
                overlap_ms=config_msg.get("overlap_ms"),
                registry=websocket.app.state.registry,
                store=websocket.app.state.store,
            )
        except OperationError as exc:
            await _send_error(websocket, str(exc))
            return

        scheduler = websocket.app.state.scheduler
        session = LongformTranscriptionSession(
            scheduler=scheduler,
            registry=websocket.app.state.registry,
            store=websocket.app.state.store,
        )
        emit_task = asyncio.create_task(_emit_longform_stt_events(websocket, session))
        try:
            try:
                await session.configure(config)
            except OperationError as exc:
                await _send_error(websocket, str(exc))
                return

            while True:
                raw = await websocket.receive()
                if raw.get("type") == "websocket.disconnect":
                    return
                if "text" in raw:
                    data = json.loads(raw["text"])
                    msg_type = data.get("type", "")
                    if msg_type == "end":
                        break
                    await _send_error(websocket, str(UnknownMessageTypeError(msg_type)))
                    continue
                if "bytes" in raw and raw["bytes"]:
                    try:
                        await session.submit_chunk(raw["bytes"])
                    except SessionNotConfiguredError as exc:
                        await _send_error(websocket, str(exc))

            await session.end_of_stream()
        finally:
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(emit_task, timeout=5.0)
            if not emit_task.done():
                emit_task.cancel()
                with suppress(Exception):
                    await emit_task
            await session.close()
    except WebSocketDisconnect:
        logger.info("Long-form STT websocket disconnected")
    except Exception as exc:
        logger.exception("Long-form STT websocket error")
        await _safe_send_error(websocket, str(exc))
    finally:
        await _safe_close(websocket)
        logger.info("long-form STT ws closed")
        request_id_var.reset(token)


@router.websocket("/v1/audio/speech/stream")
async def speech_stream(websocket: WebSocket):
    await websocket.accept()
    incoming = websocket.headers.get("x-request-id")
    rid = incoming.strip() if incoming and incoming.strip() else new_request_id()
    token = request_id_var.set(rid)
    logger.info("long-form TTS ws connected")

    try:
        config_msg = await _receive_config(websocket, "text input")
        if config_msg is None:
            return
        try:
            config = normalize_longform_tts_config(
                model=config_msg.get("model", "") or "",
                voice=config_msg.get("voice"),
                speed=float(config_msg.get("speed", 1.0) or 1.0),
                language=config_msg.get("language"),
                response_format=config_msg.get("response_format"),
                chunk_chars=config_msg.get("chunk_chars"),
                registry=websocket.app.state.registry,
                store=websocket.app.state.store,
            )
        except OperationError as exc:
            await _send_error(websocket, str(exc))
            return

        scheduler = websocket.app.state.scheduler
        session = LongformSynthesisSession(
            scheduler=scheduler,
            registry=websocket.app.state.registry,
            store=websocket.app.state.store,
        )
        emit_task = asyncio.create_task(_emit_longform_tts_events(websocket, session))
        try:
            try:
                await session.configure(config)
            except (VoiceCloningUnsupportedError, VoiceNotFoundError) as exc:
                await _send_error(websocket, str(exc))
                return
            except OperationError as exc:
                await _send_error(websocket, str(exc))
                return

            while True:
                raw = await websocket.receive()
                if raw.get("type") == "websocket.disconnect":
                    return
                if "text" not in raw:
                    await _send_error(websocket, "Binary messages are not supported for TTS input")
                    continue
                data = json.loads(raw["text"])
                msg_type = data.get("type", "")
                if msg_type == "text":
                    chunk = data.get("text", "")
                    if chunk:
                        session.append_text(chunk)
                    continue
                if msg_type == "end":
                    break
                await _send_error(websocket, str(UnknownMessageTypeError(msg_type)))

            await session.end_of_stream()
        finally:
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(emit_task, timeout=10.0)
            if not emit_task.done():
                emit_task.cancel()
                with suppress(Exception):
                    await emit_task
            await session.close()
    except WebSocketDisconnect:
        logger.info("Long-form TTS websocket disconnected")
    except Exception as exc:
        logger.exception("Long-form TTS websocket error")
        await _safe_send_error(websocket, str(exc))
    finally:
        await _safe_close(websocket)
        logger.info("long-form TTS ws closed")
        request_id_var.reset(token)


async def _receive_config(websocket: WebSocket, downstream: str) -> dict | None:
    while True:
        raw = await websocket.receive()
        if raw.get("type") == "websocket.disconnect":
            return None
        if "text" not in raw:
            await _send_error(websocket, f"Configuration message required before {downstream}")
            continue
        data = json.loads(raw["text"])
        if data.get("type") != "config":
            await _send_error(websocket, f"Configuration message required before {downstream}")
            continue
        return data


async def _emit_longform_stt_events(websocket: WebSocket, session: LongformTranscriptionSession) -> None:
    async for event in session.events():
        if isinstance(event, LongformReadyEvent):
            await websocket.send_json({
                "type": "ready",
                "model": event.model,
                "sample_rate": event.sample_rate,
                "input_format": event.input_format,
                "chunk_ms": event.chunk_ms,
                "overlap_ms": event.overlap_ms,
            })
        elif isinstance(event, LongformProgressEvent):
            await websocket.send_json({
                "type": "progress",
                "uploaded_ms": event.uploaded_ms,
                "processed_ms": event.processed_ms,
                "chunks_completed": event.chunks_completed,
            })
        elif isinstance(event, LongformDoneEvent):
            await websocket.send_json({
                "type": "done",
                "model": event.model,
                "text": event.text,
                "language": event.language,
                "duration_ms": event.duration_ms,
                "processing_ms": event.processing_ms,
                "segments": list(event.segments),
            })
            return
        elif isinstance(event, LongformErrorEvent):
            await _send_error(websocket, event.message)
            return


async def _emit_longform_tts_events(websocket: WebSocket, session: LongformSynthesisSession) -> None:
    async for event in session.events():
        if isinstance(event, TtsReadyEvent):
            await websocket.send_json({
                "type": "ready",
                "model": event.model,
                "voice": event.voice,
                "response_format": event.response_format,
                "chunk_chars": event.chunk_chars,
            })
        elif isinstance(event, TtsAudioStartEvent):
            await websocket.send_json({
                "type": "audio_start",
                "sample_rate": event.sample_rate,
                "response_format": event.response_format,
            })
        elif isinstance(event, TtsAudioChunkEvent):
            await websocket.send_bytes(event.data)
        elif isinstance(event, TtsProgressEvent):
            await websocket.send_json({
                "type": "progress",
                "completed_chars": event.completed_chars,
                "total_chars": event.total_chars,
                "chunks_completed": event.chunks_completed,
                "chunks_total": event.chunks_total,
            })
        elif isinstance(event, TtsDoneEvent):
            await websocket.send_json({
                "type": "done",
                "response_format": event.response_format,
                "audio_duration_ms": event.audio_duration_ms,
                "processing_ms": event.processing_ms,
                "text_length": event.text_length,
            })
            return
        elif isinstance(event, TtsErrorEvent):
            await _send_error(websocket, event.message)
            return


_SENTENCE_TERMINATORS = frozenset(".!?。！？．।؟")
_CLAUSE_TERMINATORS = frozenset(",;:，、；：")

_split_sentences = split_sentences
_split_clauses = split_clauses
_chunk_text = split_for_tts
_split_long_sentence = split_long_sentence
_split_by_words = split_by_words
_split_by_chars = split_by_chars


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


async def _safe_send_error(websocket: WebSocket, message: str) -> None:
    with suppress(Exception):
        await _send_error(websocket, message)


async def _safe_close(websocket: WebSocket) -> None:
    with suppress(Exception):
        await websocket.close()
