from __future__ import annotations

import json
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vox.streaming.codecs import pcm16_to_float32, resample_audio
from vox.streaming.partials import PartialTranscriptService
from vox.streaming.pipeline import StreamPipeline
from vox.streaming.session import SpeechSession
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    SpeechStarted,
    SpeechStopped,
    StreamSessionConfig,
    StreamTranscript,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_default_stt(registry, store) -> str:
    for m in store.list_models():
        if m.type.value == "stt":
            return m.full_name
    for name, tags in registry.available_models().items():
        for tag, entry in tags.items():
            if entry.get("type") == "stt":
                return f"{name}:{tag}"
    return ""


@router.websocket("/v1/audio/stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()

    scheduler = websocket.app.state.scheduler
    registry = websocket.app.state.registry
    store = websocket.app.state.store

    pipeline: StreamPipeline | None = None
    session_config: StreamSessionConfig | None = None
    session = SpeechSession()
    partial_service: PartialTranscriptService | None = None
    disconnected = False

    try:
        while True:
            raw = await websocket.receive()

            if "text" in raw:
                data = json.loads(raw["text"])
                msg_type = data.get("type", "")

                if msg_type == "config":
                    if session_config is not None:
                        await websocket.send_json({"type": "error", "message": "Session already configured"})
                        continue

                    model = data.get("model", "") or _get_default_stt(registry, store)
                    if not model:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "No STT model specified and no default STT model available",
                            }
                        )
                        continue

                    session_config = StreamSessionConfig(
                        language=data.get("language", "en"),
                        sample_rate=data.get("sample_rate", TARGET_SAMPLE_RATE),
                        model=model,
                        partials=data.get("partials", False),
                        partial_window_ms=data.get("partial_window_ms", 1500),
                        partial_stride_ms=data.get("partial_stride_ms", 700),
                        include_word_timestamps=data.get("include_word_timestamps", False),
                        temperature=data.get("temperature", 0.0),
                    )

                    pipeline = StreamPipeline(scheduler=scheduler)
                    pipeline.configure(session_config)

                    partial_service = PartialTranscriptService(
                        transcribe_async_fn=pipeline.transcribe_async,
                    )

                    logger.info("WS stream configured: model=%s, language=%s", model, session_config.language)
                    await websocket.send_json({"type": "ready"})
                    continue

                if msg_type == "end":
                    break

                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
                continue

            if "bytes" in raw and raw["bytes"]:
                if pipeline is None or session_config is None or partial_service is None:
                    await websocket.send_json({"type": "error", "message": "Session not configured"})
                    continue

                audio_bytes = raw["bytes"]
                audio = pcm16_to_float32(audio_bytes)
                src_rate = session_config.sample_rate
                if src_rate != TARGET_SAMPLE_RATE:
                    audio = resample_audio(audio, src_rate, TARGET_SAMPLE_RATE)

                session.append_audio(audio)
                async for event in pipeline.process_audio(audio):
                    await _send_event(websocket, event, session)

                if session.is_active() and session_config.partials:
                    try:
                        transcript = await partial_service.generate_partial_async(session, session_config)
                        if transcript:
                            await _send_transcript(websocket, transcript)
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        disconnected = True
        logger.info("WS stream client disconnected")
    except Exception as e:
        disconnected = True
        logger.exception("WS stream error")
        with suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if pipeline is not None and partial_service is not None and not disconnected:
            remaining = partial_service.flush_remaining_audio(session)
            if remaining is not None and len(remaining) > 0:
                try:
                    transcript = await pipeline.transcribe_async(audio=remaining)
                    if transcript.text.strip():
                        await _send_transcript(websocket, transcript)
                except Exception:
                    pass

        if pipeline is not None:
            pipeline.reset()
            pipeline.shutdown()

        with suppress(Exception):
            await websocket.close()


async def _send_event(websocket: WebSocket, event, session: SpeechSession) -> None:
    if isinstance(event, SpeechStarted):
        session.start_speech()
        await websocket.send_json({
            "type": "speech_started",
            "timestamp_ms": event.timestamp_ms,
        })
    elif isinstance(event, SpeechStopped):
        session.stop_speech()
        await websocket.send_json({
            "type": "speech_stopped",
            "timestamp_ms": event.timestamp_ms,
        })
    elif isinstance(event, StreamTranscript):
        await _send_transcript(websocket, event)


async def _send_transcript(websocket: WebSocket, transcript: StreamTranscript) -> None:
    msg = {
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
        msg["eou_probability"] = transcript.eou_probability
    await websocket.send_json(msg)
