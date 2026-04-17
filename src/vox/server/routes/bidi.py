from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from vox.audio.pipeline import prepare_for_stt
from vox.conversation.text_buffer import (
    split_by_chars,
    split_by_words,
    split_clauses,
    split_for_tts,
    split_long_sentence,
    split_sentences,
)
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.cloned_voices import resolve_voice_request
from vox.core.errors import VoiceCloningUnsupportedError, VoiceNotFoundError
from vox.logging_context import new_request_id, request_id_var
from vox.server.routes import get_default_model
from vox.streaming.codecs import float32_to_pcm16, pcm16_to_float32, resample_audio
from vox.streaming.mp3 import Mp3StreamEncoder
from vox.streaming.opus import OpusStreamEncoder
from vox.streaming.types import TARGET_SAMPLE_RATE, samples_to_ms

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_LONGFORM_CHUNK_MS = 30_000
DEFAULT_LONGFORM_OVERLAP_MS = 1_000
MAX_LONGFORM_CHUNK_MS = 120_000
MAX_LONGFORM_OVERLAP_MS = 10_000
SUPPORTED_TTS_STREAM_FORMATS = {"pcm16", "opus", "mp3"}
SUPPORTED_STT_INPUT_FORMATS = {"pcm16", "wav", "flac", "mp3", "ogg", "webm"}


@dataclass
class LongFormTranscribeConfig:
    model: str
    sample_rate: int
    input_format: str
    language: str | None
    word_timestamps: bool
    temperature: float
    chunk_ms: int
    overlap_ms: int


@dataclass
class LongFormTranscribeState:
    chunk_samples: int
    overlap_samples: int
    pending_audio: np.ndarray
    next_chunk_start_samples: int = 0
    committed_samples: int = 0
    uploaded_samples: int = 0
    processing_ms: int = 0
    chunks_completed: int = 0
    transcript_parts: list[str] | None = None
    segments: list[dict] | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if self.transcript_parts is None:
            self.transcript_parts = []
        if self.segments is None:
            self.segments = []


@router.websocket("/v1/audio/transcriptions/stream")
async def transcriptions_stream(websocket: WebSocket):
    await websocket.accept()
    incoming = websocket.headers.get("x-request-id")
    rid = incoming.strip() if incoming and incoming.strip() else new_request_id()
    token = request_id_var.set(rid)
    logger.info("long-form STT ws connected")
    config: LongFormTranscribeConfig | None = None

    try:
        config = await _receive_stt_config(websocket)
        if config is None:
            return

        scheduler = websocket.app.state.scheduler
        async with scheduler.acquire(config.model) as adapter:
            if not isinstance(adapter, STTAdapter):
                await _send_error(websocket, f"Model '{config.model}' is not an STT model")
                return

            await websocket.send_json({
                "type": "ready",
                "model": config.model,
                "sample_rate": config.sample_rate,
                "input_format": config.input_format,
                "chunk_ms": config.chunk_ms,
                "overlap_ms": config.overlap_ms,
            })

            state = LongFormTranscribeState(
                chunk_samples=int(config.chunk_ms * TARGET_SAMPLE_RATE / 1000),
                overlap_samples=int(config.overlap_ms * TARGET_SAMPLE_RATE / 1000),
                pending_audio=np.array([], dtype=np.float32),
            )

            while True:
                raw = await websocket.receive()

                if raw.get("type") == "websocket.disconnect":
                    return

                if "text" in raw:
                    data = json.loads(raw["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "end":
                        break

                    await _send_error(websocket, f"Unknown message type: {msg_type}")
                    continue

                if "bytes" in raw and raw["bytes"]:
                    audio = _decode_stt_chunk(raw["bytes"], config)
                    await _append_transcribe_audio(websocket, adapter, config, state, audio)

            if state.uploaded_samples == 0:
                await _send_error(websocket, "No audio data provided")
                return

            await _flush_transcribe_audio(adapter, config, state)
            await _send_transcribe_done(websocket, config, state)
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
        config = await _receive_tts_config(websocket)
        if config is None:
            return

        scheduler = websocket.app.state.scheduler
        store = websocket.app.state.store
        async with scheduler.acquire(config["model"]) as adapter:
            if not isinstance(adapter, TTSAdapter):
                await _send_error(websocket, f"Model '{config['model']}' is not a TTS model")
                return

            try:
                voice_arg, language_arg, reference_audio, reference_text = resolve_voice_request(
                    adapter, store, config["voice"], config["language"],
                )
            except VoiceCloningUnsupportedError as exc:
                await _send_error(websocket, str(exc))
                return
            except VoiceNotFoundError as exc:
                await _send_error(websocket, str(exc))
                return


            override = config.get("chunk_chars")
            adapter_cap = int(getattr(adapter.info(), "max_input_chars", 0) or 0)
            effective_cap = override if override is not None else adapter_cap

            await websocket.send_json({
                "type": "ready",
                "model": config["model"],
                "voice": config["voice"],
                "response_format": config["response_format"],
                "chunk_chars": effective_cap,
            })

            text_parts: list[str] = []
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
                        text_parts.append(chunk)
                    continue

                if msg_type == "end":
                    break

                await _send_error(websocket, f"Unknown message type: {msg_type}")

            full_text = "".join(text_parts).strip()
            if not full_text:
                await _send_error(websocket, "No input text provided")
                return

            text_chunks = [full_text] if effective_cap <= 0 else split_for_tts(full_text, max_chars=effective_cap)
            total_chars = sum(len(chunk) for chunk in text_chunks)
            completed_chars = 0
            completed_chunks = 0
            total_audio_samples = 0
            total_processing_ms = 0
            audio_meta_sent = False
            opus_encoder: OpusStreamEncoder | None = None
            mp3_encoder: Mp3StreamEncoder | None = None
            output_sample_rate = 0

            for text_chunk in text_chunks:
                chunk_start = time.perf_counter()
                async for chunk in adapter.synthesize(
                    text_chunk,
                    voice=voice_arg,
                    speed=config["speed"],
                    language=language_arg,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                ):
                    audio = np.frombuffer(chunk.audio, dtype=np.float32)
                    if audio.size == 0:
                        continue

                    total_audio_samples += audio.size
                    output_sample_rate = chunk.sample_rate
                    if not audio_meta_sent:
                        await websocket.send_json({
                            "type": "audio_start",
                            "sample_rate": chunk.sample_rate,
                            "response_format": config["response_format"],
                        })
                        audio_meta_sent = True

                    fmt = config["response_format"]
                    if fmt == "pcm16":
                        await websocket.send_bytes(float32_to_pcm16(audio))
                    elif fmt == "opus":
                        pcm16 = float32_to_pcm16(audio)
                        if opus_encoder is None:
                            opus_encoder = OpusStreamEncoder(source_rate=chunk.sample_rate)
                        for opus_frame in opus_encoder.encode(pcm16):
                            await websocket.send_bytes(opus_frame)
                    elif fmt == "mp3":
                        pcm16 = float32_to_pcm16(audio)
                        if mp3_encoder is None:
                            mp3_encoder = Mp3StreamEncoder(source_rate=chunk.sample_rate)
                        mp3_bytes = mp3_encoder.encode(pcm16)
                        if mp3_bytes:
                            await websocket.send_bytes(mp3_bytes)

                total_processing_ms += int((time.perf_counter() - chunk_start) * 1000)
                completed_chunks += 1
                completed_chars += len(text_chunk)
                await websocket.send_json({
                    "type": "progress",
                    "completed_chars": completed_chars,
                    "total_chars": total_chars,
                    "chunks_completed": completed_chunks,
                    "chunks_total": len(text_chunks),
                })

            if opus_encoder is not None:
                for opus_frame in opus_encoder.flush():
                    await websocket.send_bytes(opus_frame)

            if mp3_encoder is not None:
                tail = mp3_encoder.flush()
                if tail:
                    await websocket.send_bytes(tail)

            default_done_rate = 48_000 if config["response_format"] == "opus" else 24_000
            await websocket.send_json({
                "type": "done",
                "response_format": config["response_format"],
                "audio_duration_ms": samples_to_ms(
                    total_audio_samples,
                    output_sample_rate or default_done_rate,
                ),
                "processing_ms": total_processing_ms,
                "text_length": total_chars,
            })
    except WebSocketDisconnect:
        logger.info("Long-form TTS websocket disconnected")
    except Exception as exc:
        logger.exception("Long-form TTS websocket error")
        await _safe_send_error(websocket, str(exc))
    finally:
        await _safe_close(websocket)
        logger.info("long-form TTS ws closed")
        request_id_var.reset(token)


async def _receive_stt_config(websocket: WebSocket) -> LongFormTranscribeConfig | None:
    registry = websocket.app.state.registry
    store = websocket.app.state.store

    while True:
        raw = await websocket.receive()
        if raw.get("type") == "websocket.disconnect":
            return None

        if "text" not in raw:
            await _send_error(websocket, "Configuration message required before audio")
            continue

        data = json.loads(raw["text"])
        if data.get("type") != "config":
            await _send_error(websocket, "Configuration message required before audio")
            continue

        try:
            model = data.get("model", "") or get_default_model("stt", registry, store)
        except HTTPException as exc:
            await _send_error(websocket, str(exc.detail))
            return None
        input_format = str(data.get("input_format", "pcm16")).lower()
        if input_format not in SUPPORTED_STT_INPUT_FORMATS:
            await _send_error(
                websocket,
                (
                    f"Unsupported input_format '{input_format}'. "
                    f"Supported values: {sorted(SUPPORTED_STT_INPUT_FORMATS)}"
                ),
            )
            return None
        sample_rate = int(data.get("sample_rate") or TARGET_SAMPLE_RATE)
        chunk_ms = _clamp_int(data.get("chunk_ms"), DEFAULT_LONGFORM_CHUNK_MS, 1_000, MAX_LONGFORM_CHUNK_MS)
        overlap_ms = _clamp_int(data.get("overlap_ms"), DEFAULT_LONGFORM_OVERLAP_MS, 0, MAX_LONGFORM_OVERLAP_MS)
        if overlap_ms >= chunk_ms:
            await _send_error(websocket, "overlap_ms must be smaller than chunk_ms")
            return None

        if input_format == "pcm16" and sample_rate <= 0:
            await _send_error(websocket, "sample_rate must be positive")
            return None

        return LongFormTranscribeConfig(
            model=model,
            sample_rate=sample_rate,
            input_format=input_format,
            language=data.get("language"),
            word_timestamps=bool(data.get("word_timestamps", False)),
            temperature=float(data.get("temperature", 0.0)),
            chunk_ms=chunk_ms,
            overlap_ms=overlap_ms,
        )


async def _receive_tts_config(websocket: WebSocket) -> dict[str, object] | None:
    registry = websocket.app.state.registry
    store = websocket.app.state.store

    while True:
        raw = await websocket.receive()
        if raw.get("type") == "websocket.disconnect":
            return None

        if "text" not in raw:
            await _send_error(websocket, "Configuration message required before text input")
            continue

        data = json.loads(raw["text"])
        if data.get("type") != "config":
            await _send_error(websocket, "Configuration message required before text input")
            continue

        try:
            model = data.get("model", "") or get_default_model("tts", registry, store)
        except HTTPException as exc:
            await _send_error(websocket, str(exc.detail))
            return None
        response_format = str(data.get("response_format", "pcm16")).lower()
        if response_format not in SUPPORTED_TTS_STREAM_FORMATS:
            await _send_error(
                websocket,
                (
                    f"Unsupported response_format '{response_format}'. "
                    f"Supported values: {sorted(SUPPORTED_TTS_STREAM_FORMATS)}"
                ),
            )
            return None

        chunk_chars_raw = data.get("chunk_chars")
        chunk_chars: int | None
        if chunk_chars_raw is None:
            chunk_chars = None
        else:
            try:
                chunk_chars = max(0, int(chunk_chars_raw))
            except (TypeError, ValueError):
                await _send_error(websocket, "chunk_chars must be a non-negative integer")
                return None

        return {
            "model": model,
            "voice": data.get("voice"),
            "speed": float(data.get("speed", 1.0)),
            "language": data.get("language"),
            "response_format": response_format,
            "chunk_chars": chunk_chars,
        }


async def _append_transcribe_audio(
    websocket: WebSocket,
    adapter: STTAdapter,
    config: LongFormTranscribeConfig,
    state: LongFormTranscribeState,
    audio: np.ndarray,
) -> None:
    if audio.size == 0:
        return

    state.uploaded_samples += audio.size
    if state.pending_audio.size == 0:
        state.pending_audio = audio
    else:
        state.pending_audio = np.concatenate([state.pending_audio, audio])

    step_samples = state.chunk_samples - state.overlap_samples
    while state.pending_audio.size >= state.chunk_samples:
        chunk_audio = state.pending_audio[:state.chunk_samples]
        await _run_transcribe_chunk(adapter, config, state, chunk_audio, final_chunk=False)
        state.pending_audio = state.pending_audio[step_samples:]
        state.next_chunk_start_samples += step_samples
        state.committed_samples += step_samples
        await websocket.send_json({
            "type": "progress",
            "uploaded_ms": samples_to_ms(state.uploaded_samples),
            "processed_ms": samples_to_ms(state.committed_samples),
            "chunks_completed": state.chunks_completed,
        })


async def _flush_transcribe_audio(
    adapter: STTAdapter,
    config: LongFormTranscribeConfig,
    state: LongFormTranscribeState,
) -> None:
    if state.pending_audio.size == 0:
        return

    if state.chunks_completed > 0 and state.pending_audio.size <= state.overlap_samples:
        return

    await _run_transcribe_chunk(adapter, config, state, state.pending_audio, final_chunk=True)
    state.committed_samples = state.uploaded_samples


async def _run_transcribe_chunk(
    adapter: STTAdapter,
    config: LongFormTranscribeConfig,
    state: LongFormTranscribeState,
    chunk_audio: np.ndarray,
    *,
    final_chunk: bool,
) -> None:
    start_time = time.perf_counter()
    result = await asyncio.to_thread(
        adapter.transcribe,
        chunk_audio,
        language=config.language,
        word_timestamps=config.word_timestamps,
        temperature=config.temperature,
    )
    state.processing_ms += int((time.perf_counter() - start_time) * 1000)
    if state.language is None and result.language:
        state.language = result.language

    overlap_ms = 0 if state.chunks_completed == 0 else config.overlap_ms
    chunk_start_ms = samples_to_ms(state.next_chunk_start_samples)
    if result.segments:
        for segment in result.segments:
            if overlap_ms and segment.end_ms <= overlap_ms:
                continue

            start_ms = chunk_start_ms + max(segment.start_ms, overlap_ms)
            end_ms = chunk_start_ms + segment.end_ms
            words = []
            for word in segment.words:
                if overlap_ms and word.end_ms <= overlap_ms:
                    continue
                words.append({
                    "word": word.word,
                    "start_ms": chunk_start_ms + max(word.start_ms, overlap_ms),
                    "end_ms": chunk_start_ms + word.end_ms,
                    "confidence": word.confidence,
                })

            text = segment.text.strip()
            if text:
                state.transcript_parts.append(text)
            state.segments.append({
                "text": segment.text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "words": words,
            })
    else:
        text = result.text.strip()
        if text:
            state.transcript_parts.append(text)

    state.chunks_completed += 1


async def _send_transcribe_done(
    websocket: WebSocket,
    config: LongFormTranscribeConfig,
    state: LongFormTranscribeState,
) -> None:
    await websocket.send_json({
        "type": "done",
        "model": config.model,
        "text": " ".join(part for part in state.transcript_parts if part).strip(),
        "language": state.language,
        "duration_ms": samples_to_ms(state.uploaded_samples),
        "processing_ms": state.processing_ms,
        "segments": state.segments,
    })







_SENTENCE_TERMINATORS = frozenset(".!?。！？．।؟")
_CLAUSE_TERMINATORS = frozenset(",;:，、；：")

_split_sentences = split_sentences
_split_clauses = split_clauses
_chunk_text = split_for_tts
_split_long_sentence = split_long_sentence
_split_by_words = split_by_words
_split_by_chars = split_by_chars


def _clamp_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    return max(minimum, min(parsed, maximum))


def _decode_stt_chunk(chunk: bytes, config: LongFormTranscribeConfig) -> np.ndarray:
    if config.input_format == "pcm16":
        audio = pcm16_to_float32(chunk)
        if config.sample_rate != TARGET_SAMPLE_RATE:
            audio = resample_audio(audio, config.sample_rate, TARGET_SAMPLE_RATE)
        return audio




    return prepare_for_stt(chunk, format_hint=config.input_format)


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


async def _safe_send_error(websocket: WebSocket, message: str) -> None:
    with suppress(Exception):
        await _send_error(websocket, message)


async def _safe_close(websocket: WebSocket) -> None:
    with suppress(Exception):
        await websocket.close()
