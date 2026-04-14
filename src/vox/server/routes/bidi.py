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
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.server.routes import get_default_model
from vox.streaming.codecs import float32_to_pcm16, pcm16_to_float32, resample_audio
from vox.streaming.opus import OpusStreamEncoder
from vox.streaming.types import TARGET_SAMPLE_RATE, samples_to_ms

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_LONGFORM_CHUNK_MS = 30_000
DEFAULT_LONGFORM_OVERLAP_MS = 1_000
MAX_LONGFORM_CHUNK_MS = 120_000
MAX_LONGFORM_OVERLAP_MS = 10_000
DEFAULT_TTS_CHUNK_CHARS = 250
SUPPORTED_TTS_STREAM_FORMATS = {"pcm16", "opus"}
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


@router.websocket("/v1/audio/speech/stream")
async def speech_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        config = await _receive_tts_config(websocket)
        if config is None:
            return

        scheduler = websocket.app.state.scheduler
        async with scheduler.acquire(config["model"]) as adapter:
            if not isinstance(adapter, TTSAdapter):
                await _send_error(websocket, f"Model '{config['model']}' is not a TTS model")
                return

            await websocket.send_json({
                "type": "ready",
                "model": config["model"],
                "voice": config["voice"],
                "response_format": config["response_format"],
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

            text_chunks = _chunk_text(full_text, DEFAULT_TTS_CHUNK_CHARS)
            total_chars = sum(len(chunk) for chunk in text_chunks)
            completed_chars = 0
            completed_chunks = 0
            total_audio_samples = 0
            total_processing_ms = 0
            audio_meta_sent = False
            opus_encoder: OpusStreamEncoder | None = None
            output_sample_rate = 0

            for text_chunk in text_chunks:
                chunk_start = time.perf_counter()
                async for chunk in adapter.synthesize(
                    text_chunk,
                    voice=config["voice"],
                    speed=config["speed"],
                    language=config["language"],
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

                    if config["response_format"] == "pcm16":
                        await websocket.send_bytes(float32_to_pcm16(audio))
                    else:
                        pcm16 = float32_to_pcm16(audio)
                        if opus_encoder is None:
                            opus_encoder = OpusStreamEncoder(source_rate=chunk.sample_rate)
                        for opus_frame in opus_encoder.encode(pcm16):
                            await websocket.send_bytes(opus_frame)

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

            await websocket.send_json({
                "type": "done",
                "response_format": config["response_format"],
                "audio_duration_ms": samples_to_ms(
                    total_audio_samples,
                    output_sample_rate or (48_000 if config["response_format"] == "opus" else 24_000),
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

        return {
            "model": model,
            "voice": data.get("voice"),
            "speed": float(data.get("speed", 1.0)),
            "language": data.get("language"),
            "response_format": response_format,
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


def _chunk_text(text: str, max_chars: int) -> list[str]:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    ]
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(sentence) <= max_chars:
            current = sentence
            continue

        chunks.extend(_split_long_sentence(sentence, max_chars))

    if current:
        chunks.append(current)
    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"(?<=[,;:])\s+", sentence) if piece.strip()]
    if len(pieces) == 1:
        words = sentence.split()
        chunks: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word
        if current:
            chunks.append(current)
        return chunks

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current} {piece}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(piece) <= max_chars:
                current = piece
            else:
                chunks.extend(_split_long_sentence(piece, max_chars))
                current = ""
    if current:
        chunks.append(current)
    return chunks


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

    # Encoded input frames must be self-contained decodable chunks, such as
    # MediaRecorder blobs or other containerized slices, not arbitrary byte
    # ranges from a larger compressed file.
    return prepare_for_stt(chunk, format_hint=config.input_format)


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


async def _safe_send_error(websocket: WebSocket, message: str) -> None:
    with suppress(Exception):
        await _send_error(websocket, message)


async def _safe_close(websocket: WebSocket) -> None:
    with suppress(Exception):
        await websocket.close()
