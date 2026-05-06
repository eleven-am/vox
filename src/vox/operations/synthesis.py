from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from vox.audio.pipeline import get_content_type, prepare_for_output
from vox.conversation.text_buffer import split_for_tts
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import resolve_voice_request
from vox.core.errors import ModelNotFoundError, VoxError
from vox.operations.defaults import resolve_default_model
from vox.operations.errors import (
    EmptyInputError,
    NoAudioGeneratedError,
    NoDefaultModelError,
    WrongModelTypeError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisRequest:
    input: str
    model: str = ""
    voice: str | None = None
    speed: float = 1.0
    language: str | None = None
    response_format: str = "wav"


@dataclass(frozen=True)
class SynthesisFullResult:
    audio: bytes
    content_type: str
    sample_rate: int
    audio_ms: int
    processing_ms: int
    response_format: str


@dataclass(frozen=True)
class SynthesisRawChunk:
    audio: bytes
    sample_rate: int
    is_final: bool


def _split_for_adapter(text: str, adapter: TTSAdapter) -> list[str]:
    max_chars = int(getattr(adapter.info(), "max_input_chars", 0) or 0)
    if max_chars <= 0:
        return [text] if text.strip() else []
    return split_for_tts(text, max_chars=max_chars)


def _resolve_model(registry: Any, store: Any | None, requested: str) -> str:
    model = requested or resolve_default_model("tts", registry, store) or ""
    if not model:
        raise NoDefaultModelError("tts")
    return model


def _validate_input(text: str) -> None:
    if not text or not text.strip():
        raise EmptyInputError()


async def synthesize_full(
    *,
    scheduler: Any,
    registry: Any,
    store: Any,
    request: SynthesisRequest,
) -> SynthesisFullResult:
    model = _resolve_model(registry, store, request.model)
    _validate_input(request.input)

    start_time = time.perf_counter()
    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, TTSAdapter):
                raise WrongModelTypeError(model, "TTS")
            voice, language, reference_audio, reference_text = resolve_voice_request(
                adapter, store, request.voice, request.language
            )
            text_chunks = _split_for_adapter(request.input, adapter)
            chunks: list[np.ndarray] = []
            sample_rate = 0
            for text_chunk in text_chunks:
                async for chunk in adapter.synthesize(
                    text_chunk,
                    voice=voice,
                    speed=request.speed,
                    language=language,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                ):
                    audio_data = np.frombuffer(chunk.audio, dtype=np.float32)
                    if audio_data.size > 0:
                        chunks.append(audio_data)
                    sample_rate = chunk.sample_rate

            if not chunks:
                raise NoAudioGeneratedError()
            audio = np.concatenate(chunks)
            if audio.size == 0:
                raise NoAudioGeneratedError()

            encoded, content_type = prepare_for_output(audio, sample_rate, request.response_format)
    except (WrongModelTypeError, NoAudioGeneratedError, ModelNotFoundError, VoxError):
        raise
    except Exception:
        logger.exception(f"Synthesis failed for model {model}")
        raise

    processing_ms = int((time.perf_counter() - start_time) * 1000)
    audio_ms = int(1000 * audio.size / max(sample_rate, 1))
    logger.info(
        "synthesize %s chars=%d audio_ms=%d processing_ms=%d format=%s",
        model, len(request.input or ""), audio_ms, processing_ms, request.response_format,
    )
    return SynthesisFullResult(
        audio=encoded,
        content_type=content_type,
        sample_rate=sample_rate,
        audio_ms=audio_ms,
        processing_ms=processing_ms,
        response_format=request.response_format,
    )


async def synthesize_stream(
    *,
    scheduler: Any,
    registry: Any,
    store: Any,
    request: SynthesisRequest,
) -> AsyncIterator[bytes]:
    model = _resolve_model(registry, store, request.model)
    _validate_input(request.input)

    async def _gen() -> AsyncIterator[bytes]:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, TTSAdapter):
                raise WrongModelTypeError(model, "TTS")
            voice, language, reference_audio, reference_text = resolve_voice_request(
                adapter, store, request.voice, request.language
            )
            for text_chunk in _split_for_adapter(request.input, adapter):
                async for chunk in adapter.synthesize(
                    text_chunk,
                    voice=voice,
                    speed=request.speed,
                    language=language,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                ):
                    audio_data = np.frombuffer(chunk.audio, dtype=np.float32)
                    if audio_data.size > 0:
                        encoded, _ = prepare_for_output(audio_data, chunk.sample_rate, request.response_format)
                        yield encoded

    return _gen()


def stream_content_type(response_format: str) -> str:
    return get_content_type(response_format)


async def synthesize_raw(
    *,
    scheduler: Any,
    registry: Any,
    store: Any,
    request: SynthesisRequest,
) -> AsyncIterator[SynthesisRawChunk]:
    model = _resolve_model(registry, store, request.model)
    _validate_input(request.input)

    async def _gen() -> AsyncIterator[SynthesisRawChunk]:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, TTSAdapter):
                raise WrongModelTypeError(model, "TTS")
            voice, language, reference_audio, reference_text = resolve_voice_request(
                adapter, store, request.voice, request.language
            )
            text_chunks = _split_for_adapter(request.input, adapter)
            for idx, text_chunk in enumerate(text_chunks):
                is_last_text_chunk = idx == len(text_chunks) - 1
                async for chunk in adapter.synthesize(
                    text_chunk,
                    voice=voice,
                    speed=request.speed if request.speed > 0 else 1.0,
                    language=language,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                ):
                    yield SynthesisRawChunk(
                        audio=chunk.audio,
                        sample_rate=chunk.sample_rate,
                        is_final=chunk.is_final and is_last_text_chunk,
                    )

    return _gen()
