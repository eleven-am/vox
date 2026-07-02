from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from vox.audio.codecs import encode_pcm, encode_wav_stream_header
from vox.audio.pipeline import get_content_type, prepare_for_output
from vox.core.adapter import TTSAdapter
from vox.core.errors import VoxError
from vox.core.types import SynthesizeChunk
from vox.operations.defaults import resolve_requested_or_default_model
from vox.operations.errors import (
    EmptyInputError,
    NoAudioGeneratedError,
    WrongModelTypeError,
)
from vox.operations.model_acquisition import acquire_typed_adapter
from vox.operations.tts_chunking import split_text_for_tts_adapter
from vox.operations.voice_resolution import resolve_tts_voice_request

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisRequest:
    input: str
    model: str = ""
    voice: str | None = None
    speed: float = 1.0
    language: str | None = None
    response_format: str = "wav"


def synthesis_request_from_fields(
    *,
    input: str,
    model: str = "",
    voice: str | None = None,
    speed: float = 1.0,
    language: str | None = None,
    response_format: str | None = "wav",
) -> SynthesisRequest:
    return SynthesisRequest(
        input=input,
        model=model,
        voice=voice or None,
        speed=speed if speed > 0 else 1.0,
        language=language or None,
        response_format=(response_format or "wav").lower(),
    )


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


@dataclass(frozen=True)
class SynthesisAudioResponse:
    chunks: AsyncIterator[bytes]
    content_type: str
    response_format: str

    @property
    def filename(self) -> str:
        return f"speech.{self.response_format}"


@dataclass(frozen=True)
class _TtsSynthesisContext:
    adapter: TTSAdapter
    voice: str | None
    language: str | None
    reference_audio: bytes | None
    reference_text: str | None
    text_chunks: tuple[str, ...]


def _split_for_adapter(text: str, adapter: TTSAdapter) -> list[str]:
    return split_text_for_tts_adapter(text, adapter)


def _resolve_model(registry: Any, store: Any | None, requested: str) -> str:
    return resolve_requested_or_default_model("tts", requested, registry, store)


def _validate_input(text: str) -> None:
    if not text or not text.strip():
        raise EmptyInputError()


@asynccontextmanager
async def _acquire_tts_synthesis_context(
    *,
    scheduler: Any,
    store: Any,
    model: str,
    request: SynthesisRequest,
):
    async with acquire_typed_adapter(
        scheduler,
        model=model,
        adapter_type=TTSAdapter,
        expected_type="TTS",
    ) as adapter:
        voice, language, reference_audio, reference_text = resolve_tts_voice_request(
            adapter, store, request.voice, request.language
        )
        yield _TtsSynthesisContext(
            adapter=adapter,
            voice=voice,
            language=language,
            reference_audio=reference_audio,
            reference_text=reference_text,
            text_chunks=tuple(_split_for_adapter(request.input, adapter)),
        )


async def _iter_tts_synthesis_chunks(
    context: _TtsSynthesisContext,
    request: SynthesisRequest,
) -> AsyncIterator[tuple[int, SynthesizeChunk]]:
    for idx, text_chunk in enumerate(context.text_chunks):
        async for chunk in context.adapter.synthesize(
            text_chunk,
            voice=context.voice,
            speed=request.speed if request.speed > 0 else 1.0,
            language=context.language,
            reference_audio=context.reference_audio,
            reference_text=context.reference_text,
        ):
            yield idx, chunk


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
        async with _acquire_tts_synthesis_context(
            scheduler=scheduler,
            store=store,
            model=model,
            request=request,
        ) as context:
            chunks: list[np.ndarray] = []
            sample_rate = 0
            async for _, chunk in _iter_tts_synthesis_chunks(context, request):
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
    except (WrongModelTypeError, NoAudioGeneratedError, VoxError):
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


async def preflight_synthesis(
    *,
    scheduler: Any,
    registry: Any,
    store: Any,
    request: SynthesisRequest,
) -> None:
    model = _resolve_model(registry, store, request.model)
    _validate_input(request.input)
    async with _acquire_tts_synthesis_context(
        scheduler=scheduler,
        store=store,
        model=model,
        request=request,
    ):
        pass


async def synthesize_stream(
    *,
    scheduler: Any,
    registry: Any,
    store: Any,
    request: SynthesisRequest,
) -> AsyncIterator[bytes]:
    return await synthesize_incremental(
        scheduler=scheduler,
        registry=registry,
        store=store,
        request=request,
    )


async def synthesize_audio_response(
    *,
    scheduler: Any,
    registry: Any,
    store: Any,
    request: SynthesisRequest,
    stream: bool = False,
) -> SynthesisAudioResponse:
    response_format = request.response_format
    if stream:
        await preflight_synthesis(
            scheduler=scheduler,
            registry=registry,
            store=store,
            request=request,
        )
        chunks = await synthesize_stream(
            scheduler=scheduler,
            registry=registry,
            store=store,
            request=request,
        )
        return SynthesisAudioResponse(
            chunks=chunks,
            content_type=stream_content_type(response_format),
            response_format=response_format,
        )

    if supports_incremental_output(response_format):
        await preflight_synthesis(
            scheduler=scheduler,
            registry=registry,
            store=store,
            request=request,
        )
        chunks = await synthesize_incremental(
            scheduler=scheduler,
            registry=registry,
            store=store,
            request=request,
        )
        return SynthesisAudioResponse(
            chunks=chunks,
            content_type=stream_content_type(response_format),
            response_format=response_format,
        )

    bundle = await synthesize_full(
        scheduler=scheduler,
        registry=registry,
        store=store,
        request=request,
    )

    async def _single() -> AsyncIterator[bytes]:
        yield bundle.audio

    return SynthesisAudioResponse(
        chunks=_single(),
        content_type=bundle.content_type,
        response_format=bundle.response_format,
    )


def supports_incremental_output(response_format: str) -> bool:
    return response_format.lower() in {"wav", "pcm", "mp3"}


async def synthesize_incremental(
    *,
    scheduler: Any,
    registry: Any,
    store: Any,
    request: SynthesisRequest,
) -> AsyncIterator[bytes]:
    model = _resolve_model(registry, store, request.model)
    _validate_input(request.input)
    fmt = request.response_format.lower()

    if not supports_incremental_output(fmt):
        bundle = await synthesize_full(
            scheduler=scheduler,
            registry=registry,
            store=store,
            request=request,
        )

        async def _single() -> AsyncIterator[bytes]:
            yield bundle.audio

        return _single()

    async def _gen() -> AsyncIterator[bytes]:
        start_time = time.perf_counter()
        audio_samples = 0
        sample_rate = 0
        yielded_audio = False
        wav_header_sent = False
        mp3_encoder = None

        try:
            async with _acquire_tts_synthesis_context(
                scheduler=scheduler,
                store=store,
                model=model,
                request=request,
            ) as context:
                async for _, chunk in _iter_tts_synthesis_chunks(context, request):
                    audio_data = np.frombuffer(chunk.audio, dtype=np.float32)
                    if sample_rate <= 0:
                        sample_rate = chunk.sample_rate
                    if audio_data.size <= 0:
                        continue

                    yielded_audio = True
                    audio_samples += int(audio_data.size)

                    if fmt == "wav":
                        if not wav_header_sent:
                            yield encode_wav_stream_header(chunk.sample_rate)
                            wav_header_sent = True
                        yield encode_pcm(audio_data)
                    elif fmt == "pcm":
                        yield encode_pcm(audio_data)
                    elif fmt == "mp3":
                        if mp3_encoder is None:
                            import lameenc

                            mp3_encoder = lameenc.Encoder()
                            mp3_encoder.set_bit_rate(128)
                            mp3_encoder.set_in_sample_rate(chunk.sample_rate)
                            mp3_encoder.set_channels(1)
                            mp3_encoder.set_quality(2)
                        encoded = mp3_encoder.encode(encode_pcm(audio_data))
                        if encoded:
                            yield bytes(encoded)

                if mp3_encoder is not None:
                    tail = mp3_encoder.flush()
                    if tail:
                        yield bytes(tail)

                if not yielded_audio:
                    raise NoAudioGeneratedError()
        except (WrongModelTypeError, NoAudioGeneratedError, VoxError):
            raise
        except Exception:
            logger.exception(f"Synthesis failed for model {model}")
            raise
        finally:
            if yielded_audio:
                processing_ms = int((time.perf_counter() - start_time) * 1000)
                audio_ms = int(1000 * audio_samples / max(sample_rate, 1))
                logger.info(
                    "synthesize_streamed %s chars=%d audio_ms=%d processing_ms=%d format=%s",
                    model,
                    len(request.input or ""),
                    audio_ms,
                    processing_ms,
                    request.response_format,
                )

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
        async with _acquire_tts_synthesis_context(
            scheduler=scheduler,
            store=store,
            model=model,
            request=request,
        ) as context:
            async for idx, chunk in _iter_tts_synthesis_chunks(context, request):
                is_last_text_chunk = idx == len(context.text_chunks) - 1
                yield SynthesisRawChunk(
                    audio=chunk.audio,
                    sample_rate=chunk.sample_rate,
                    is_final=chunk.is_final and is_last_text_chunk,
                )

    return _gen()
