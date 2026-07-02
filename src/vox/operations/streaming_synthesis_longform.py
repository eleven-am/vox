from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from vox.core.adapter import TTSAdapter
from vox.operations.defaults import resolve_requested_or_default_model
from vox.operations.errors import (
    EmptyInputError,
    InvalidConfigError,
    OperationError,
    SessionAlreadyConfiguredError,
    SessionNotConfiguredError,
    UnsupportedFormatError,
)
from vox.operations.model_acquisition import (
    EnteredAdapter,
    enter_typed_adapter,
    release_entered_adapter,
    release_entered_adapter_suppressing,
)
from vox.operations.streaming_reporting import StreamingOperationErrorReporter
from vox.operations.tts_chunking import effective_tts_text_cap, split_text_for_tts_adapter
from vox.operations.voice_resolution import resolve_tts_voice_request
from vox.streaming.codecs import float32_to_pcm16
from vox.streaming.mp3 import Mp3StreamEncoder
from vox.streaming.opus import OpusStreamEncoder
from vox.streaming.types import samples_to_ms

logger = logging.getLogger(__name__)


SUPPORTED_LONGFORM_TTS_FORMATS = {"pcm16", "opus", "mp3"}


@dataclass(frozen=True)
class LongformSynthesisConfig:
    model: str = ""
    voice: str | None = None
    speed: float = 1.0
    language: str | None = None
    response_format: str = "pcm16"
    chunk_chars: int | None = None


@dataclass(frozen=True)
class TtsReadyEvent:
    model: str
    voice: str | None
    response_format: str
    chunk_chars: int


@dataclass(frozen=True)
class TtsAudioStartEvent:
    sample_rate: int
    response_format: str


@dataclass(frozen=True)
class TtsAudioChunkEvent:
    data: bytes


@dataclass(frozen=True)
class TtsProgressEvent:
    completed_chars: int
    total_chars: int
    chunks_completed: int
    chunks_total: int


@dataclass(frozen=True)
class TtsDoneEvent:
    response_format: str
    audio_duration_ms: int
    processing_ms: int
    text_length: int


@dataclass(frozen=True)
class TtsErrorEvent:
    message: str


@dataclass(frozen=True)
class _ResolvedTtsVoice:
    voice: str | None
    language: str | None
    reference_audio: bytes | None
    reference_text: str | None


TtsEvent = (
    TtsReadyEvent
    | TtsAudioStartEvent
    | TtsAudioChunkEvent
    | TtsProgressEvent
    | TtsDoneEvent
    | TtsErrorEvent
)


def longform_tts_event_payload(event: TtsEvent) -> dict[str, Any] | None:
    if isinstance(event, TtsReadyEvent):
        return {
            "type": "ready",
            "model": event.model,
            "voice": event.voice,
            "response_format": event.response_format,
            "chunk_chars": event.chunk_chars,
        }
    if isinstance(event, TtsAudioStartEvent):
        return {
            "type": "audio_start",
            "sample_rate": event.sample_rate,
            "response_format": event.response_format,
        }
    if isinstance(event, TtsProgressEvent):
        return {
            "type": "progress",
            "completed_chars": event.completed_chars,
            "total_chars": event.total_chars,
            "chunks_completed": event.chunks_completed,
            "chunks_total": event.chunks_total,
        }
    if isinstance(event, TtsDoneEvent):
        return {
            "type": "done",
            "response_format": event.response_format,
            "audio_duration_ms": event.audio_duration_ms,
            "processing_ms": event.processing_ms,
            "text_length": event.text_length,
        }
    if isinstance(event, TtsErrorEvent):
        return {"type": "error", "message": event.message}
    return None


def normalize_longform_tts_config(
    *,
    model: str,
    voice: str | None,
    speed: float,
    language: str | None,
    response_format: str | None,
    chunk_chars: object,
    registry: Any,
    store: Any | None,
) -> LongformSynthesisConfig:
    resolved_model = resolve_requested_or_default_model("tts", model, registry, store)

    fmt = (response_format or "pcm16").lower()
    if fmt not in SUPPORTED_LONGFORM_TTS_FORMATS:
        raise UnsupportedFormatError("response_format", fmt, sorted(SUPPORTED_LONGFORM_TTS_FORMATS))

    cap: int | None
    if chunk_chars in (None, ""):
        cap = None
    else:
        try:
            cap = max(0, int(chunk_chars))
        except (TypeError, ValueError) as exc:
            raise InvalidConfigError("chunk_chars must be a non-negative integer") from exc

    resolved_speed = float(speed or 1.0)
    if resolved_speed <= 0:
        resolved_speed = 1.0

    return LongformSynthesisConfig(
        model=resolved_model,
        voice=voice,
        speed=resolved_speed,
        language=language,
        response_format=fmt,
        chunk_chars=cap,
    )


class LongformSynthesisSession(StreamingOperationErrorReporter):

    def __init__(self, *, scheduler: Any, registry: Any, store: Any | None) -> None:
        self._scheduler = scheduler
        self._registry = registry
        self._store = store

        self._config: LongformSynthesisConfig | None = None
        self._adapter: TTSAdapter | None = None
        self._adapter_lease: EnteredAdapter[TTSAdapter] | None = None
        self._resolved_voice: _ResolvedTtsVoice | None = None
        self._effective_cap: int = 0
        self._text_parts: list[str] = []
        self._events: asyncio.Queue[TtsEvent] = asyncio.Queue()
        self._closed = False

    async def configure(self, config: LongformSynthesisConfig) -> None:
        if self._config is not None:
            raise SessionAlreadyConfiguredError()
        self._config = config

        entered = await enter_typed_adapter(
            self._scheduler,
            model=config.model,
            adapter_type=TTSAdapter,
            expected_type="TTS",
        )
        self._adapter_lease = entered
        self._adapter = entered.adapter
        adapter = entered.adapter

        try:
            voice_arg, language_arg, reference_audio, reference_text = resolve_tts_voice_request(
                adapter, self._store, config.voice, config.language,
            )
        except OperationError:
            await release_entered_adapter(entered)
            self._adapter_lease = None
            self._adapter = None
            raise
        self._resolved_voice = _ResolvedTtsVoice(
            voice=voice_arg,
            language=language_arg,
            reference_audio=reference_audio,
            reference_text=reference_text,
        )

        self._effective_cap = effective_tts_text_cap(adapter, config.chunk_chars)

        await self._events.put(TtsReadyEvent(
            model=config.model,
            voice=config.voice,
            response_format=config.response_format,
            chunk_chars=self._effective_cap,
        ))

    async def configure_or_report(self, config: LongformSynthesisConfig) -> bool:
        return await self.run_or_report_operation_error(lambda: self.configure(config))

    def append_text(self, text: str) -> None:
        if not text:
            return
        self._text_parts.append(text)

    async def end_of_stream(self) -> None:
        if self._config is None or self._adapter is None:
            raise SessionNotConfiguredError()
        full_text = "".join(self._text_parts).strip()
        if not full_text:
            await self.report_error(str(EmptyInputError()))
            return
        await self._synthesize(full_text)

    async def end_of_stream_or_report(self) -> bool:
        return await self.run_or_report_operation_error(self.end_of_stream)

    async def report_error(self, message: str) -> None:
        await self._events.put(TtsErrorEvent(message=message))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._adapter_lease is not None:
            await release_entered_adapter_suppressing(self._adapter_lease)
            self._adapter_lease = None
            self._adapter = None

    async def events(self) -> AsyncIterator[TtsEvent]:
        while True:
            event = await self._events.get()
            yield event
            if isinstance(event, (TtsDoneEvent, TtsErrorEvent)):
                return

    async def _synthesize(self, full_text: str) -> None:
        config = self._config
        adapter = self._adapter
        resolved_voice = self._resolved_voice
        assert config is not None and adapter is not None and resolved_voice is not None

        text_chunks = split_text_for_tts_adapter(
            full_text,
            adapter,
            override_chars=self._effective_cap,
        )
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
                voice=resolved_voice.voice,
                speed=config.speed,
                language=resolved_voice.language,
                reference_audio=resolved_voice.reference_audio,
                reference_text=resolved_voice.reference_text,
            ):
                audio = np.frombuffer(chunk.audio, dtype=np.float32)
                if audio.size == 0:
                    continue
                total_audio_samples += audio.size
                output_sample_rate = chunk.sample_rate

                if not audio_meta_sent:
                    await self._events.put(TtsAudioStartEvent(
                        sample_rate=chunk.sample_rate,
                        response_format=config.response_format,
                    ))
                    audio_meta_sent = True

                fmt = config.response_format
                if fmt == "pcm16":
                    await self._events.put(TtsAudioChunkEvent(data=float32_to_pcm16(audio)))
                elif fmt == "opus":
                    pcm16 = float32_to_pcm16(audio)
                    if opus_encoder is None:
                        opus_encoder = OpusStreamEncoder(source_rate=chunk.sample_rate)
                    for opus_frame in opus_encoder.encode(pcm16):
                        await self._events.put(TtsAudioChunkEvent(data=opus_frame))
                elif fmt == "mp3":
                    pcm16 = float32_to_pcm16(audio)
                    if mp3_encoder is None:
                        mp3_encoder = Mp3StreamEncoder(source_rate=chunk.sample_rate)
                    mp3_bytes = mp3_encoder.encode(pcm16)
                    if mp3_bytes:
                        await self._events.put(TtsAudioChunkEvent(data=mp3_bytes))
            total_processing_ms += int((time.perf_counter() - chunk_start) * 1000)
            completed_chunks += 1
            completed_chars += len(text_chunk)
            await self._events.put(TtsProgressEvent(
                completed_chars=completed_chars,
                total_chars=total_chars,
                chunks_completed=completed_chunks,
                chunks_total=len(text_chunks),
            ))

        if opus_encoder is not None:
            for frame in opus_encoder.flush():
                await self._events.put(TtsAudioChunkEvent(data=frame))
        if mp3_encoder is not None:
            tail = mp3_encoder.flush()
            if tail:
                await self._events.put(TtsAudioChunkEvent(data=tail))

        default_done_rate = 48_000 if config.response_format == "opus" else 24_000
        await self._events.put(TtsDoneEvent(
            response_format=config.response_format,
            audio_duration_ms=samples_to_ms(
                total_audio_samples, output_sample_rate or default_done_rate,
            ),
            processing_ms=total_processing_ms,
            text_length=total_chars,
        ))
