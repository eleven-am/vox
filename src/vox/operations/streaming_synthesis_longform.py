from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from vox.conversation.text_buffer import split_for_tts
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import resolve_voice_request
from vox.core.errors import VoiceCloningUnsupportedError, VoiceNotFoundError
from vox.operations.defaults import resolve_default_model
from vox.operations.errors import (
    EmptyInputError,
    InvalidConfigError,
    NoDefaultModelError,
    SessionAlreadyConfiguredError,
    SessionNotConfiguredError,
    UnsupportedFormatError,
    WrongModelTypeError,
)
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


TtsEvent = (
    TtsReadyEvent
    | TtsAudioStartEvent
    | TtsAudioChunkEvent
    | TtsProgressEvent
    | TtsDoneEvent
    | TtsErrorEvent
)


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
    resolved_model = model or resolve_default_model("tts", registry, store) or ""
    if not resolved_model:
        raise NoDefaultModelError("tts")

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

    return LongformSynthesisConfig(
        model=resolved_model,
        voice=voice,
        speed=float(speed or 1.0),
        language=language,
        response_format=fmt,
        chunk_chars=cap,
    )


class LongformSynthesisSession:

    def __init__(self, *, scheduler: Any, registry: Any, store: Any | None) -> None:
        self._scheduler = scheduler
        self._registry = registry
        self._store = store

        self._config: LongformSynthesisConfig | None = None
        self._adapter: TTSAdapter | None = None
        self._scheduler_cm = None
        self._voice_arg: str | None = None
        self._language_arg: str | None = None
        self._reference_audio: bytes | None = None
        self._reference_text: str | None = None
        self._effective_cap: int = 0
        self._text_parts: list[str] = []
        self._events: asyncio.Queue[TtsEvent] = asyncio.Queue()
        self._closed = False

    async def configure(self, config: LongformSynthesisConfig) -> None:
        if self._config is not None:
            raise SessionAlreadyConfiguredError()
        self._config = config

        cm = self._scheduler.acquire(config.model)
        adapter = await cm.__aenter__()
        self._scheduler_cm = cm
        if not isinstance(adapter, TTSAdapter):
            await cm.__aexit__(None, None, None)
            self._scheduler_cm = None
            raise WrongModelTypeError(config.model, "TTS")
        self._adapter = adapter

        try:
            voice_arg, language_arg, reference_audio, reference_text = resolve_voice_request(
                adapter, self._store, config.voice, config.language,
            )
        except (VoiceCloningUnsupportedError, VoiceNotFoundError):
            await cm.__aexit__(None, None, None)
            self._scheduler_cm = None
            self._adapter = None
            raise
        self._voice_arg = voice_arg
        self._language_arg = language_arg
        self._reference_audio = reference_audio
        self._reference_text = reference_text

        adapter_cap = int(getattr(adapter.info(), "max_input_chars", 0) or 0)
        self._effective_cap = config.chunk_chars if config.chunk_chars is not None else adapter_cap

        await self._events.put(TtsReadyEvent(
            model=config.model,
            voice=config.voice,
            response_format=config.response_format,
            chunk_chars=self._effective_cap,
        ))

    def append_text(self, text: str) -> None:
        if not text:
            return
        self._text_parts.append(text)

    async def end_of_stream(self) -> None:
        if self._config is None or self._adapter is None:
            raise SessionNotConfiguredError()
        full_text = "".join(self._text_parts).strip()
        if not full_text:
            await self._events.put(TtsErrorEvent(message=str(EmptyInputError())))
            return
        await self._synthesize(full_text)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._scheduler_cm is not None:
            with suppress(Exception):
                await self._scheduler_cm.__aexit__(None, None, None)
            self._scheduler_cm = None
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
        assert config is not None and adapter is not None

        text_chunks = (
            [full_text]
            if self._effective_cap <= 0
            else split_for_tts(full_text, max_chars=self._effective_cap)
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
                voice=self._voice_arg,
                speed=config.speed,
                language=self._language_arg,
                reference_audio=self._reference_audio,
                reference_text=self._reference_text,
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
