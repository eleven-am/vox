from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vox.audio.pipeline import prepare_for_stt
from vox.core.adapter import STTAdapter
from vox.operations.defaults import resolve_default_model
from vox.operations.errors import (
    EmptyAudioError,
    InvalidConfigError,
    NoDefaultModelError,
    SessionAlreadyConfiguredError,
    SessionNotConfiguredError,
    UnsupportedFormatError,
    WrongModelTypeError,
)
from vox.streaming.codecs import pcm16_to_float32, resample_audio
from vox.streaming.types import TARGET_SAMPLE_RATE, samples_to_ms

logger = logging.getLogger(__name__)


DEFAULT_LONGFORM_CHUNK_MS = 30_000
DEFAULT_LONGFORM_OVERLAP_MS = 1_000
MAX_LONGFORM_CHUNK_MS = 120_000
MAX_LONGFORM_OVERLAP_MS = 10_000
SUPPORTED_LONGFORM_INPUT_FORMATS = {"pcm16", "wav", "flac", "mp3", "ogg", "webm"}


@dataclass(frozen=True)
class LongformTranscriptionConfig:
    model: str = ""
    sample_rate: int = TARGET_SAMPLE_RATE
    input_format: str = "pcm16"
    language: str | None = None
    word_timestamps: bool = False
    temperature: float = 0.0
    chunk_ms: int = DEFAULT_LONGFORM_CHUNK_MS
    overlap_ms: int = DEFAULT_LONGFORM_OVERLAP_MS


@dataclass(frozen=True)
class LongformReadyEvent:
    model: str
    sample_rate: int
    input_format: str
    chunk_ms: int
    overlap_ms: int


@dataclass(frozen=True)
class LongformProgressEvent:
    uploaded_ms: int
    processed_ms: int
    chunks_completed: int


@dataclass(frozen=True)
class LongformDoneEvent:
    model: str
    text: str
    language: str | None
    duration_ms: int
    processing_ms: int
    segments: tuple[dict, ...]


@dataclass(frozen=True)
class LongformErrorEvent:
    message: str


LongformEvent = (
    LongformReadyEvent
    | LongformProgressEvent
    | LongformDoneEvent
    | LongformErrorEvent
)


@dataclass
class _LongformState:
    chunk_samples: int
    overlap_samples: int
    pending_audio: np.ndarray
    next_chunk_start_samples: int = 0
    committed_samples: int = 0
    uploaded_samples: int = 0
    processing_ms: int = 0
    chunks_completed: int = 0
    transcript_parts: list[str] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    language: str | None = None


def _clamp_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    return max(minimum, min(parsed, maximum))


def normalize_longform_config(
    *,
    model: str,
    sample_rate: int | None,
    input_format: str | None,
    language: str | None,
    word_timestamps: bool,
    temperature: float,
    chunk_ms: object,
    overlap_ms: object,
    registry: Any,
    store: Any | None,
) -> LongformTranscriptionConfig:
    resolved_model = model or resolve_default_model("stt", registry, store) or ""
    if not resolved_model:
        raise NoDefaultModelError("stt")

    fmt = (input_format or "pcm16").lower()
    if fmt not in SUPPORTED_LONGFORM_INPUT_FORMATS:
        raise UnsupportedFormatError("input_format", fmt, sorted(SUPPORTED_LONGFORM_INPUT_FORMATS))

    rate = int(sample_rate or TARGET_SAMPLE_RATE)
    if fmt == "pcm16" and rate <= 0:
        raise InvalidConfigError("sample_rate must be positive")

    chunk = _clamp_int(chunk_ms, DEFAULT_LONGFORM_CHUNK_MS, 1_000, MAX_LONGFORM_CHUNK_MS)
    overlap = _clamp_int(overlap_ms, DEFAULT_LONGFORM_OVERLAP_MS, 0, MAX_LONGFORM_OVERLAP_MS)
    if overlap >= chunk:
        raise InvalidConfigError("overlap_ms must be smaller than chunk_ms")

    return LongformTranscriptionConfig(
        model=resolved_model,
        sample_rate=rate,
        input_format=fmt,
        language=language,
        word_timestamps=bool(word_timestamps),
        temperature=float(temperature or 0.0),
        chunk_ms=chunk,
        overlap_ms=overlap,
    )


class LongformTranscriptionSession:

    def __init__(self, *, scheduler: Any, registry: Any, store: Any | None) -> None:
        self._scheduler = scheduler
        self._registry = registry
        self._store = store

        self._config: LongformTranscriptionConfig | None = None
        self._state: _LongformState | None = None
        self._adapter: STTAdapter | None = None
        self._scheduler_cm = None
        self._events: asyncio.Queue[LongformEvent] = asyncio.Queue()
        self._closed = False

    async def configure(self, config: LongformTranscriptionConfig) -> None:
        if self._config is not None:
            raise SessionAlreadyConfiguredError()
        self._config = config

        cm = self._scheduler.acquire(config.model)
        adapter = await cm.__aenter__()
        self._scheduler_cm = cm
        if not isinstance(adapter, STTAdapter):
            await cm.__aexit__(None, None, None)
            self._scheduler_cm = None
            raise WrongModelTypeError(config.model, "STT")
        self._adapter = adapter

        self._state = _LongformState(
            chunk_samples=int(config.chunk_ms * TARGET_SAMPLE_RATE / 1000),
            overlap_samples=int(config.overlap_ms * TARGET_SAMPLE_RATE / 1000),
            pending_audio=np.array([], dtype=np.float32),
        )
        await self._events.put(LongformReadyEvent(
            model=config.model,
            sample_rate=config.sample_rate,
            input_format=config.input_format,
            chunk_ms=config.chunk_ms,
            overlap_ms=config.overlap_ms,
        ))

    async def submit_chunk(self, data: bytes) -> None:
        if self._config is None or self._state is None or self._adapter is None:
            raise SessionNotConfiguredError()
        try:
            audio = self._decode_chunk(data)
        except Exception as exc:
            await self._events.put(LongformErrorEvent(message=str(exc)))
            return
        if audio.size == 0:
            return

        state = self._state
        state.uploaded_samples += audio.size
        state.pending_audio = (
            audio if state.pending_audio.size == 0 else np.concatenate([state.pending_audio, audio])
        )
        step_samples = state.chunk_samples - state.overlap_samples
        while state.pending_audio.size >= state.chunk_samples:
            chunk_audio = state.pending_audio[:state.chunk_samples]
            await self._run_chunk(chunk_audio, final_chunk=False)
            state.pending_audio = state.pending_audio[step_samples:]
            state.next_chunk_start_samples += step_samples
            state.committed_samples += step_samples
            await self._events.put(LongformProgressEvent(
                uploaded_ms=samples_to_ms(state.uploaded_samples),
                processed_ms=samples_to_ms(state.committed_samples),
                chunks_completed=state.chunks_completed,
            ))

    async def end_of_stream(self) -> None:
        if self._config is None or self._state is None or self._adapter is None:
            raise SessionNotConfiguredError()
        state = self._state
        config = self._config
        if state.uploaded_samples == 0:
            await self._events.put(LongformErrorEvent(message=str(EmptyAudioError())))
            return

        if state.pending_audio.size > 0 and not (
            state.chunks_completed > 0 and state.pending_audio.size <= state.overlap_samples
        ):
            await self._run_chunk(state.pending_audio, final_chunk=True)
            state.committed_samples = state.uploaded_samples

        await self._events.put(LongformDoneEvent(
            model=config.model,
            text=" ".join(part for part in state.transcript_parts if part).strip(),
            language=state.language,
            duration_ms=samples_to_ms(state.uploaded_samples),
            processing_ms=state.processing_ms,
            segments=tuple(state.segments),
        ))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._scheduler_cm is not None:
            with suppress(Exception):
                await self._scheduler_cm.__aexit__(None, None, None)
            self._scheduler_cm = None
            self._adapter = None

    async def events(self) -> AsyncIterator[LongformEvent]:
        while True:
            event = await self._events.get()
            yield event
            if isinstance(event, (LongformDoneEvent, LongformErrorEvent)):
                return

    def _decode_chunk(self, chunk: bytes) -> np.ndarray:
        config = self._config
        assert config is not None
        if config.input_format == "pcm16":
            audio = pcm16_to_float32(chunk)
            if config.sample_rate != TARGET_SAMPLE_RATE:
                audio = resample_audio(audio, config.sample_rate, TARGET_SAMPLE_RATE)
            return audio
        return prepare_for_stt(chunk, format_hint=config.input_format)

    async def _run_chunk(self, chunk_audio: np.ndarray, *, final_chunk: bool) -> None:
        assert self._adapter is not None
        assert self._config is not None
        assert self._state is not None

        config = self._config
        state = self._state
        start_time = time.perf_counter()
        result = await asyncio.to_thread(
            self._adapter.transcribe,
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
