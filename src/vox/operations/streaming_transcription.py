from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.audio.pipeline import prepare_for_stt
from vox.operations.defaults import resolve_default_model
from vox.operations.errors import (
    InvalidConfigError,
    NoDefaultModelError,
    SessionAlreadyConfiguredError,
    SessionNotConfiguredError,
)
from vox.streaming.annotation import enrich_transcript
from vox.streaming.codecs import pcm16_to_float32, resample_audio
from vox.streaming.opus import OPUS_SAMPLE_RATE, OpusStreamDecoder
from vox.streaming.partials import PartialTranscriptService
from vox.streaming.pipeline import StreamPipeline, StreamPipelineConfig
from vox.streaming.session import SpeechSession
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    SpeechStarted,
    SpeechStopped,
    StreamSessionConfig,
    StreamTranscript,
    samples_to_ms,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingTranscriptionConfig:
    model: str = ""
    language: str = "en"
    sample_rate: int = TARGET_SAMPLE_RATE
    partials: bool = False
    partial_window_ms: int = 1500
    partial_stride_ms: int = 700
    include_word_timestamps: bool = False
    temperature: float = 0.0


@dataclass(frozen=True)
class SessionReadyEvent:
    model: str
    language: str
    sample_rate: int


@dataclass(frozen=True)
class SpeechStartedEvent:
    timestamp_ms: int


@dataclass(frozen=True)
class SpeechStoppedEvent:
    timestamp_ms: int


@dataclass(frozen=True)
class TranscriptEvent:
    transcript: StreamTranscript


@dataclass(frozen=True)
class ErrorEvent:
    message: str


@dataclass(frozen=True)
class DoneEvent:
    pass


SessionEvent = (
    SessionReadyEvent
    | SpeechStartedEvent
    | SpeechStoppedEvent
    | TranscriptEvent
    | ErrorEvent
    | DoneEvent
)


class StreamingTranscriptionSession:

    def __init__(
        self,
        *,
        scheduler: Any,
        registry: Any,
        store: Any | None,
        pipeline_config: StreamPipelineConfig | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._registry = registry
        self._store = store
        self._pipeline_config = pipeline_config

        self._config: StreamingTranscriptionConfig | None = None
        self._session_config: StreamSessionConfig | None = None
        self._pipeline: StreamPipeline | None = None
        self._partial_service: PartialTranscriptService | None = None
        self._opus_decoder: OpusStreamDecoder | None = None
        self._session = SpeechSession()
        self._events: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._closed = False

    async def configure(self, config: StreamingTranscriptionConfig) -> None:
        if self._session_config is not None:
            raise SessionAlreadyConfiguredError()

        model = config.model or resolve_default_model("stt", self._registry, self._store) or ""
        if not model:
            raise NoDefaultModelError("stt")

        sample_rate = int(config.sample_rate or TARGET_SAMPLE_RATE)
        if sample_rate <= 0:
            raise InvalidConfigError("sample_rate must be positive")

        session_config = StreamSessionConfig(
            language=config.language or "en",
            sample_rate=sample_rate,
            model=model,
            partials=bool(config.partials),
            partial_window_ms=int(config.partial_window_ms or 1500),
            partial_stride_ms=int(config.partial_stride_ms or 700),
            include_word_timestamps=bool(config.include_word_timestamps),
            temperature=float(config.temperature or 0.0),
        )
        self._config = config
        self._session_config = session_config
        self._pipeline = StreamPipeline(
            scheduler=self._scheduler,
            config=self._pipeline_config,
        )
        self._pipeline.configure(session_config)
        self._partial_service = PartialTranscriptService(
            transcribe_async_fn=self._pipeline.transcribe_async,
        )
        await self._events.put(SessionReadyEvent(
            model=model,
            language=session_config.language,
            sample_rate=session_config.sample_rate,
        ))

    async def submit_pcm16(self, pcm16: bytes, sample_rate: int | None = None) -> None:
        if self._pipeline is None or self._session_config is None:
            raise SessionNotConfiguredError()
        if not pcm16:
            return
        audio = pcm16_to_float32(pcm16)
        src_rate = int(sample_rate or self._session_config.sample_rate)
        if src_rate != TARGET_SAMPLE_RATE:
            audio = resample_audio(audio, src_rate, TARGET_SAMPLE_RATE)
        await self._ingest_audio(audio)

    async def submit_opus(
        self,
        data: bytes,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> None:
        if self._pipeline is None or self._session_config is None:
            raise SessionNotConfiguredError()
        try:
            if self._opus_decoder is None:
                self._opus_decoder = OpusStreamDecoder(
                    sample_rate=sample_rate or OPUS_SAMPLE_RATE,
                    channels=channels or 1,
                )
            audio = self._opus_decoder.decode_frame(data)
            audio = resample_audio(audio, OPUS_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            await self._ingest_audio(audio)
        except Exception as exc:
            await self._events.put(ErrorEvent(message=str(exc)))

    async def submit_encoded(self, data: bytes, format_hint: str | None = None) -> None:
        if self._pipeline is None or self._session_config is None:
            raise SessionNotConfiguredError()
        try:
            audio = prepare_for_stt(data, format_hint=format_hint)
            await self._ingest_audio(audio)
        except Exception as exc:
            await self._events.put(ErrorEvent(message=str(exc)))

    async def end_of_stream(self) -> None:
        if self._pipeline is None or self._session_config is None:
            await self._events.put(DoneEvent())
            return
        if self._opus_decoder is not None:
            try:
                tail = self._opus_decoder.flush()
                if tail.size > 0:
                    audio = resample_audio(tail, OPUS_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                    await self._ingest_audio(audio)
            except Exception as exc:
                await self._events.put(ErrorEvent(message=str(exc)))
        if self._partial_service is not None:
            remaining = self._partial_service.flush_remaining_audio(self._session)
            if remaining is not None and len(remaining) > 0:
                try:
                    duration_ms = samples_to_ms(len(remaining))
                    if duration_ms > 0:
                        transcript = await self._pipeline.transcribe_async(
                            audio=remaining,
                            language=self._session_config.language,
                            word_timestamps=self._session_config.include_word_timestamps,
                        )
                        if transcript.text and transcript.text.strip():
                            enrich_transcript(transcript, self._session_config.language)
                            await self._events.put(TranscriptEvent(transcript=transcript))
                except Exception as exc:
                    await self._events.put(ErrorEvent(message=str(exc)))
        await self._events.put(DoneEvent())

    async def report_error(self, message: str) -> None:
        await self._events.put(ErrorEvent(message=message))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pipeline is not None:
            with suppress(Exception):
                self._pipeline.reset()
            with suppress(Exception):
                self._pipeline.shutdown()

    async def events(self) -> AsyncIterator[SessionEvent]:
        while True:
            event = await self._events.get()
            yield event
            if isinstance(event, DoneEvent):
                return

    async def _ingest_audio(self, audio: NDArray[np.float32]) -> None:
        if self._pipeline is None or self._session_config is None or self._partial_service is None:
            raise SessionNotConfiguredError()
        if audio.size == 0:
            return
        self._session.append_audio(audio)
        try:
            async for stream_event in self._pipeline.process_audio(audio):
                self._dispatch_stream_event(stream_event)
        except Exception as exc:
            await self._events.put(ErrorEvent(message=str(exc)))
            return
        if self._session.is_active() and self._session_config.partials:
            try:
                transcript = await self._partial_service.generate_partial_async(
                    self._session, self._session_config,
                )
                if transcript:
                    await self._events.put(TranscriptEvent(transcript=transcript))
            except Exception as exc:
                await self._events.put(ErrorEvent(message=str(exc)))

    def _dispatch_stream_event(self, event) -> None:
        if isinstance(event, SpeechStarted):
            self._session.start_speech()
            self._events.put_nowait(SpeechStartedEvent(timestamp_ms=event.timestamp_ms))
        elif isinstance(event, SpeechStopped):
            self._session.stop_speech()
            self._events.put_nowait(SpeechStoppedEvent(timestamp_ms=event.timestamp_ms))
        elif isinstance(event, StreamTranscript):
            language = self._session_config.language if self._session_config else "en"
            enrich_transcript(event, language)
            self._events.put_nowait(TranscriptEvent(transcript=event))
