from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from vox.core.adapter import STTAdapter
from vox.core.scheduler import Scheduler
from vox.core.types import TranscribeResult
from vox.streaming.eou import ConversationTurn, EOUConfig, EOUModel
from vox.streaming.types import (
    SpeechStarted,
    SpeechStopped,
    StreamEvent,
    StreamSessionConfig,
    StreamTranscript,
)
from vox.streaming.vad import SpeechSegment, VADConfig, VADProcessor


def _segments_and_words(result: TranscribeResult) -> tuple[list[dict] | None, list[dict] | None]:
    if not result.segments:
        return None, None
    segments: list[dict] = []
    all_words: list[dict] = []
    for s in result.segments:
        seg_words: list[dict] = [
            {
                "word": w.word,
                "start_ms": int(w.start_ms),
                "end_ms": int(w.end_ms),
                "confidence": w.confidence,
            }
            for w in s.words
        ]
        segments.append({
            "text": s.text,
            "start_ms": int(s.start_ms),
            "end_ms": int(s.end_ms),
            "words": seg_words,
            "language": s.language,
            "confidence": s.confidence,
        })
        all_words.extend(seg_words)
    return segments, (all_words or None)

logger = logging.getLogger(__name__)


@dataclass
class StreamPipelineConfig:
    vad_config: VADConfig = field(default_factory=VADConfig)
    eou_config: EOUConfig = field(default_factory=EOUConfig)
    stt_workers: int = 4


class StreamPipeline:

    def __init__(
        self,
        scheduler: Scheduler,
        config: StreamPipelineConfig | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._config = config or StreamPipelineConfig()
        self._vad = VADProcessor(config=self._config.vad_config)
        self._eou_model = EOUModel()
        self._conversation_history: list[ConversationTurn] = []
        self._pending_user_text = ""
        self._low_eou_streak = 0
        self._eou_disabled = False
        self._session_config: StreamSessionConfig | None = None
        self._executor = ThreadPoolExecutor(max_workers=self._config.stt_workers, thread_name_prefix="stt")

    def configure(self, config: StreamSessionConfig) -> None:
        self._session_config = config
        self._vad.reset()
        self._conversation_history.clear()
        self._pending_user_text = ""
        self._low_eou_streak = 0

    def _history_limit(self) -> int:
        return max(1, int(self._config.eou_config.max_context_turns))

    def add_assistant_turn(self, text: str) -> None:
        if text.strip():
            self._conversation_history.append(ConversationTurn(role="assistant", content=text.strip()))
            history_limit = self._history_limit() * 2
            if len(self._conversation_history) > history_limit:
                self._conversation_history = self._conversation_history[-history_limit:]

    def reset(self) -> None:
        self._vad.reset()
        self._pending_user_text = ""
        self._low_eou_streak = 0

    async def process_audio(self, audio: NDArray[np.float32]) -> AsyncIterator[StreamEvent]:
        loop = asyncio.get_running_loop()
        event, segment = await loop.run_in_executor(self._executor, self._vad.append, audio)

        if isinstance(event, SpeechStarted):
            yield event

        if isinstance(event, SpeechStopped):
            yield event

            if segment is not None and len(segment.audio) > 0:
                transcript = await self._transcribe_segment(segment)
                if not transcript.text or not transcript.text.strip():
                    return
                transcript = await loop.run_in_executor(
                    self._executor, self._add_eou_probability, transcript
                )
                yield transcript

    async def _transcribe_segment(self, segment: SpeechSegment) -> StreamTranscript:
        if not self._session_config:
            return StreamTranscript()

        model = self._session_config.model
        if not model:
            return StreamTranscript()

        language = self._session_config.language
        word_timestamps = self._session_config.include_word_timestamps

        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        async with self._scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                return StreamTranscript()
            result = await loop.run_in_executor(
                self._executor,
                lambda: adapter.transcribe(
                    segment.audio,
                    language=language or None,
                    word_timestamps=word_timestamps,
                ),
            )
        processing_ms = int((time.perf_counter() - start) * 1000)
        segments, words = _segments_and_words(result)

        return StreamTranscript(
            text=result.text,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            audio_duration_ms=result.duration_ms,
            processing_duration_ms=processing_ms,
            model=model,
            segments=segments,
            words=words,
        )

    async def transcribe_async(
        self,
        audio: NDArray[np.float32],
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> StreamTranscript:
        if not self._session_config:
            return StreamTranscript()

        model = self._session_config.model

        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        async with self._scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                return StreamTranscript()

            result = await loop.run_in_executor(
                self._executor,
                lambda: adapter.transcribe(
                    audio,
                    language=language or None,
                    word_timestamps=word_timestamps,
                ),
            )
        processing_ms = int((time.perf_counter() - start) * 1000)
        segments, words = _segments_and_words(result)

        return StreamTranscript(
            text=result.text,
            audio_duration_ms=result.duration_ms,
            processing_duration_ms=processing_ms,
            model=model,
            segments=segments,
            words=words,
        )

    def _add_eou_probability(self, transcript: StreamTranscript) -> StreamTranscript:
        self._pending_user_text = (self._pending_user_text + " " + transcript.text).strip()

        if self._eou_disabled:
            self._flush_pending_user_text()
            transcript.eou_probability = None
            return transcript

        history_with_current = self._conversation_history.copy()
        history_with_current.append(ConversationTurn(role="user", content=self._pending_user_text))

        try:
            eou_probability = self._eou_model.predict(
                history_with_current,
                max_context_turns=self._history_limit(),
            )
            transcript.eou_probability = eou_probability

            if eou_probability >= self._config.eou_config.threshold:
                self._flush_pending_user_text()
            else:
                self._low_eou_streak += 1

                pending_tokens = self._eou_model.token_count(self._pending_user_text)
                if (
                    self._low_eou_streak >= 3
                    or pending_tokens >= self._config.eou_config.max_pending_tokens
                ):
                    self._flush_pending_user_text()
        except Exception:
            logger.exception("EOU inference failed; disabling EOU and continuing without turn scoring")
            self._eou_disabled = True
            transcript.eou_probability = None
            self._flush_pending_user_text()
            return transcript

        return transcript

    def _flush_pending_user_text(self) -> None:
        if not self._pending_user_text:
            self._low_eou_streak = 0
            return

        self._conversation_history.append(
            ConversationTurn(role="user", content=self._pending_user_text)
        )
        history_limit = self._history_limit() * 2
        if len(self._conversation_history) > history_limit:
            self._conversation_history = self._conversation_history[-history_limit:]

        self._pending_user_text = ""
        self._low_eou_streak = 0

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
