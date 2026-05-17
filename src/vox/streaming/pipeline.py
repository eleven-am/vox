from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from vox.audio.merger import merge_transcripts
from vox.audio.stt_context import add_stt_leading_context, strip_stt_leading_context
from vox.core.adapter import STTAdapter
from vox.core.scheduler import Scheduler
from vox.core.types import TranscribeResult
from vox.streaming.eou import ConversationTurn, EOUConfig, create_turn_detector
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    SpeechStarted,
    SpeechStopped,
    StreamEvent,
    StreamSessionConfig,
    StreamTranscript,
    samples_to_ms,
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

INTERNAL_SILENCE_SPLIT_MIN_AUDIO_MS = 2_500
INTERNAL_SILENCE_FRAME_MS = 40
INTERNAL_SILENCE_MIN_GAP_MS = 350
INTERNAL_SILENCE_PAD_MS = 120
INTERNAL_SILENCE_MIN_SPAN_MS = 250


def _speech_spans_for_transcription(audio: NDArray[np.float32]) -> list[tuple[int, int]]:
    if samples_to_ms(audio.size) < INTERNAL_SILENCE_SPLIT_MIN_AUDIO_MS:
        return [(0, audio.size)]

    frame_samples = max(1, int(INTERNAL_SILENCE_FRAME_MS * TARGET_SAMPLE_RATE / 1000))
    rms_values: list[float] = []
    frame_ranges: list[tuple[int, int]] = []
    for start in range(0, audio.size, frame_samples):
        end = min(start + frame_samples, audio.size)
        frame = audio[start:end]
        if frame.size == 0:
            continue
        rms_values.append(float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)))))
        frame_ranges.append((start, end))

    if not rms_values:
        return [(0, audio.size)]

    rms = np.asarray(rms_values, dtype=np.float32)
    peak = float(np.max(rms))
    if peak < 1e-5:
        return [(0, audio.size)]

    noise_floor = float(np.percentile(rms, 20))
    threshold = max(peak * 0.12, noise_floor * 3.0, 1e-4)
    speech_frames = rms > threshold
    if not bool(np.any(speech_frames)):
        return [(0, audio.size)]

    raw_spans: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_speech in enumerate(speech_frames):
        if is_speech and run_start is None:
            run_start = index
        elif not is_speech and run_start is not None:
            raw_spans.append((frame_ranges[run_start][0], frame_ranges[index - 1][1]))
            run_start = None
    if run_start is not None:
        raw_spans.append((frame_ranges[run_start][0], frame_ranges[-1][1]))

    if len(raw_spans) <= 1:
        return [(0, audio.size)]

    min_gap_samples = int(INTERNAL_SILENCE_MIN_GAP_MS * TARGET_SAMPLE_RATE / 1000)
    min_span_samples = int(INTERNAL_SILENCE_MIN_SPAN_MS * TARGET_SAMPLE_RATE / 1000)
    merged: list[tuple[int, int]] = []
    for start, end in raw_spans:
        if end - start < min_span_samples:
            continue
        if merged and start - merged[-1][1] < min_gap_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    if len(merged) <= 1:
        return [(0, audio.size)]

    pad_samples = int(INTERNAL_SILENCE_PAD_MS * TARGET_SAMPLE_RATE / 1000)
    return [
        (max(0, start - pad_samples), min(audio.size, end + pad_samples))
        for start, end in merged
    ]


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
        self._eou_model = create_turn_detector(self._config.eou_config.model)
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
            has_segment = segment is not None and len(segment.audio) > 0
            yield SpeechStopped(
                timestamp_ms=event.timestamp_ms,
                expects_transcript=has_segment,
            )

            if has_segment:
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
        async with self._scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                return StreamTranscript()
            result = await self._transcribe_audio_with_context(
                adapter=adapter,
                audio=segment.audio,
                language=language or None,
                word_timestamps=word_timestamps,
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
        async with self._scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                return StreamTranscript()

            result = await self._transcribe_audio_with_context(
                adapter=adapter,
                audio=audio,
                language=language or None,
                word_timestamps=word_timestamps,
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

    async def _transcribe_audio_with_context(
        self,
        *,
        adapter: STTAdapter,
        audio: NDArray[np.float32],
        language: str | None,
        word_timestamps: bool,
    ) -> TranscribeResult:
        loop = asyncio.get_running_loop()
        spans = _speech_spans_for_transcription(audio)
        if len(spans) <= 1:
            return await self._transcribe_audio_span(
                loop=loop,
                adapter=adapter,
                audio=audio,
                language=language,
                word_timestamps=word_timestamps,
            )

        per_span: list[tuple[TranscribeResult, int]] = []
        for start_sample, end_sample in spans:
            span_audio = audio[start_sample:end_sample]
            if span_audio.size == 0:
                continue
            partial = await self._transcribe_audio_span(
                loop=loop,
                adapter=adapter,
                audio=span_audio,
                language=language,
                word_timestamps=word_timestamps,
            )
            if partial.text and partial.text.strip():
                per_span.append((partial, samples_to_ms(start_sample)))

        if not per_span:
            return TranscribeResult(text="", language=language, duration_ms=samples_to_ms(audio.size))
        merged = merge_transcripts(per_span)
        return replace(merged, duration_ms=samples_to_ms(audio.size))

    async def _transcribe_audio_span(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        adapter: STTAdapter,
        audio: NDArray[np.float32],
        language: str | None,
        word_timestamps: bool,
    ) -> TranscribeResult:
        audio_with_context, leading_context_ms = add_stt_leading_context(
            audio,
            sample_rate=TARGET_SAMPLE_RATE,
        )
        result = await loop.run_in_executor(
            self._executor,
            lambda: adapter.transcribe(
                audio_with_context,
                language=language,
                word_timestamps=word_timestamps,
            ),
        )
        return strip_stt_leading_context(
            result,
            context_ms=leading_context_ms,
            duration_ms=samples_to_ms(audio.size),
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
