from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from vox.audio.merger import merge_transcripts
from vox.audio.pipeline import AudioChunk
from vox.audio.stt_runner import run_stt_with_leading_context
from vox.core.adapter import STTAdapter
from vox.core.adapter_acquisition import AdapterTypeMismatchError, acquire_typed_adapter
from vox.core.scheduler import Scheduler
from vox.core.types import TranscribeResult
from vox.speech_context.service import SpeechContextService, cancel_speech_context_task
from vox.speech_context.types import SpeechContext
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
        segments.append(
            {
                "text": s.text,
                "start_ms": int(s.start_ms),
                "end_ms": int(s.end_ms),
                "words": seg_words,
                "language": s.language,
                "confidence": s.confidence,
            }
        )
        all_words.extend(seg_words)
    return segments, (all_words or None)


logger = logging.getLogger(__name__)

_EOU_FAILURE_LIMIT = 3
_EOU_AUDIO_LIMIT_SAMPLES = TARGET_SAMPLE_RATE * 8

INTERNAL_SILENCE_SPLIT_MIN_AUDIO_MS = 2_500
INTERNAL_SILENCE_FRAME_MS = 40
INTERNAL_SILENCE_MIN_GAP_MS = 350
INTERNAL_SILENCE_PAD_MS = 120
INTERNAL_SILENCE_MIN_SPAN_MS = 250
SPARSE_FINAL_TRANSCRIPT_MIN_AUDIO_MS = 2_500
SPARSE_FINAL_TRANSCRIPT_MAX_WORDS = 2


def _transcript_word_count(text: str) -> int:
    return len([word for word in text.strip().split() if word])


def _transcript_score(result: TranscribeResult) -> int:
    text = (result.text or "").strip()
    return len(text) + (_transcript_word_count(text) * 8)


def _is_sparse_transcript(result: TranscribeResult, *, duration_ms: int) -> bool:
    text = (result.text or "").strip()
    if not text:
        return True
    if duration_ms < SPARSE_FINAL_TRANSCRIPT_MIN_AUDIO_MS:
        return False
    return _transcript_word_count(text) <= SPARSE_FINAL_TRANSCRIPT_MAX_WORDS


def _should_rescue_with_silence_spans(
    result: TranscribeResult,
    *,
    duration_ms: int,
    spans: list[tuple[int, int]],
) -> bool:
    return len(spans) > 1 and _is_sparse_transcript(result, duration_ms=duration_ms)


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
    return [(max(0, start - pad_samples), min(audio.size, end + pad_samples)) for start, end in merged]


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
        speech_context_service: SpeechContextService | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._config = config or StreamPipelineConfig()
        self._speech_context_service = speech_context_service
        self._vad = VADProcessor(config=self._config.vad_config)
        self._eou_model = create_turn_detector(
            self._config.eou_config.model,
            scheduler=scheduler,
        )
        self._conversation_history: list[ConversationTurn] = []
        self._history_lock = threading.Lock()
        self._pending_user_text = ""
        self._pending_user_audio = np.array([], dtype=np.float32)
        self._low_eou_streak = 0
        self._eou_failure_streak = 0
        self._eou_disabled = False
        self._session_config: StreamSessionConfig | None = None
        self._early_context_tasks: dict[int, asyncio.Task[SpeechContext]] = {}
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=self._config.stt_workers,
            thread_name_prefix="stt",
        )

    def configure(self, config: StreamSessionConfig) -> None:
        self._cancel_early_context_tasks()
        self._session_config = config
        self._vad.reset()
        with self._history_lock:
            self._conversation_history.clear()
        self._pending_user_text = ""
        self._pending_user_audio = np.array([], dtype=np.float32)
        self._low_eou_streak = 0

    def _history_limit(self) -> int:
        return max(1, int(self._config.eou_config.max_context_turns))

    def add_assistant_turn(self, text: str) -> None:
        if text.strip():
            with self._history_lock:
                self._conversation_history.append(ConversationTurn(role="assistant", content=text.strip()))
                history_limit = self._history_limit() * 2
                if len(self._conversation_history) > history_limit:
                    self._conversation_history = self._conversation_history[-history_limit:]

    def reset(self) -> None:
        self._cancel_early_context_tasks()
        self._vad.reset()
        self._pending_user_text = ""
        self._pending_user_audio = np.array([], dtype=np.float32)
        self._low_eou_streak = 0

    async def process_audio(self, audio: NDArray[np.float32]) -> AsyncIterator[StreamEvent]:
        loop = asyncio.get_running_loop()
        executor = self._executor
        if executor is None:
            raise RuntimeError("stream pipeline is shut down")
        event, segment = await loop.run_in_executor(executor, self._vad.append, audio)

        if isinstance(event, SpeechStarted):
            self._start_early_context(event)
            yield event

        if isinstance(event, SpeechStopped):
            has_segment = segment is not None and len(segment.audio) > 0
            if not has_segment:
                await cancel_speech_context_task(
                    self._early_context_tasks.pop(event.utterance_id, None)
                )
            yield SpeechStopped(
                timestamp_ms=event.timestamp_ms,
                expects_transcript=has_segment,
                utterance_id=event.utterance_id,
                start_ms=segment.start_ms if has_segment else event.timestamp_ms,
                end_ms=segment.end_ms if has_segment else event.timestamp_ms,
                audio=(
                    segment.audio
                    if has_segment and self._session_config is not None and self._session_config.speech_context
                    else None
                ),
            )

            if has_segment:
                transcript = await self._transcribe_segment(segment)
                if not transcript.text or not transcript.text.strip():
                    await cancel_speech_context_task(transcript._speech_context_task)
                    yield replace(transcript, _speech_context_task=None)
                    return
                self._append_pending_user_audio(segment.audio)
                try:
                    transcript = await self._add_eou_probability(transcript)
                except asyncio.CancelledError:
                    await cancel_speech_context_task(transcript._speech_context_task)
                    raise
                except Exception:
                    await cancel_speech_context_task(transcript._speech_context_task)
                    raise
                transcript = replace(
                    transcript,
                    speech_context=await self._resolve_context_task(transcript._speech_context_task),
                    _speech_context_task=None,
                )
                yield transcript

    async def _transcribe_segment(self, segment: SpeechSegment) -> StreamTranscript:
        def empty_transcript(model: str = "") -> StreamTranscript:
            return StreamTranscript(
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                audio_duration_ms=max(0, segment.end_ms - segment.start_ms),
                model=model,
                utterance_id=segment.utterance_id,
            )

        if not self._session_config:
            return empty_transcript()

        model = self._session_config.model
        if not model:
            return empty_transcript()

        language = self._session_config.language
        word_timestamps = self._session_config.include_word_timestamps
        context_task = self._early_context_tasks.pop(segment.utterance_id, None)
        if context_task is None:
            context_task = self._start_context_analysis(
                segment.audio,
                timeline_offset_ms=segment.start_ms,
                enabled=self._session_config.speech_context,
            )

        start = time.perf_counter()
        try:
            async with acquire_typed_adapter(
                self._scheduler,
                model=model,
                adapter_type=STTAdapter,
                expected_type="STT",
            ) as adapter:
                result = await self._transcribe_audio_with_context(
                    adapter=adapter,
                    audio=segment.audio,
                    language=language or None,
                    word_timestamps=word_timestamps,
                    temperature=self._session_config.temperature,
                )
        except asyncio.CancelledError:
            await cancel_speech_context_task(context_task)
            raise
        except AdapterTypeMismatchError:
            await cancel_speech_context_task(context_task)
            return empty_transcript(model)
        except Exception:
            await cancel_speech_context_task(context_task)
            raise
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
            utterance_id=segment.utterance_id,
            audio=segment.audio if self._session_config.speech_context else None,
            _speech_context_task=context_task,
        )

    async def transcribe_async(
        self,
        audio: NDArray[np.float32],
        language: str | None = None,
        word_timestamps: bool = False,
        include_speech_context: bool = False,
        timeline_offset_ms: int = 0,
    ) -> StreamTranscript:
        if not self._session_config:
            return StreamTranscript()

        model = self._session_config.model
        context_task = self._start_context_analysis(
            audio,
            timeline_offset_ms=timeline_offset_ms,
            enabled=include_speech_context,
        )

        start = time.perf_counter()
        try:
            async with acquire_typed_adapter(
                self._scheduler,
                model=model,
                adapter_type=STTAdapter,
                expected_type="STT",
            ) as adapter:
                result = await self._transcribe_audio_with_context(
                    adapter=adapter,
                    audio=audio,
                    language=language or None,
                    word_timestamps=word_timestamps,
                    temperature=self._session_config.temperature,
                )
        except asyncio.CancelledError:
            await cancel_speech_context_task(context_task)
            raise
        except AdapterTypeMismatchError:
            await cancel_speech_context_task(context_task)
            return StreamTranscript()
        except Exception:
            await cancel_speech_context_task(context_task)
            raise
        speech_context = await self._resolve_context_task(context_task)
        processing_ms = int((time.perf_counter() - start) * 1000)
        segments, words = _segments_and_words(result)

        return StreamTranscript(
            text=result.text,
            audio_duration_ms=result.duration_ms,
            processing_duration_ms=processing_ms,
            model=model,
            segments=segments,
            words=words,
            speech_context=speech_context,
            audio=audio if include_speech_context else None,
        )

    def _start_context_analysis(
        self,
        audio: NDArray[np.float32],
        *,
        timeline_offset_ms: int,
        enabled: bool,
    ) -> asyncio.Task[SpeechContext] | None:
        if not enabled:
            return None
        if self._speech_context_service is None:
            return asyncio.create_task(self._unavailable_context())
        chunk = AudioChunk(
            data=audio,
            sample_rate=TARGET_SAMPLE_RATE,
            duration_ms=samples_to_ms(audio.size),
            offset_ms=timeline_offset_ms,
        )
        return asyncio.create_task(
            self._speech_context_service.analyze_chunks(
                (chunk,),
                timeline_offset_ms=timeline_offset_ms,
            )
        )

    def _start_early_context(self, event: SpeechStarted) -> None:
        config = self._session_config
        if config is None or not config.speech_context or event.audio is None or not event.audio.size:
            return
        previous = self._early_context_tasks.pop(event.utterance_id, None)
        if previous is not None:
            previous.cancel()
        task = self._start_context_analysis(
            event.audio,
            timeline_offset_ms=event.timestamp_ms,
            enabled=True,
        )
        if task is not None:
            self._early_context_tasks[event.utterance_id] = task

    def _cancel_early_context_tasks(self) -> None:
        tasks = tuple(self._early_context_tasks.values())
        self._early_context_tasks = {}
        for task in tasks:
            task.cancel()

    @staticmethod
    async def _unavailable_context() -> SpeechContext:
        return SpeechContext(
            status="failed",
            unavailable=("speaker", "sounds"),
        )

    @staticmethod
    async def _resolve_context_task(
        task: asyncio.Task[SpeechContext] | None,
    ) -> SpeechContext | None:
        if task is None:
            return None
        try:
            return await task
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Speech context analysis failed")
            return SpeechContext(
                status="failed",
                unavailable=("speaker", "sounds"),
            )

    async def _transcribe_audio_with_context(
        self,
        *,
        adapter: STTAdapter,
        audio: NDArray[np.float32],
        language: str | None,
        word_timestamps: bool,
        temperature: float,
    ) -> TranscribeResult:
        duration_ms = samples_to_ms(audio.size)

        whole = await self._transcribe_audio_span(
            adapter=adapter,
            audio=audio,
            language=language,
            word_timestamps=word_timestamps,
            temperature=temperature,
        )

        spans = _speech_spans_for_transcription(audio)
        if not _should_rescue_with_silence_spans(whole, duration_ms=duration_ms, spans=spans):
            return replace(whole, duration_ms=duration_ms)

        merged = await self._transcribe_silence_spans(
            adapter=adapter,
            audio=audio,
            spans=spans,
            language=language,
            word_timestamps=word_timestamps,
            temperature=temperature,
        )
        if merged is None:
            return replace(whole, duration_ms=duration_ms)
        merged = replace(merged, duration_ms=duration_ms)
        if _transcript_score(merged) <= _transcript_score(whole):
            return replace(whole, duration_ms=duration_ms)

        logger.info(
            "rescued sparse final transcript from silence-split spans "
            "whole_words=%d merged_words=%d audio_ms=%d spans=%d",
            _transcript_word_count(whole.text or ""),
            _transcript_word_count(merged.text or ""),
            duration_ms,
            len(spans),
        )
        return merged

    async def _transcribe_silence_spans(
        self,
        *,
        adapter: STTAdapter,
        audio: NDArray[np.float32],
        spans: list[tuple[int, int]],
        language: str | None,
        word_timestamps: bool,
        temperature: float,
    ) -> TranscribeResult | None:
        per_span: list[tuple[TranscribeResult, int]] = []
        for start_sample, end_sample in spans:
            span_audio = audio[start_sample:end_sample]
            if span_audio.size == 0:
                continue
            partial = await self._transcribe_audio_span(
                adapter=adapter,
                audio=span_audio,
                language=language,
                word_timestamps=word_timestamps,
                temperature=temperature,
            )
            if partial.text and partial.text.strip():
                per_span.append((partial, samples_to_ms(start_sample)))

        if not per_span:
            return None
        return merge_transcripts(per_span)

    async def _transcribe_audio_span(
        self,
        *,
        adapter: STTAdapter,
        audio: NDArray[np.float32],
        language: str | None,
        word_timestamps: bool,
        temperature: float,
    ) -> TranscribeResult:
        return await run_stt_with_leading_context(
            adapter,
            audio,
            sample_rate=TARGET_SAMPLE_RATE,
            duration_ms=samples_to_ms(audio.size),
            language=language,
            word_timestamps=word_timestamps,
            temperature=temperature,
        )

    def _append_pending_user_audio(self, audio: NDArray[np.float32]) -> None:
        if audio.size == 0:
            return
        pieces: list[NDArray[np.float32]] = []
        if self._pending_user_audio.size:
            silence_samples = int(max(0, self._config.vad_config.min_silence_duration_ms) * TARGET_SAMPLE_RATE / 1000)
            if silence_samples:
                pieces.append(np.zeros(silence_samples, dtype=np.float32))
        pieces.append(np.asarray(audio, dtype=np.float32))
        self._pending_user_audio = np.concatenate([self._pending_user_audio, *pieces])[-_EOU_AUDIO_LIMIT_SAMPLES:]

    async def _add_eou_probability(self, transcript: StreamTranscript) -> StreamTranscript:
        self._pending_user_text = (self._pending_user_text + " " + transcript.text).strip()

        if self._eou_disabled:
            self._flush_pending_user_text()
            transcript.eou_probability = None
            return transcript

        with self._history_lock:
            history_with_current = self._conversation_history.copy()
        history_with_current.append(ConversationTurn(role="user", content=self._pending_user_text))

        try:
            prediction = self._eou_model.predict(
                history_with_current,
                audio=self._pending_user_audio,
                sample_rate=TARGET_SAMPLE_RATE,
                max_context_turns=self._history_limit(),
            )
            eou_probability = await prediction if inspect.isawaitable(prediction) else prediction
            self._eou_failure_streak = 0
            transcript.eou_probability = eou_probability

            if eou_probability >= self._config.eou_config.threshold:
                self._flush_pending_user_text()
            else:
                self._low_eou_streak += 1

                pending_tokens = self._eou_model.token_count(self._pending_user_text)
                if self._low_eou_streak >= 3 or pending_tokens >= self._config.eou_config.max_pending_tokens:
                    self._flush_pending_user_text()
        except Exception:
            self._eou_failure_streak += 1
            if self._eou_failure_streak >= _EOU_FAILURE_LIMIT:
                logger.exception(
                    "EOU inference failed %d times in a row; disabling EOU for this session",
                    self._eou_failure_streak,
                )
                self._eou_disabled = True
            else:
                logger.exception("EOU inference failed; continuing without turn scoring for this transcript")
            transcript.eou_probability = None
            self._flush_pending_user_text()
            return transcript

        return transcript

    def _flush_pending_user_text(self) -> None:
        if not self._pending_user_text:
            self._low_eou_streak = 0
            self._pending_user_audio = np.array([], dtype=np.float32)
            return

        with self._history_lock:
            self._conversation_history.append(ConversationTurn(role="user", content=self._pending_user_text))
            history_limit = self._history_limit() * 2
            if len(self._conversation_history) > history_limit:
                self._conversation_history = self._conversation_history[-history_limit:]

        self._pending_user_text = ""
        self._pending_user_audio = np.array([], dtype=np.float32)
        self._low_eou_streak = 0

    async def shutdown(self) -> None:
        context_tasks = tuple(self._early_context_tasks.values())
        self._early_context_tasks = {}
        for task in context_tasks:
            task.cancel()
        if context_tasks:
            await asyncio.gather(*context_tasks, return_exceptions=True)
        executor = self._executor
        if executor is None:
            return
        self._executor = None
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
