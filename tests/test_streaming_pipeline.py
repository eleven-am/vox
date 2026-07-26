from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.fakes import FakeTTSAdapter
from vox.core.adapter import STTAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, TranscribeResult, TranscriptSegment
from vox.speech_context.types import SpeechContext
from vox.streaming.pipeline import StreamPipeline
from vox.streaming.types import TARGET_SAMPLE_RATE, SpeechStopped, StreamSessionConfig
from vox.streaming.vad import SpeechSegment


class GapSensitiveSTTAdapter(STTAdapter):
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.kwargs: list[dict[str, Any]] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="gap-sensitive-stt",
            type=ModelType.STT,
            architectures=("fake",),
            default_sample_rate=TARGET_SAMPLE_RATE,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls.append(audio.copy())
        self.kwargs.append(dict(kwargs))
        payload = _without_leading_context(audio)
        text = "Yeah." if float(np.max(np.abs(payload))) < 0.4 else "second phrase"
        if _has_internal_silence_gap(payload):
            text = "Yeah."
        duration_ms = int(payload.size / TARGET_SAMPLE_RATE * 1000)
        return TranscribeResult(
            text=text,
            language=kwargs.get("language") or "en",
            duration_ms=duration_ms,
            segments=(TranscriptSegment(text=text, start_ms=0, end_ms=duration_ms),),
        )


class WholeUtteranceSTTAdapter(STTAdapter):
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.kwargs: list[dict[str, Any]] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="whole-utterance-stt",
            type=ModelType.STT,
            architectures=("fake",),
            default_sample_rate=TARGET_SAMPLE_RATE,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls.append(audio.copy())
        self.kwargs.append(dict(kwargs))
        payload = _without_leading_context(audio)
        if _has_internal_silence_gap(payload):
            text = "just to hold my waist from behind and he turns me over"
        else:
            text = "tail fragment"
        duration_ms = int(payload.size / TARGET_SAMPLE_RATE * 1000)
        return TranscribeResult(
            text=text,
            language=kwargs.get("language") or "en",
            duration_ms=duration_ms,
            segments=(TranscriptSegment(text=text, start_ms=0, end_ms=duration_ms),),
        )


class EmptySTTAdapter(WholeUtteranceSTTAdapter):
    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls.append(audio.copy())
        self.kwargs.append(dict(kwargs))
        payload = _without_leading_context(audio)
        duration_ms = int(payload.size / TARGET_SAMPLE_RATE * 1000)
        return TranscribeResult(
            text="",
            language=kwargs.get("language") or "en",
            duration_ms=duration_ms,
        )


class FakeScheduler:
    def __init__(self, adapter: STTAdapter) -> None:
        self._adapter = adapter

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self._adapter


class TrackingWrongTypeScheduler:
    def __init__(self) -> None:
        self.closed = False

    def acquire(self, _model: str):
        scheduler = self

        class Manager:
            async def __aenter__(self):
                return FakeTTSAdapter()

            async def __aexit__(self, _exc_type, _exc, _tb):
                scheduler.closed = True
                return False

        return Manager()


def _without_leading_context(audio: np.ndarray) -> np.ndarray:
    context_samples = 5 * TARGET_SAMPLE_RATE
    if audio.size > context_samples and np.allclose(audio[:context_samples], 0):
        return audio[context_samples:]
    return audio


def _has_internal_silence_gap(audio: np.ndarray) -> bool:
    silent = np.abs(audio) < 0.01
    min_gap = int(0.5 * TARGET_SAMPLE_RATE)
    run = 0
    for index, is_silent in enumerate(silent):
        if is_silent:
            run += 1
            continue
        if run >= min_gap and index - run > 0:
            return True
        run = 0
    return False


@pytest.mark.asyncio
async def test_transcribe_segment_keeps_wrong_adapter_type_as_empty_transcript_and_closes():
    scheduler = TrackingWrongTypeScheduler()
    pipeline = StreamPipeline(scheduler=scheduler)
    pipeline.configure(StreamSessionConfig(model="fake-tts:latest", language="en"))

    transcript = await pipeline._transcribe_segment(
        SpeechSegment(
            audio=np.full(int(0.8 * TARGET_SAMPLE_RATE), 0.25, dtype=np.float32),
            start_ms=0,
            end_ms=800,
        )
    )

    assert transcript.text == ""
    assert scheduler.closed

    await pipeline.shutdown()


@pytest.mark.asyncio
async def test_transcribe_segment_prefers_complete_whole_utterance_over_gap_spans():
    adapter = WholeUtteranceSTTAdapter()
    pipeline = StreamPipeline(scheduler=FakeScheduler(adapter))
    pipeline.configure(StreamSessionConfig(model="m:1", language="en"))

    first = np.full(int(0.8 * TARGET_SAMPLE_RATE), 0.25, dtype=np.float32)
    gap = np.zeros(int(0.9 * TARGET_SAMPLE_RATE), dtype=np.float32)
    second = np.full(int(1.3 * TARGET_SAMPLE_RATE), 0.5, dtype=np.float32)
    audio = np.concatenate([first, gap, second])

    transcript = await pipeline._transcribe_segment(SpeechSegment(audio=audio, start_ms=0, end_ms=3000, utterance_id=7))

    assert transcript.text == "just to hold my waist from behind and he turns me over"
    assert transcript.start_ms == 0
    assert transcript.end_ms == 3000
    assert transcript.audio_duration_ms == 3000
    assert transcript.utterance_id == 7
    assert len(adapter.calls) == 1
    assert np.allclose(adapter.calls[0][: 5 * TARGET_SAMPLE_RATE], 0)

    await pipeline.shutdown()


@pytest.mark.asyncio
async def test_transcribe_segment_passes_session_stt_options_to_every_rescue_call():
    adapter = GapSensitiveSTTAdapter()
    pipeline = StreamPipeline(scheduler=FakeScheduler(adapter))
    pipeline.configure(
        StreamSessionConfig(
            model="m:1",
            language="fr",
            include_word_timestamps=True,
            temperature=0.4,
        )
    )

    first = np.full(int(0.8 * TARGET_SAMPLE_RATE), 0.25, dtype=np.float32)
    gap = np.zeros(int(0.9 * TARGET_SAMPLE_RATE), dtype=np.float32)
    second = np.full(int(1.3 * TARGET_SAMPLE_RATE), 0.5, dtype=np.float32)
    audio = np.concatenate([first, gap, second])

    await pipeline._transcribe_segment(SpeechSegment(audio=audio, start_ms=0, end_ms=3000))

    assert len(adapter.kwargs) == 3
    assert adapter.kwargs == [
        {"language": "fr", "word_timestamps": True, "temperature": 0.4},
        {"language": "fr", "word_timestamps": True, "temperature": 0.4},
        {"language": "fr", "word_timestamps": True, "temperature": 0.4},
    ]

    await pipeline.shutdown()


@pytest.mark.asyncio
async def test_transcribe_segment_splits_clear_internal_silence_gap_when_whole_is_sparse():
    adapter = GapSensitiveSTTAdapter()
    pipeline = StreamPipeline(scheduler=FakeScheduler(adapter))
    pipeline.configure(StreamSessionConfig(model="m:1", language="en"))

    first = np.full(int(0.8 * TARGET_SAMPLE_RATE), 0.25, dtype=np.float32)
    gap = np.zeros(int(0.9 * TARGET_SAMPLE_RATE), dtype=np.float32)
    second = np.full(int(1.3 * TARGET_SAMPLE_RATE), 0.5, dtype=np.float32)
    audio = np.concatenate([first, gap, second])

    transcript = await pipeline._transcribe_segment(SpeechSegment(audio=audio, start_ms=0, end_ms=3000))

    assert transcript.text == "Yeah. second phrase"
    assert transcript.start_ms == 0
    assert transcript.end_ms == 3000
    assert transcript.audio_duration_ms == 3000
    assert len(adapter.calls) == 3
    assert all(np.allclose(call[: 5 * TARGET_SAMPLE_RATE], 0) for call in adapter.calls)

    await pipeline.shutdown()


@pytest.mark.asyncio
async def test_transcribe_segment_splits_long_silence_gap_inside_one_vad_segment():
    adapter = GapSensitiveSTTAdapter()
    pipeline = StreamPipeline(scheduler=FakeScheduler(adapter))
    pipeline.configure(StreamSessionConfig(model="m:1", language="en"))

    first = np.full(int(0.8 * TARGET_SAMPLE_RATE), 0.25, dtype=np.float32)
    gap = np.zeros(int(2.0 * TARGET_SAMPLE_RATE), dtype=np.float32)
    second = np.full(int(1.3 * TARGET_SAMPLE_RATE), 0.5, dtype=np.float32)
    audio = np.concatenate([first, gap, second])

    transcript = await pipeline._transcribe_segment(SpeechSegment(audio=audio, start_ms=0, end_ms=4100))

    assert transcript.text == "Yeah. second phrase"
    assert transcript.audio_duration_ms == 4100
    assert len(adapter.calls) == 3

    first_transcript = await pipeline._transcribe_segment(SpeechSegment(audio=first, start_ms=0, end_ms=800))
    second_transcript = await pipeline._transcribe_segment(SpeechSegment(audio=second, start_ms=2800, end_ms=4100))

    assert first_transcript.text == "Yeah."
    assert second_transcript.text == "second phrase"

    await pipeline.shutdown()


class _StoppedVad:
    def __init__(self, segment: SpeechSegment) -> None:
        self._segment = segment

    def append(self, audio):
        return (
            SpeechStopped(
                timestamp_ms=self._segment.end_ms,
                utterance_id=self._segment.utterance_id,
            ),
            self._segment,
        )


class _EouGatedContextService:
    def __init__(self) -> None:
        self.eou_started = asyncio.Event()

    async def analyze_chunks(self, chunks, *, timeline_offset_ms: int = 0) -> SpeechContext:
        assert tuple(chunks)
        assert timeline_offset_ms == 100
        await self.eou_started.wait()
        return SpeechContext(status="failed", unavailable=("speaker", "sounds"))


@pytest.mark.asyncio
async def test_realtime_pipeline_preserves_vad_audio_when_stt_is_empty():
    audio = np.full(3_200, 0.2, dtype=np.float32)
    segment = SpeechSegment(audio=audio, start_ms=100, end_ms=300, utterance_id=4)
    pipeline = StreamPipeline(scheduler=FakeScheduler(EmptySTTAdapter()))
    pipeline.configure(
        StreamSessionConfig(
            model="m:1",
            language="en",
            speech_context=True,
        )
    )
    pipeline._vad = _StoppedVad(segment)

    events = await anext(_collect_pipeline_events(pipeline, audio))

    assert len(events) == 1
    stopped = events[0]
    assert isinstance(stopped, SpeechStopped)
    assert stopped.start_ms == 100
    assert stopped.end_ms == 300
    assert stopped.utterance_id == 4
    assert np.array_equal(stopped.audio, audio)
    await pipeline.shutdown()


@pytest.mark.asyncio
async def test_realtime_pipeline_scores_eou_before_waiting_for_speech_context():
    audio = np.full(3_200, 0.2, dtype=np.float32)
    segment = SpeechSegment(audio=audio, start_ms=100, end_ms=300, utterance_id=4)
    context_service = _EouGatedContextService()
    pipeline = StreamPipeline(
        scheduler=FakeScheduler(WholeUtteranceSTTAdapter()),
        speech_context_service=context_service,
    )
    pipeline.configure(
        StreamSessionConfig(
            model="m:1",
            language="en",
            speech_context=True,
        )
    )
    pipeline._vad = _StoppedVad(segment)

    async def score_eou(transcript):
        context_service.eou_started.set()
        return transcript

    pipeline._add_eou_probability = score_eou
    events = await asyncio.wait_for(
        anext(_collect_pipeline_events(pipeline, audio)),
        timeout=1,
    )

    assert events[-1].speech_context is not None
    assert events[-1].speech_context.status == "failed"
    await pipeline.shutdown()


@pytest.mark.asyncio
async def test_realtime_pipeline_cancels_context_when_eou_scoring_fails():
    class CancellationContextService:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.stopped = asyncio.Event()

        async def analyze_chunks(self, chunks, *, timeline_offset_ms: int = 0):
            assert tuple(chunks)
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.stopped.set()
            return SpeechContext(status="failed", unavailable=("speaker", "sounds"))

    audio = np.full(3_200, 0.2, dtype=np.float32)
    segment = SpeechSegment(audio=audio, start_ms=100, end_ms=300, utterance_id=4)
    context_service = CancellationContextService()
    pipeline = StreamPipeline(
        scheduler=FakeScheduler(WholeUtteranceSTTAdapter()),
        speech_context_service=context_service,
    )
    pipeline.configure(StreamSessionConfig(model="m:1", language="en", speech_context=True))
    pipeline._vad = _StoppedVad(segment)

    async def fail_eou(transcript):
        await context_service.started.wait()
        raise RuntimeError("EOU failed")

    pipeline._add_eou_probability = fail_eou

    with pytest.raises(RuntimeError, match="EOU failed"):
        await anext(_collect_pipeline_events(pipeline, audio))

    assert context_service.stopped.is_set()
    await pipeline.shutdown()


@pytest.mark.asyncio
async def test_pipeline_shutdown_does_not_block_event_loop():
    class BlockingExecutor:
        def __init__(self) -> None:
            self.thread_id: int | None = None

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is True
            assert cancel_futures is True
            self.thread_id = threading.get_ident()
            threading.Event().wait(0.1)

    pipeline = StreamPipeline(MagicMock())
    pipeline._executor.shutdown(wait=True)
    executor = BlockingExecutor()
    pipeline._executor = executor
    ticked = asyncio.Event()

    async def tick() -> None:
        await asyncio.sleep(0.01)
        ticked.set()

    ticker = asyncio.create_task(tick())
    await pipeline.shutdown()
    await ticker

    assert ticked.is_set()
    assert executor.thread_id != threading.get_ident()


async def _collect_pipeline_events(pipeline: StreamPipeline, audio: np.ndarray):
    yield [event async for event in pipeline.process_audio(audio)]
