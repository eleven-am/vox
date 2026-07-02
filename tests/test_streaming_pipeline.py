from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest

from vox.core.adapter import STTAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, TranscribeResult, TranscriptSegment
from vox.streaming.pipeline import StreamPipeline
from vox.streaming.types import TARGET_SAMPLE_RATE, StreamSessionConfig
from vox.streaming.vad import SpeechSegment
from tests.fakes import FakeTTSAdapter


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

    pipeline.shutdown()


@pytest.mark.asyncio
async def test_transcribe_segment_prefers_complete_whole_utterance_over_gap_spans():
    adapter = WholeUtteranceSTTAdapter()
    pipeline = StreamPipeline(scheduler=FakeScheduler(adapter))
    pipeline.configure(StreamSessionConfig(model="m:1", language="en"))

    first = np.full(int(0.8 * TARGET_SAMPLE_RATE), 0.25, dtype=np.float32)
    gap = np.zeros(int(0.9 * TARGET_SAMPLE_RATE), dtype=np.float32)
    second = np.full(int(1.3 * TARGET_SAMPLE_RATE), 0.5, dtype=np.float32)
    audio = np.concatenate([first, gap, second])

    transcript = await pipeline._transcribe_segment(
        SpeechSegment(audio=audio, start_ms=0, end_ms=3000)
    )

    assert transcript.text == "just to hold my waist from behind and he turns me over"
    assert transcript.start_ms == 0
    assert transcript.end_ms == 3000
    assert transcript.audio_duration_ms == 3000
    assert len(adapter.calls) == 1
    assert np.allclose(adapter.calls[0][: 5 * TARGET_SAMPLE_RATE], 0)

    pipeline.shutdown()


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

    pipeline.shutdown()


@pytest.mark.asyncio
async def test_transcribe_segment_splits_clear_internal_silence_gap_when_whole_is_sparse():
    adapter = GapSensitiveSTTAdapter()
    pipeline = StreamPipeline(scheduler=FakeScheduler(adapter))
    pipeline.configure(StreamSessionConfig(model="m:1", language="en"))

    first = np.full(int(0.8 * TARGET_SAMPLE_RATE), 0.25, dtype=np.float32)
    gap = np.zeros(int(0.9 * TARGET_SAMPLE_RATE), dtype=np.float32)
    second = np.full(int(1.3 * TARGET_SAMPLE_RATE), 0.5, dtype=np.float32)
    audio = np.concatenate([first, gap, second])

    transcript = await pipeline._transcribe_segment(
        SpeechSegment(audio=audio, start_ms=0, end_ms=3000)
    )

    assert transcript.text == "Yeah. second phrase"
    assert transcript.start_ms == 0
    assert transcript.end_ms == 3000
    assert transcript.audio_duration_ms == 3000
    assert len(adapter.calls) == 3
    assert all(np.allclose(call[: 5 * TARGET_SAMPLE_RATE], 0) for call in adapter.calls)

    pipeline.shutdown()


@pytest.mark.asyncio
async def test_transcribe_segment_splits_long_silence_gap_inside_one_vad_segment():
    adapter = GapSensitiveSTTAdapter()
    pipeline = StreamPipeline(scheduler=FakeScheduler(adapter))
    pipeline.configure(StreamSessionConfig(model="m:1", language="en"))

    first = np.full(int(0.8 * TARGET_SAMPLE_RATE), 0.25, dtype=np.float32)
    gap = np.zeros(int(2.0 * TARGET_SAMPLE_RATE), dtype=np.float32)
    second = np.full(int(1.3 * TARGET_SAMPLE_RATE), 0.5, dtype=np.float32)
    audio = np.concatenate([first, gap, second])

    transcript = await pipeline._transcribe_segment(
        SpeechSegment(audio=audio, start_ms=0, end_ms=4100)
    )

    assert transcript.text == "Yeah. second phrase"
    assert transcript.audio_duration_ms == 4100
    assert len(adapter.calls) == 3

    first_transcript = await pipeline._transcribe_segment(
        SpeechSegment(audio=first, start_ms=0, end_ms=800)
    )
    second_transcript = await pipeline._transcribe_segment(
        SpeechSegment(audio=second, start_ms=2800, end_ms=4100)
    )

    assert first_transcript.text == "Yeah."
    assert second_transcript.text == "second phrase"

    pipeline.shutdown()
