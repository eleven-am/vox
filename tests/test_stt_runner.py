from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from tests.fakes import FakeSTTAdapter
from vox.audio.stt_runner import run_stt, run_stt_with_leading_context
from vox.core.types import TranscribeResult, TranscriptSegment, WordTimestamp


class RecordingSTT(FakeSTTAdapter):
    def __init__(self, result: TranscribeResult) -> None:
        super().__init__()
        self._result = result
        self.calls: list[tuple[np.ndarray, dict, str]] = []

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls.append((audio.copy(), kwargs, threading.current_thread().name))
        return self._result


class BlockingSTT(FakeSTTAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.first_release = threading.Event()
        self.second_started = threading.Event()
        self.active = 0
        self.max_active = 0
        self.thread_ids: set[int] = set()

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.thread_ids.add(threading.get_ident())
        try:
            if float(audio[0]) == 1.0:
                self.first_started.set()
                self.first_release.wait(5.0)
                return TranscribeResult(text="late")
            self.second_started.set()
            return TranscribeResult(text="second")
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_run_stt_preserves_adapter_kwargs_and_uses_adapter_lane():
    adapter = RecordingSTT(TranscribeResult(text="ok", duration_ms=1000))
    audio = np.ones(16_000, dtype=np.float32)

    result = await run_stt(
        adapter,
        audio,
        language="fr",
        word_timestamps=True,
        temperature=0.7,
    )

    assert result.text == "ok"
    assert len(adapter.calls) == 1
    _, kwargs, thread_name = adapter.calls[0]
    assert kwargs == {"language": "fr", "word_timestamps": True, "temperature": 0.7}
    assert thread_name.startswith("vox-adapter")
    adapter.close_execution_lane()


@pytest.mark.asyncio
async def test_cancelled_stt_remains_owned_and_serializes_replacement():
    adapter = BlockingSTT()
    first = asyncio.create_task(
        run_stt(
            adapter,
            np.ones(16_000, dtype=np.float32),
            language=None,
            word_timestamps=False,
        )
    )
    assert await asyncio.to_thread(adapter.first_started.wait, 1.0)

    started = time.perf_counter()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert time.perf_counter() - started < 0.1
    assert adapter.physical_work_count == 1

    second = asyncio.create_task(
        run_stt(
            adapter,
            np.full(16_000, 2.0, dtype=np.float32),
            language=None,
            word_timestamps=False,
        )
    )
    await asyncio.sleep(0.05)

    assert adapter.second_started.is_set() is False
    assert adapter.max_active == 1

    adapter.first_release.set()
    result = await second
    for _ in range(100):
        if adapter.physical_work_count == 0:
            break
        await asyncio.sleep(0.01)

    assert result.text == "second"
    assert adapter.physical_work_count == 0
    assert adapter.max_active == 1
    assert len(adapter.thread_ids) == 1
    adapter.close_execution_lane()


@pytest.mark.asyncio
async def test_run_stt_with_leading_context_strips_segment_and_word_timestamps():
    padded_result = TranscribeResult(
        text="first word",
        language="en",
        duration_ms=6000,
        segments=(
            TranscriptSegment(
                text="first word",
                start_ms=5000,
                end_ms=6000,
                words=(
                    WordTimestamp(word="first", start_ms=5000, end_ms=5400),
                    WordTimestamp(word="word", start_ms=5400, end_ms=6000),
                ),
            ),
        ),
    )
    adapter = RecordingSTT(padded_result)
    audio = np.full(16_000, 0.2, dtype=np.float32)

    result = await run_stt_with_leading_context(
        adapter,
        audio,
        sample_rate=16_000,
        duration_ms=1000,
        language="en",
        word_timestamps=True,
        temperature=0.0,
    )

    assert result.duration_ms == 1000
    assert adapter.calls[0][0].shape[0] == 6 * 16_000
    assert np.allclose(adapter.calls[0][0][: 5 * 16_000], 0)
    assert np.allclose(adapter.calls[0][0][5 * 16_000 :], audio)
    assert [(s.start_ms, s.end_ms) for s in result.segments] == [(0, 1000)]
    assert [(w.word, w.start_ms, w.end_ms) for w in result.segments[0].words] == [
        ("first", 0, 400),
        ("word", 400, 1000),
    ]


@pytest.mark.asyncio
async def test_run_stt_with_leading_context_clamps_segments_that_overlap_padding():
    padded_result = TranscribeResult(
        text="clamped",
        duration_ms=6100,
        segments=(
            TranscriptSegment(
                text="clamped",
                start_ms=4500,
                end_ms=6100,
                words=(
                    WordTimestamp(word="before", start_ms=4500, end_ms=5050),
                    WordTimestamp(word="after", start_ms=5050, end_ms=6100),
                ),
            ),
        ),
    )
    adapter = RecordingSTT(padded_result)

    result = await run_stt_with_leading_context(
        adapter,
        np.full(16_000, 0.2, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=1000,
        language=None,
        word_timestamps=True,
    )

    assert [(s.start_ms, s.end_ms) for s in result.segments] == [(0, 1000)]
    assert [(w.word, w.start_ms, w.end_ms) for w in result.segments[0].words] == [
        ("before", 0, 50),
        ("after", 50, 1000),
    ]
