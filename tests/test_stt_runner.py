from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.asyncio
async def test_run_stt_preserves_adapter_kwargs_and_uses_executor_when_given():
    adapter = RecordingSTT(TranscribeResult(text="ok", duration_ms=1000))
    audio = np.ones(16_000, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="assert-stt") as executor:
        result = await run_stt(
            adapter,
            audio,
            language="fr",
            word_timestamps=True,
            temperature=0.7,
            executor=executor,
        )

    assert result.text == "ok"
    assert len(adapter.calls) == 1
    _, kwargs, thread_name = adapter.calls[0]
    assert kwargs == {"language": "fr", "word_timestamps": True, "temperature": 0.7}
    assert thread_name.startswith("assert-stt")


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
