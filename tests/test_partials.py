from __future__ import annotations

import numpy as np
import pytest

from vox.streaming.partials import (
    PARTIAL_OVERLAP_MS,
    PartialTranscriptService,
    _dedup_by_timestamp,
    _shift_segments,
    _shift_words,
    deduplicate_words,
)
from vox.streaming.session import SpeechSession
from vox.streaming.types import TARGET_SAMPLE_RATE, StreamSessionConfig, StreamTranscript


class TestShiftWords:
    def test_noop_for_zero_offset(self):
        words = [{"word": "hi", "start_ms": 100, "end_ms": 200}]
        result = _shift_words(words, 0)
        assert result is words

    def test_shifts_positive(self):
        words = [{"word": "hi", "start_ms": 100, "end_ms": 200}]
        result = _shift_words(words, 500)
        assert result == [{"word": "hi", "start_ms": 600, "end_ms": 700}]

    def test_preserves_other_fields(self):
        words = [{"word": "hi", "start_ms": 0, "end_ms": 100, "confidence": 0.95}]
        result = _shift_words(words, 50)
        assert result[0]["confidence"] == 0.95

    def test_none_or_empty(self):
        assert _shift_words(None, 100) is None
        assert _shift_words([], 100) == []


class TestShiftSegments:
    def test_shifts_segments_and_nested_words(self):
        segments = [{
            "text": "hello",
            "start_ms": 0,
            "end_ms": 500,
            "words": [{"word": "hello", "start_ms": 0, "end_ms": 500}],
        }]
        result = _shift_segments(segments, 1000)
        assert result[0]["start_ms"] == 1000
        assert result[0]["end_ms"] == 1500
        assert result[0]["words"][0]["start_ms"] == 1000


class TestDedupByTimestamp:
    def test_keeps_words_at_or_after_boundary(self):
        words = [
            {"word": "hello", "start_ms": 500, "end_ms": 900},
            {"word": "world", "start_ms": 1000, "end_ms": 1400},
            {"word": "today", "start_ms": 1600, "end_ms": 2000},
        ]
        fresh, text = _dedup_by_timestamp(words, last_partial_ms=1000)
        assert [w["word"] for w in fresh] == ["world", "today"]
        assert text == "world today"

    def test_drops_all_before_boundary(self):
        words = [{"word": "stale", "start_ms": 100, "end_ms": 200}]
        fresh, text = _dedup_by_timestamp(words, last_partial_ms=1000)
        assert fresh == []
        assert text == ""


class TestPartialServiceTailMath:

    @pytest.mark.asyncio
    async def test_sets_start_end_ms_with_tail_offset(self):
        session = SpeechSession()
        session.start_speech()

        audio = np.zeros(3 * TARGET_SAMPLE_RATE, dtype=np.float32)
        session.append_audio(audio)

        config = StreamSessionConfig(
            partials=True,
            partial_window_ms=1500,
            partial_stride_ms=700,
            include_word_timestamps=True,
        )

        captured: dict = {}

        async def fake_transcribe(**kwargs):
            captured["audio_len"] = len(kwargs["audio"])
            return StreamTranscript(
                text="hello world",
                words=[
                    {"word": "hello", "start_ms": 0, "end_ms": 500},
                    {"word": "world", "start_ms": 500, "end_ms": 1000},
                ],
            )

        svc = PartialTranscriptService(transcribe_async_fn=fake_transcribe)
        result = await svc.generate_partial_async(session, config)

        assert result is not None
        expected_tail_start = 3000 - (config.partial_window_ms + PARTIAL_OVERLAP_MS)
        assert result.start_ms == expected_tail_start
        assert result.end_ms == 3000
        assert result.is_partial is True

        assert result.words[0]["start_ms"] == expected_tail_start
        assert result.words[1]["start_ms"] == expected_tail_start + 500

    @pytest.mark.asyncio
    async def test_no_tail_shift_when_buffer_shorter_than_tail(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(int(1.6 * TARGET_SAMPLE_RATE), dtype=np.float32)
        session.append_audio(audio)

        config = StreamSessionConfig(
            partials=True,
            partial_window_ms=1500,
            partial_stride_ms=700,
            include_word_timestamps=True,
        )

        async def fake_transcribe(**kwargs):
            return StreamTranscript(
                text="hi",
                words=[{"word": "hi", "start_ms": 100, "end_ms": 300}],
            )

        svc = PartialTranscriptService(transcribe_async_fn=fake_transcribe)
        result = await svc.generate_partial_async(session, config)

        assert result is not None
        assert result.start_ms == 0

        assert result.words[0]["start_ms"] == 100

    @pytest.mark.asyncio
    async def test_timestamp_aware_dedup_drops_stale_words(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(3 * TARGET_SAMPLE_RATE, dtype=np.float32)
        session.append_audio(audio)
        session.update_partial(2000, ["hello"])

        config = StreamSessionConfig(
            partials=True,
            partial_window_ms=1500,
            partial_stride_ms=700,
            include_word_timestamps=True,
        )



        async def fake_transcribe(**kwargs):
            return StreamTranscript(
                text="hello world",
                words=[
                    {"word": "hello", "start_ms": 300, "end_ms": 700},
                    {"word": "world", "start_ms": 900, "end_ms": 1200},
                ],
            )

        svc = PartialTranscriptService(transcribe_async_fn=fake_transcribe)
        result = await svc.generate_partial_async(session, config)

        assert result is not None
        assert [w["word"] for w in result.words] == ["world"]
        assert result.text == "world"

    @pytest.mark.asyncio
    async def test_returns_none_when_stride_not_met(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(int(1.8 * TARGET_SAMPLE_RATE), dtype=np.float32)
        session.append_audio(audio)
        session.update_partial(1700, [])

        config = StreamSessionConfig(
            partials=True,
            partial_window_ms=1500,
            partial_stride_ms=700,
        )

        async def fake_transcribe(**kwargs):
            raise AssertionError("should not be called")

        svc = PartialTranscriptService(transcribe_async_fn=fake_transcribe)
        result = await svc.generate_partial_async(session, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_string_dedup_fallback_without_timestamps(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(3 * TARGET_SAMPLE_RATE, dtype=np.float32)
        session.append_audio(audio)
        session.update_partial(0, ["hello"])

        config = StreamSessionConfig(
            partials=True,
            partial_window_ms=1500,
            partial_stride_ms=700,
            include_word_timestamps=False,
        )

        async def fake_transcribe(**kwargs):
            return StreamTranscript(text="hello world")

        svc = PartialTranscriptService(transcribe_async_fn=fake_transcribe)
        result = await svc.generate_partial_async(session, config)

        assert result is not None
        assert result.text == "world"


class TestDeduplicateWords:
    def test_no_overlap(self):
        new_text, words = deduplicate_words("hello world", [])
        assert new_text == "hello world"
        assert words == ["hello", "world"]

    def test_full_overlap(self):
        new_text, _ = deduplicate_words("hello world", ["hello", "world"])
        assert new_text == ""

    def test_partial_overlap(self):
        new_text, words = deduplicate_words("world foo bar", ["hello", "world"])
        assert new_text == "foo bar"
        assert words == ["hello", "world", "foo", "bar"]
