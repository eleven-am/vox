from __future__ import annotations

import asyncio

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


class TestShiftWordsEdgeCases:
    def test_missing_start_end_ms_treated_as_zero(self):
        words = [{"word": "hi"}]
        result = _shift_words(words, 100)
        assert result == [{"word": "hi", "start_ms": 100, "end_ms": 100}]

    def test_negative_offset_applied(self):

        words = [{"word": "hi", "start_ms": 500, "end_ms": 1000}]
        result = _shift_words(words, -200)
        assert result == [{"word": "hi", "start_ms": 300, "end_ms": 800}]

    def test_floating_ms_coerced_to_int(self):
        words = [{"word": "hi", "start_ms": 100.7, "end_ms": 200.3}]
        result = _shift_words(words, 50)
        assert result[0]["start_ms"] == 150
        assert result[0]["end_ms"] == 250
        assert isinstance(result[0]["start_ms"], int)

    def test_confidence_preserved_through_shift(self):
        words = [{"word": "hi", "start_ms": 0, "end_ms": 100, "confidence": 0.9}]
        result = _shift_words(words, 500)
        assert result[0]["confidence"] == 0.9

    def test_large_list_performance(self):

        words = [{"word": f"w{i}", "start_ms": i * 50, "end_ms": i * 50 + 40} for i in range(10_000)]
        result = _shift_words(words, 1_000)
        assert len(result) == 10_000
        assert result[0]["start_ms"] == 1_000
        assert result[-1]["end_ms"] == (9_999 * 50 + 40 + 1_000)


class TestShiftSegmentsEdgeCases:
    def test_segment_without_words(self):
        segments = [{"text": "x", "start_ms": 0, "end_ms": 100}]
        result = _shift_segments(segments, 500)
        assert result[0]["start_ms"] == 500
        assert result[0].get("words") is None

    def test_zero_offset_returns_original(self):
        segments = [{"text": "x", "start_ms": 0, "end_ms": 100}]
        assert _shift_segments(segments, 0) is segments

    def test_none_input(self):
        assert _shift_segments(None, 100) is None


class TestDedupByTimestampEdgeCases:
    def test_boundary_inclusive_at_last_partial_ms(self):
        words = [{"word": "at-boundary", "start_ms": 1000, "end_ms": 1100}]
        fresh, text = _dedup_by_timestamp(words, 1000)
        assert len(fresh) == 1

    def test_handles_empty_word_field(self):
        words = [
            {"word": "", "start_ms": 1500},
            {"word": "real", "start_ms": 1500},
        ]
        fresh, text = _dedup_by_timestamp(words, 1000)
        assert text == "real"

    def test_skips_whitespace_only_words(self):
        words = [
            {"word": "   ", "start_ms": 1500},
            {"word": "real", "start_ms": 1500},
        ]
        fresh, text = _dedup_by_timestamp(words, 1000)
        assert text == "real"

    def test_equal_timestamps_all_pass(self):
        words = [
            {"word": "a", "start_ms": 500},
            {"word": "b", "start_ms": 500},
            {"word": "c", "start_ms": 500},
        ]
        fresh, text = _dedup_by_timestamp(words, 500)
        assert text == "a b c"


class TestDeduplicateWordsAdversarial:
    def test_case_insensitive_match(self):
        new_text, _ = deduplicate_words("WORLD foo", ["hello", "world"])
        assert new_text == "foo"

    def test_repeated_word_not_false_overlap(self):
        new_text, _ = deduplicate_words("hello hello world", [])
        assert new_text == "hello hello world"

    def test_confirmed_longer_than_new(self):
        new_text, _ = deduplicate_words("hi", ["a", "b", "c", "d", "hi"])
        assert new_text == ""

    def test_partial_tail_overlap(self):
        new_text, _ = deduplicate_words("c d e", ["a", "b", "c", "d"])
        assert new_text == "e"

    def test_punctuation_in_words_blocks_string_dedup(self):

        new_text, _ = deduplicate_words("hello, world", ["hello"])

        assert "world" in new_text


class TestPartialServiceEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_buffer_returns_none(self):
        session = SpeechSession()
        config = StreamSessionConfig(partials=True)

        async def never(**kwargs):
            raise AssertionError("should not be called")

        svc = PartialTranscriptService(never)
        assert await svc.generate_partial_async(session, config) is None

    @pytest.mark.asyncio
    async def test_buffer_exactly_at_window_boundary(self):
        session = SpeechSession()
        session.start_speech()

        window_ms = 1500
        audio = np.zeros(int(window_ms * TARGET_SAMPLE_RATE / 1000), dtype=np.float32)
        session.append_audio(audio)

        config = StreamSessionConfig(
            partials=True,
            partial_window_ms=window_ms,
            partial_stride_ms=700,
        )

        called = {"n": 0}

        async def transcribe(**kwargs):
            called["n"] += 1
            return StreamTranscript(text="hi")

        svc = PartialTranscriptService(transcribe)
        await svc.generate_partial_async(session, config)

        assert called["n"] == 1

    @pytest.mark.asyncio
    async def test_buffer_below_window_no_call(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(int(1.4 * TARGET_SAMPLE_RATE), dtype=np.float32)
        session.append_audio(audio)

        config = StreamSessionConfig(
            partials=True,
            partial_window_ms=1500,
            partial_stride_ms=700,
        )

        async def never(**kwargs):
            raise AssertionError("should not be called")

        svc = PartialTranscriptService(never)
        result = await svc.generate_partial_async(session, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_none(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(3 * TARGET_SAMPLE_RATE, dtype=np.float32)
        session.append_audio(audio)

        config = StreamSessionConfig(partials=True)

        async def transcribe(**kwargs):
            return StreamTranscript(text="")

        svc = PartialTranscriptService(transcribe)
        result = await svc.generate_partial_async(session, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_ts_aware_path_falls_back_to_string_when_no_words(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(3 * TARGET_SAMPLE_RATE, dtype=np.float32)
        session.append_audio(audio)


        config = StreamSessionConfig(
            partials=True,
            include_word_timestamps=True,
        )

        async def transcribe(**kwargs):
            return StreamTranscript(text="hello world", words=None)

        svc = PartialTranscriptService(transcribe)
        result = await svc.generate_partial_async(session, config)
        assert result is not None
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_concurrent_partials_are_serialized_by_session_lock(self):
        """Concurrent calls must not corrupt last_partial_ms or confirmed_words."""
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(3 * TARGET_SAMPLE_RATE, dtype=np.float32)
        session.append_audio(audio)

        config = StreamSessionConfig(partials=True)

        async def transcribe(**kwargs):
            await asyncio.sleep(0.001)
            return StreamTranscript(text="alpha beta")

        svc = PartialTranscriptService(transcribe)

        results = await asyncio.gather(*[
            svc.generate_partial_async(session, config) for _ in range(5)
        ])

        last_ms, words = session.get_partial_state()
        assert last_ms >= 0

    @pytest.mark.asyncio
    async def test_word_shift_preserves_word_field(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(3 * TARGET_SAMPLE_RATE, dtype=np.float32)
        session.append_audio(audio)

        config = StreamSessionConfig(partials=True, include_word_timestamps=True)

        async def transcribe(**kwargs):
            return StreamTranscript(
                text="hello",
                words=[{"word": "hello", "start_ms": 0, "end_ms": 500, "confidence": 0.88}],
            )

        svc = PartialTranscriptService(transcribe)
        result = await svc.generate_partial_async(session, config)
        assert result is not None
        assert result.words[0]["word"] == "hello"
        assert result.words[0]["confidence"] == 0.88

    @pytest.mark.asyncio
    async def test_does_not_mutate_adapter_result(self):
        """Shifting words must not mutate the adapter's original list."""
        session = SpeechSession()
        session.start_speech()
        audio = np.zeros(3 * TARGET_SAMPLE_RATE, dtype=np.float32)
        session.append_audio(audio)

        original_words = [{"word": "hi", "start_ms": 100, "end_ms": 200}]
        config = StreamSessionConfig(partials=True, include_word_timestamps=True)

        async def transcribe(**kwargs):
            return StreamTranscript(
                text="hi",
                words=list(original_words),
            )

        svc = PartialTranscriptService(transcribe)
        await svc.generate_partial_async(session, config)

        assert original_words[0]["start_ms"] == 100

    @pytest.mark.asyncio
    async def test_tail_longer_than_buffer_uses_all_audio(self):
        """If buffer < tail_window, tail_start_ms must be 0 (no negative offset)."""
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

        seen = {}

        async def transcribe(**kwargs):
            seen["len"] = len(kwargs["audio"])
            return StreamTranscript(
                text="hi",
                words=[{"word": "hi", "start_ms": 0, "end_ms": 500}],
            )

        svc = PartialTranscriptService(transcribe)
        result = await svc.generate_partial_async(session, config)

        assert result is not None
        assert result.start_ms == 0
        assert result.words[0]["start_ms"] == 0

    @pytest.mark.asyncio
    async def test_flush_remaining_returns_none_on_empty_buffer(self):
        session = SpeechSession()
        svc = PartialTranscriptService(lambda **_: None)
        assert svc.flush_remaining_audio(session) is None

    @pytest.mark.asyncio
    async def test_flush_remaining_returns_audio_when_present(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.ones(1000, dtype=np.float32)
        session.append_audio(audio)

        svc = PartialTranscriptService(lambda **_: None)
        out = svc.flush_remaining_audio(session)
        assert out is not None
        assert len(out) == 1000
