"""Tests for audio chunking + transcript merging (long-audio STT path)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vox.audio.merger import merge_transcripts
from vox.audio.pipeline import (
    STT_CHUNK_DURATION_MS,
    AudioChunk,
    chunk_audio,
    prepare_for_stt_chunks,
)
from vox.core.types import TranscribeResult, TranscriptSegment, WordTimestamp

SAMPLE_RATE = 16000


def _silent(duration_ms: int) -> np.ndarray:
    samples = int(duration_ms / 1000 * SAMPLE_RATE)
    return np.zeros(samples, dtype=np.float32)


class TestChunkAudio:
    def test_short_audio_returns_single_chunk(self):
        audio = _silent(30_000)
        chunks = chunk_audio(audio, sample_rate=SAMPLE_RATE)

        assert len(chunks) == 1
        assert chunks[0].offset_ms == 0
        assert chunks[0].duration_ms == 30_000
        assert chunks[0].sample_rate == SAMPLE_RATE
        assert chunks[0].data.shape == audio.shape

    def test_audio_at_chunk_boundary_stays_single_chunk(self):
        audio = _silent(STT_CHUNK_DURATION_MS)
        chunks = chunk_audio(audio, sample_rate=SAMPLE_RATE)

        assert len(chunks) == 1
        assert chunks[0].duration_ms == STT_CHUNK_DURATION_MS

    def test_long_audio_splits_at_five_minute_boundaries(self):
        audio = _silent(12 * 60 * 1000)
        chunks = chunk_audio(audio, sample_rate=SAMPLE_RATE)

        assert len(chunks) == 3
        assert chunks[0].offset_ms == 0
        assert chunks[1].offset_ms == STT_CHUNK_DURATION_MS
        assert chunks[2].offset_ms == 2 * STT_CHUNK_DURATION_MS

    def test_chunks_cover_all_samples_without_overlap(self):
        audio = _silent(7 * 60 * 1000)
        chunks = chunk_audio(audio, sample_rate=SAMPLE_RATE)
        total_samples = sum(c.data.shape[0] for c in chunks)

        assert total_samples == audio.shape[0]

    def test_tail_chunk_has_remainder_duration(self):
        audio = _silent(7 * 60 * 1000)
        chunks = chunk_audio(audio, sample_rate=SAMPLE_RATE)

        assert len(chunks) == 2
        assert chunks[0].duration_ms == STT_CHUNK_DURATION_MS
        assert chunks[1].duration_ms == 2 * 60 * 1000
        assert chunks[1].offset_ms == STT_CHUNK_DURATION_MS

    def test_custom_chunk_duration_is_respected(self):
        audio = _silent(20_000)
        chunks = chunk_audio(audio, sample_rate=SAMPLE_RATE, chunk_duration_ms=5_000)

        assert len(chunks) == 4
        assert [c.offset_ms for c in chunks] == [0, 5_000, 10_000, 15_000]

    def test_invalid_chunk_duration_raises(self):
        audio = _silent(1_000)
        with pytest.raises(ValueError, match="chunk_duration_ms"):
            chunk_audio(audio, sample_rate=SAMPLE_RATE, chunk_duration_ms=0)

    def test_invalid_sample_rate_raises(self):
        audio = _silent(1_000)
        with pytest.raises(ValueError, match="sample_rate"):
            chunk_audio(audio, sample_rate=0)

    def test_sub_sample_chunk_duration_still_progresses(self):
        audio = np.zeros(3, dtype=np.float32)
        chunks = chunk_audio(audio, sample_rate=500, chunk_duration_ms=1)

        assert len(chunks) == 3
        assert sum(c.data.shape[0] for c in chunks) == 3


class TestPrepareForSttChunks:
    def test_wav_bytes_are_decoded_and_chunked(self):
        from vox.audio.codecs import encode_wav

        audio = np.sin(
            2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * 44_100), endpoint=False, dtype=np.float32)
        ).astype(np.float32)
        wav = encode_wav(audio, 44_100)

        chunks = prepare_for_stt_chunks(wav, target_rate=SAMPLE_RATE)

        assert len(chunks) == 1
        assert chunks[0].sample_rate == SAMPLE_RATE
        assert chunks[0].offset_ms == 0
        assert abs(chunks[0].duration_ms - 500) <= 5


class TestMergeTranscripts:
    def _seg(self, text: str, start_ms: int, end_ms: int) -> TranscriptSegment:
        return TranscriptSegment(
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
            words=(
                WordTimestamp(
                    word=text, start_ms=start_ms, end_ms=end_ms, confidence=None
                ),
            ),
            language="en",
            confidence=None,
        )

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_transcripts([])

    def test_single_zero_offset_returns_same_object(self):
        only = TranscribeResult(
            text="hello",
            segments=(self._seg("hello", 0, 500),),
            language="en",
            duration_ms=500,
            model="parakeet",
        )

        merged = merge_transcripts([(only, 0)])
        assert merged is only

    def test_single_nonzero_offset_still_shifts_segments(self):
        only = TranscribeResult(
            text="hello",
            segments=(self._seg("hello", 0, 500),),
            language="en",
            duration_ms=500,
            model="parakeet",
        )

        merged = merge_transcripts([(only, 3_000)])
        assert merged.text == "hello"
        assert merged.duration_ms == 500
        assert merged.segments[0].start_ms == 3_000
        assert merged.segments[0].end_ms == 3_500
        assert merged.segments[0].words[0].start_ms == 3_000

    def test_concatenates_text_across_chunks(self):
        a = TranscribeResult(text="hello", segments=(), duration_ms=5_000, model="p")
        b = TranscribeResult(text="world", segments=(), duration_ms=3_000, model="p")

        merged = merge_transcripts([(a, 0), (b, 5_000)])
        assert merged.text == "hello world"
        assert merged.duration_ms == 8_000

    def test_shifts_segment_and_word_timestamps_by_chunk_offset(self):
        a = TranscribeResult(
            text="hi",
            segments=(self._seg("hi", 100, 400),),
            duration_ms=1_000,
            model="p",
        )
        b = TranscribeResult(
            text="there",
            segments=(self._seg("there", 50, 300),),
            duration_ms=1_000,
            model="p",
        )

        merged = merge_transcripts([(a, 0), (b, 1_000)])
        assert merged.segments[0].start_ms == 100
        assert merged.segments[0].end_ms == 400
        assert merged.segments[1].start_ms == 1_050
        assert merged.segments[1].end_ms == 1_300
        assert merged.segments[1].words[0].start_ms == 1_050
        assert merged.segments[1].words[0].end_ms == 1_300

    def test_skips_empty_chunks_in_text_but_counts_duration(self):
        a = TranscribeResult(text="hi", segments=(), duration_ms=1_000, model="p")
        empty = TranscribeResult(text="   ", segments=(), duration_ms=2_000, model="p")
        c = TranscribeResult(text="again", segments=(), duration_ms=1_000, model="p")

        merged = merge_transcripts([(a, 0), (empty, 1_000), (c, 3_000)])
        assert merged.text == "hi again"
        assert merged.duration_ms == 4_000

    def test_first_nonempty_model_and_language_propagate(self):
        a = TranscribeResult(text="", segments=(), duration_ms=500, model="", language=None)
        b = TranscribeResult(
            text="x", segments=(), duration_ms=500, model="parakeet", language="en"
        )

        merged = merge_transcripts([(a, 0), (b, 500)])
        assert merged.model == "parakeet"
        assert merged.language == "en"


class TestAudioChunkDataclass:
    def test_audio_chunk_is_frozen(self):
        chunk = AudioChunk(
            data=np.zeros(10, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            duration_ms=1,
            offset_ms=0,
        )
        with pytest.raises(FrozenInstanceError):
            chunk.offset_ms = 999
