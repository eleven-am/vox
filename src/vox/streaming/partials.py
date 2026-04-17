from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from vox.streaming.session import SpeechSession
from vox.streaming.types import TARGET_SAMPLE_RATE, StreamSessionConfig, StreamTranscript, samples_to_ms

logger = logging.getLogger(__name__)

PARTIAL_OVERLAP_MS = 300

AsyncTranscribeFn = Callable[..., Awaitable[StreamTranscript]]


def deduplicate_words(text: str, confirmed_words: list[str]) -> tuple[str, list[str]]:
    words = [w for w in text.strip().split() if w]
    overlap = 0
    max_overlap = min(len(words), len(confirmed_words))
    for i in range(max_overlap, 0, -1):
        if [w.lower() for w in confirmed_words[-i:]] == [w.lower() for w in words[:i]]:
            overlap = i
            break
    new_words = words[overlap:]
    if new_words:
        confirmed_words.extend(new_words)
    return " ".join(new_words), confirmed_words


def _shift_words(words: list[dict] | None, offset_ms: int) -> list[dict] | None:
    if not words:
        return words
    if offset_ms == 0:
        return words
    return [
        {
            **w,
            "start_ms": int(w.get("start_ms", 0)) + offset_ms,
            "end_ms": int(w.get("end_ms", 0)) + offset_ms,
        }
        for w in words
    ]


def _shift_segments(segments: list[dict] | None, offset_ms: int) -> list[dict] | None:
    if not segments:
        return segments
    if offset_ms == 0:
        return segments
    return [
        {
            **s,
            "start_ms": int(s.get("start_ms", 0)) + offset_ms,
            "end_ms": int(s.get("end_ms", 0)) + offset_ms,
            "words": _shift_words(s.get("words"), offset_ms),
        }
        for s in segments
    ]


def _dedup_by_timestamp(
    words: list[dict],
    last_partial_ms: int,
) -> tuple[list[dict], str]:
    fresh = [w for w in words if int(w.get("start_ms", 0)) >= last_partial_ms]
    text = " ".join(str(w.get("word", "")).strip() for w in fresh if str(w.get("word", "")).strip())
    return fresh, text


class PartialTranscriptService:

    def __init__(self, transcribe_async_fn: AsyncTranscribeFn) -> None:
        self._transcribe_async_fn = transcribe_async_fn

    async def generate_partial_async(
        self,
        session: SpeechSession,
        config: StreamSessionConfig,
    ) -> StreamTranscript | None:
        buf_samples = session.get_buffer_length()
        if buf_samples == 0:
            return None

        buf_ms = samples_to_ms(buf_samples)
        partial_window_ms = config.partial_window_ms
        partial_stride_ms = config.partial_stride_ms

        last_partial_ms, confirmed_words = session.get_partial_state()

        if buf_ms - last_partial_ms < partial_stride_ms or buf_ms < partial_window_ms:
            return None

        tail_window_ms = partial_window_ms + PARTIAL_OVERLAP_MS
        tail_samples = int(tail_window_ms * TARGET_SAMPLE_RATE / 1000)

        if buf_samples > tail_samples:
            tail_audio = session.get_buffer_tail(tail_samples)
            tail_start_ms = buf_ms - tail_window_ms
        else:
            tail_audio = session.get_buffer_audio()
            tail_start_ms = 0

        transcript = await self._transcribe_async_fn(
            audio=tail_audio,
            language=config.language,
            word_timestamps=config.include_word_timestamps,
        )

        transcript.words = _shift_words(transcript.words, tail_start_ms)
        transcript.segments = _shift_segments(transcript.segments, tail_start_ms)

        if transcript.words and config.include_word_timestamps:
            fresh_words, new_text = _dedup_by_timestamp(transcript.words, last_partial_ms)
            updated_words = confirmed_words + [
                str(w.get("word", "")) for w in fresh_words if str(w.get("word", "")).strip()
            ]
            session.update_partial(buf_ms, updated_words)
            if not fresh_words:
                return None
            transcript.words = fresh_words
            transcript.text = new_text
        else:
            new_text, updated_words = deduplicate_words(transcript.text, confirmed_words)
            session.update_partial(buf_ms, updated_words)
            if not new_text:
                return None
            transcript.text = new_text

        transcript.is_partial = True
        transcript.start_ms = tail_start_ms
        transcript.end_ms = buf_ms
        return transcript

    def flush_remaining_audio(self, session: SpeechSession) -> NDArray[np.float32] | None:
        if session.get_buffer_length() > 0:
            return session.get_buffer_audio()
        return None
