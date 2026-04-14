from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from vox.streaming.session import SpeechSession
from vox.streaming.types import TARGET_SAMPLE_RATE, StreamSessionConfig, StreamTranscript, samples_to_ms
from vox.streaming.vad import SpeechSegment

logger = logging.getLogger(__name__)

PARTIAL_OVERLAP_MS = 300

TranscribeFn = Callable[..., StreamTranscript]
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
        else:
            tail_audio = session.get_buffer_audio()

        transcript = await self._transcribe_async_fn(
            audio=tail_audio,
            language=config.language,
            word_timestamps=config.include_word_timestamps,
        )

        new_text, updated_words = deduplicate_words(transcript.text, confirmed_words)
        session.update_partial(buf_ms, updated_words)

        if new_text:
            transcript.text = new_text
            transcript.is_partial = True
            return transcript
        return None

    def flush_remaining_audio(self, session: SpeechSession) -> NDArray[np.float32] | None:
        if session.get_buffer_length() > 0:
            return session.get_buffer_audio()
        return None
