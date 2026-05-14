from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from vox.core.types import TranscribeResult, TranscriptSegment, WordTimestamp

STT_LEADING_CONTEXT_MS = 5_000


def add_stt_leading_context(
    audio: NDArray[np.float32],
    *,
    sample_rate: int,
    context_ms: int = STT_LEADING_CONTEXT_MS,
) -> tuple[NDArray[np.float32], int]:
    """Prepend silence to one-shot STT calls so model front-ends do not drop speech onset."""
    if context_ms <= 0 or audio.size == 0:
        return audio, 0

    context_samples = int(sample_rate * context_ms / 1000)
    if context_samples <= 0:
        return audio, 0

    padding = np.zeros(context_samples, dtype=np.float32)
    padded = np.concatenate([padding, audio.astype(np.float32, copy=False)])
    actual_context_ms = int(context_samples / sample_rate * 1000)
    return padded, actual_context_ms


def strip_stt_leading_context(
    result: TranscribeResult,
    *,
    context_ms: int,
    duration_ms: int,
) -> TranscribeResult:
    """Remove synthetic STT context from timestamps while preserving recognized text."""
    if context_ms <= 0:
        return replace(result, duration_ms=duration_ms)

    segments = _strip_segments(result.segments, context_ms=context_ms, duration_ms=duration_ms)
    return replace(result, segments=segments, duration_ms=duration_ms)


def _strip_segments(
    segments: tuple[TranscriptSegment, ...],
    *,
    context_ms: int,
    duration_ms: int,
) -> tuple[TranscriptSegment, ...]:
    if not segments:
        return ()

    includes_context = any(segment.end_ms > context_ms for segment in segments)
    stripped: list[TranscriptSegment] = []
    for segment in segments:
        if includes_context and segment.end_ms <= context_ms:
            continue

        start_ms, end_ms = _strip_time_range(
            segment.start_ms,
            segment.end_ms,
            context_ms=context_ms,
            duration_ms=duration_ms,
            includes_context=includes_context,
        )
        words = _strip_words(
            segment.words,
            context_ms=context_ms,
            duration_ms=duration_ms,
            includes_context=includes_context,
        )
        stripped.append(replace(segment, start_ms=start_ms, end_ms=end_ms, words=words))
    return tuple(stripped)


def _strip_words(
    words: tuple[WordTimestamp, ...],
    *,
    context_ms: int,
    duration_ms: int,
    includes_context: bool,
) -> tuple[WordTimestamp, ...]:
    if not words:
        return ()

    words_include_context = includes_context and any(word.end_ms > context_ms for word in words)
    stripped: list[WordTimestamp] = []
    for word in words:
        if words_include_context and word.end_ms <= context_ms:
            continue
        start_ms, end_ms = _strip_time_range(
            word.start_ms,
            word.end_ms,
            context_ms=context_ms,
            duration_ms=duration_ms,
            includes_context=words_include_context,
        )
        stripped.append(replace(word, start_ms=start_ms, end_ms=end_ms))
    return tuple(stripped)


def _strip_time_range(
    start_ms: int,
    end_ms: int,
    *,
    context_ms: int,
    duration_ms: int,
    includes_context: bool,
) -> tuple[int, int]:
    if includes_context:
        start_ms = max(0, start_ms - context_ms)
        end_ms = max(start_ms, end_ms - context_ms)

    start_ms = min(max(0, start_ms), duration_ms)
    end_ms = min(max(start_ms, end_ms), duration_ms)
    return start_ms, end_ms
