"""Merge chunked-transcription results back into a single TranscribeResult."""

from __future__ import annotations

from vox.core.types import TranscribeResult, TranscriptSegment, WordTimestamp


def merge_transcripts(
    results: list[tuple[TranscribeResult, int]],
) -> TranscribeResult:
    """Concatenate text and shift per-chunk timestamps by each chunk's offset.

    ``results`` is a list of ``(result, offset_ms)`` pairs in chunk order. Empty
    chunks (no text) are skipped from text concatenation but their duration still
    counts toward ``duration_ms``.
    """
    if not results:
        raise ValueError("Cannot merge an empty transcript list")

    if len(results) == 1:
        only, offset_ms = results[0]
        if offset_ms == 0:
            return only

    merged_text_parts: list[str] = []
    merged_segments: list[TranscriptSegment] = []
    total_duration_ms = 0
    model = ""
    language: str | None = None

    for result, offset_ms in results:
        stripped = result.text.strip() if result.text else ""
        if stripped:
            merged_text_parts.append(stripped)

        for seg in result.segments:
            shifted_words = tuple(
                WordTimestamp(
                    word=w.word,
                    start_ms=w.start_ms + offset_ms,
                    end_ms=w.end_ms + offset_ms,
                    confidence=w.confidence,
                )
                for w in seg.words
            )
            merged_segments.append(
                TranscriptSegment(
                    text=seg.text,
                    start_ms=seg.start_ms + offset_ms,
                    end_ms=seg.end_ms + offset_ms,
                    words=shifted_words,
                    language=seg.language,
                    confidence=seg.confidence,
                )
            )

        total_duration_ms += result.duration_ms
        if not model and result.model:
            model = result.model
        if language is None and result.language:
            language = result.language

    return TranscribeResult(
        text=" ".join(merged_text_parts),
        segments=tuple(merged_segments),
        language=language,
        duration_ms=total_duration_ms,
        model=model,
    )
