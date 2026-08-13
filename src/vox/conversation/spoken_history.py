from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from vox.streaming.codecs import StreamResampler, float32_to_pcm16, pcm16_to_float32
from vox.streaming.types import TARGET_SAMPLE_RATE

MAX_PARTIAL_CAPTURE_BYTES = 512 * 1024
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SpokenHistorySnapshot:
    completed_text: str
    partial_source_text: str
    partial_audio_pcm16: bytes
    sample_rate: int
    playout_available: bool
    partial_capture_complete: bool
    played_audio_ms: int


@dataclass(frozen=True, slots=True)
class SpokenTextResolution:
    spoken_text: str
    partial_status: str
    played_audio_ms: int


@dataclass(slots=True)
class _SpokenSpan:
    span_id: int
    text: str
    registered_bytes: int = 0
    played_bytes: int = 0
    finished: bool = False
    capture_complete: bool = True
    capture: bytearray = field(default_factory=bytearray)
    resampler: StreamResampler = field(default_factory=lambda: StreamResampler(TARGET_SAMPLE_RATE))


class ResponseSpokenHistory:
    def __init__(
        self,
        *,
        playout_available: bool,
        max_partial_capture_bytes: int = MAX_PARTIAL_CAPTURE_BYTES,
    ) -> None:
        self._playout_available = bool(playout_available)
        self._max_partial_capture_bytes = max(1, int(max_partial_capture_bytes))
        self._next_span_id = 0
        self._spans: list[_SpokenSpan] = []
        self._completed_parts: list[str] = []
        self._invalid_playout = False

    @property
    def retained_audio_bytes(self) -> int:
        return sum(len(span.capture) for span in self._spans)

    def begin_span(self, text: str) -> int:
        self._next_span_id += 1
        self._spans.append(_SpokenSpan(span_id=self._next_span_id, text=text.strip()))
        return self._next_span_id

    def register_audio(self, span_id: int, pcm16_audio: bytes) -> None:
        span = self._find_span(span_id)
        if span is None or span.finished or not pcm16_audio:
            return
        span.registered_bytes += len(pcm16_audio) - (len(pcm16_audio) % 2)

    def finish_span(self, span_id: int) -> None:
        span = self._find_span(span_id)
        if span is None:
            return
        span.finished = True
        self._release_completed_spans()

    def discard_span(self, span_id: int) -> None:
        span = self._find_span(span_id)
        if span is None or span.registered_bytes or span.played_bytes:
            return
        self._spans.remove(span)

    def observe_playout(self, pcm16_audio: bytes, sample_rate: int) -> None:
        if not self._playout_available or not pcm16_audio or sample_rate <= 0:
            return
        audio = pcm16_audio[: len(pcm16_audio) - (len(pcm16_audio) % 2)]
        offset = 0
        while offset < len(audio):
            span = self._next_playable_span()
            if span is None:
                self._invalid_playout = True
                return
            remaining = span.registered_bytes - span.played_bytes
            if remaining <= 0:
                self._invalid_playout = True
                return
            take = min(remaining, len(audio) - offset)
            piece = audio[offset : offset + take]
            span.played_bytes += take
            offset += take
            self._capture(span, piece, sample_rate)
            self._release_completed_spans()

    def snapshot(self) -> SpokenHistorySnapshot:
        completed_text = _join_parts(self._completed_parts)
        if not self._playout_available or self._invalid_playout:
            return SpokenHistorySnapshot(
                completed_text="",
                partial_source_text="",
                partial_audio_pcm16=b"",
                sample_rate=TARGET_SAMPLE_RATE,
                playout_available=False,
                partial_capture_complete=False,
                played_audio_ms=0,
            )

        partial = next((span for span in self._spans if span.played_bytes > 0), None)
        if partial is None:
            return SpokenHistorySnapshot(
                completed_text=completed_text,
                partial_source_text="",
                partial_audio_pcm16=b"",
                sample_rate=TARGET_SAMPLE_RATE,
                playout_available=True,
                partial_capture_complete=True,
                played_audio_ms=0,
            )

        tail = float32_to_pcm16(partial.resampler.flush())
        capture_complete = partial.capture_complete
        if len(partial.capture) + len(tail) <= self._max_partial_capture_bytes:
            partial.capture.extend(tail)
        else:
            capture_complete = False
        played_audio_ms = int(len(partial.capture) / (TARGET_SAMPLE_RATE * 2) * 1000)
        return SpokenHistorySnapshot(
            completed_text=completed_text,
            partial_source_text=partial.text,
            partial_audio_pcm16=bytes(partial.capture),
            sample_rate=TARGET_SAMPLE_RATE,
            playout_available=True,
            partial_capture_complete=capture_complete,
            played_audio_ms=played_audio_ms,
        )

    def completed_text(self) -> str:
        return _join_parts(self._completed_parts)

    def _find_span(self, span_id: int) -> _SpokenSpan | None:
        return next((span for span in self._spans if span.span_id == span_id), None)

    def _next_playable_span(self) -> _SpokenSpan | None:
        return next((span for span in self._spans if span.played_bytes < span.registered_bytes), None)

    def _capture(self, span: _SpokenSpan, pcm16_audio: bytes, sample_rate: int) -> None:
        if not span.capture_complete:
            return
        resampled = span.resampler.process(pcm16_to_float32(pcm16_audio), sample_rate)
        encoded = float32_to_pcm16(resampled)
        if len(span.capture) + len(encoded) > self._max_partial_capture_bytes:
            span.capture_complete = False
            span.capture.clear()
            return
        span.capture.extend(encoded)

    def _release_completed_spans(self) -> None:
        while self._spans:
            span = self._spans[0]
            if not span.finished or span.played_bytes < span.registered_bytes:
                return
            self._spans.pop(0)
            if span.registered_bytes > 0 and span.text:
                self._completed_parts.append(span.text)


def resolve_spoken_text(
    snapshot: SpokenHistorySnapshot,
    transcript: str | None,
    *,
    resolver_failed: bool = False,
) -> SpokenTextResolution:
    if not snapshot.playout_available:
        return SpokenTextResolution("", "playout_unavailable", 0)
    if snapshot.partial_source_text and not snapshot.partial_capture_complete:
        return SpokenTextResolution(snapshot.completed_text, "partial_omitted", snapshot.played_audio_ms)
    if not snapshot.partial_source_text or not snapshot.partial_audio_pcm16:
        return SpokenTextResolution(snapshot.completed_text, "not_needed", snapshot.played_audio_ms)
    if resolver_failed:
        return SpokenTextResolution(snapshot.completed_text, "resolver_failed", snapshot.played_audio_ms)
    partial_prefix = longest_confident_source_prefix(snapshot.partial_source_text, transcript or "")
    if not partial_prefix:
        return SpokenTextResolution(snapshot.completed_text, "partial_omitted", snapshot.played_audio_ms)
    return SpokenTextResolution(
        _join_parts((snapshot.completed_text, partial_prefix)),
        "matched",
        snapshot.played_audio_ms,
    )


def longest_confident_source_prefix(source_text: str, transcript: str) -> str:
    source_words = _word_spans(source_text)
    transcript_words = [_normalize_word(match.group(0)) for match in _WORD_PATTERN.finditer(transcript)]
    transcript_words = [word for word in transcript_words if word]
    if not source_words or not transcript_words:
        return ""
    source_normalized = [normalized for _, _, normalized in source_words]
    first_score = SequenceMatcher(None, source_normalized[0], transcript_words[0]).ratio()
    if first_score < 0.65:
        return ""
    best_count = 0
    best_score = 0.0
    for count in range(1, min(len(source_words), len(transcript_words)) + 1):
        source_prefix = source_normalized[:count]
        transcript_prefix = transcript_words[:count]
        aligned_scores = [
            SequenceMatcher(None, source_word, transcript_word).ratio()
            for source_word, transcript_word in zip(source_prefix, transcript_prefix, strict=True)
        ]
        if min(aligned_scores) < 0.55 or aligned_scores[-1] < 0.65:
            continue
        token_score = SequenceMatcher(None, source_prefix, transcript_prefix).ratio()
        char_score = SequenceMatcher(None, " ".join(source_prefix), " ".join(transcript_prefix)).ratio()
        score = token_score * 0.65 + char_score * 0.35
        if score >= 0.78 and (score > best_score or count > best_count):
            best_count = count
            best_score = score
    if best_count == 0:
        return ""
    if best_count == len(source_words):
        return source_text.strip()
    return source_text[: source_words[best_count - 1][1]].strip()


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _WORD_PATTERN.finditer(text):
        normalized = _normalize_word(match.group(0))
        if normalized:
            spans.append((match.start(), match.end(), normalized))
    return spans


def _normalize_word(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _join_parts(parts) -> str:
    return " ".join(str(part).strip() for part in parts if str(part).strip()).strip()
