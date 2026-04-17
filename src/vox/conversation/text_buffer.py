"""Multilingual text chunker for TTS streaming.

Single source of truth for sentence / clause / word / character chunking across
the codebase — used by both the conversation streaming responses and the
long-form TTS route.

Public API:
  * StreamingTextBuffer  — incremental buffer for LLM deltas; yields sentence
    chunks as they form, with a soft-limit fallback for punctuation-free streams.
  * split_for_tts(text, max_chars)  — split a committed string so no chunk
    exceeds an adapter's max_input_chars cap.
  * split_sentences / split_clauses / split_long_sentence /
    split_by_words / split_by_chars  — language-neutral helpers covering
    Latin (.!?), CJK (。！？．), Devanagari (।), Arabic (؟).
"""

from __future__ import annotations

import re


_SENTENCE_TERMINATORS = ".!?。！？．।؟"

_CLAUSE_BREAKS = ",;:，；：、"
_SOFT_LIMIT_CHARS = 160


class StreamingTextBuffer:
    """Incrementally turns LLM deltas into TTS-sized chunks.

    The primary path is sentence-based: once we see a sentence terminator, we
    yield that sentence immediately. If the model streams a long answer without
    punctuation, a soft limit forces a chunk at a clause/whitespace boundary so
    TTS can begin before the full reply is finished.
    """

    def __init__(self, *, soft_limit_chars: int = _SOFT_LIMIT_CHARS) -> None:
        self._soft_limit_chars = max(1, soft_limit_chars)
        self._buffer = ""

    def push(self, text: str) -> list[str]:
        if text:
            self._buffer += text
        return self._drain(allow_partial=False)

    def flush(self) -> list[str]:
        chunks = self._drain(allow_partial=True)
        self._buffer = ""
        return chunks

    def _drain(self, *, allow_partial: bool) -> list[str]:
        chunks: list[str] = []
        while self._buffer:
            cut = _sentence_boundary(self._buffer)
            if cut is None and len(self._buffer) >= self._soft_limit_chars:
                cut = _soft_boundary(self._buffer, self._soft_limit_chars)

            if cut is None:
                if allow_partial:
                    chunk = self._buffer.strip()
                    if chunk:
                        chunks.append(chunk)
                    self._buffer = ""
                break

            chunk = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:]
            if chunk:
                chunks.append(chunk)

        return chunks


def split_for_tts(text: str, *, max_chars: int) -> list[str]:
    """Split a chunk so no single TTS call exceeds an adapter cap."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return [cleaned]

    sentences = split_sentences(cleaned)
    if not sentences:
        return [cleaned]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sentence) <= max_chars:
            current = sentence
            continue
        chunks.extend(split_long_sentence(sentence, max_chars))
    if current:
        chunks.append(current)
    return chunks


def _sentence_boundary(text: str) -> int | None:
    for idx, ch in enumerate(text):
        if ch in _SENTENCE_TERMINATORS:
            cut = idx + 1
            while cut < len(text) and text[cut] in "\"')]} \t\r\n":
                cut += 1
            return cut
    return None


def _soft_boundary(text: str, soft_limit_chars: int) -> int | None:
    window = text[:soft_limit_chars]
    for marker in ("\n\n", "\n"):
        idx = window.rfind(marker)
        if idx != -1:
            return idx + len(marker)
    for chars in (_CLAUSE_BREAKS, " \t"):
        for idx in range(len(window) - 1, -1, -1):
            if window[idx] in chars:
                cut = idx + 1
                while cut < len(text) and text[cut] in " \t":
                    cut += 1
                return cut
    return soft_limit_chars


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for ch in text.strip():
        current.append(ch)
        if ch in _SENTENCE_TERMINATORS:
            chunk = "".join(current).strip()
            if chunk:
                sentences.append(chunk)
            current = []
    if current:
        tail = "".join(current).strip()
        if tail:
            sentences.append(tail)
    return sentences


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    pieces = split_clauses(sentence)
    if len(pieces) == 1 and pieces[0] == sentence:
        if any(char.isspace() for char in sentence):
            return split_by_words(sentence, max_chars)
        return split_by_chars(sentence, max_chars)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current} {piece}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(piece) <= max_chars:
            current = piece
            continue
        chunks.extend(split_long_sentence(piece, max_chars))
    if current:
        chunks.append(current)
    return chunks


def split_clauses(sentence: str) -> list[str]:
    clauses: list[str] = []
    current: list[str] = []
    for ch in sentence:
        current.append(ch)
        if ch in _CLAUSE_BREAKS:
            chunk = "".join(current).strip()
            if chunk:
                clauses.append(chunk)
            current = []
    if current:
        tail = "".join(current).strip()
        if tail:
            clauses.append(tail)
    return clauses or [sentence]


def split_by_words(text: str, max_chars: int) -> list[str]:
    words = re.findall(r"\S+\s*", text)
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = current + word
        if current and len(candidate.strip()) > max_chars:
            chunks.append(current.strip())
            current = word
            continue
        current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text.strip()]


def split_by_chars(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        cleaned = text.strip()
        return [cleaned] if cleaned else []
    return [
        text[idx:idx + max_chars].strip()
        for idx in range(0, len(text), max_chars)
        if text[idx:idx + max_chars].strip()
    ]



_split_sentences = split_sentences
_split_clauses = split_clauses
_split_long_sentence = split_long_sentence
_split_by_words = split_by_words
_split_by_chars = split_by_chars
