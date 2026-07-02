from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from vox.streaming.types import StreamTranscript

WIRE_TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"
TRANSCRIPT_REVISION_SIMILARITY = 0.78


def normalise_transcript_text(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def is_transcript_revision(previous: str, current: str) -> bool:
    previous_norm = normalise_transcript_text(previous)
    current_norm = normalise_transcript_text(current)
    if not previous_norm or not current_norm:
        return False
    if previous_norm in current_norm or current_norm in previous_norm:
        return True
    return SequenceMatcher(None, previous_norm, current_norm).ratio() >= TRANSCRIPT_REVISION_SIMILARITY


def append_transcript_text(previous: str, current: str) -> str:
    previous = previous.strip()
    current = current.strip()
    if not previous:
        return current
    if not current:
        return previous
    return f"{previous} {current}"


def transcript_done_payload(transcript: StreamTranscript, *, language: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": transcript.text,
        "language": language,
        "start_ms": transcript.start_ms,
        "end_ms": transcript.end_ms,
    }
    if transcript.eou_probability is not None:
        payload["eou_probability"] = transcript.eou_probability
    if transcript.entities:
        payload["entities"] = transcript.entities
    if transcript.topics:
        payload["topics"] = transcript.topics
    if transcript.words:
        payload["words"] = transcript.words
    return payload


def coalesce_transcript_payload(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_text = str(previous.get("transcript") or "")
    current_text = str(current.get("transcript") or "")

    if is_transcript_revision(previous_text, current_text):
        previous_norm = normalise_transcript_text(previous_text)
        current_norm = normalise_transcript_text(current_text)
        if len(current_norm) >= len(previous_norm):
            return dict(current)
        return dict(previous)

    merged = dict(current)
    merged["transcript"] = append_transcript_text(previous_text, current_text)
    merged["start_ms"] = int(previous.get("start_ms", current.get("start_ms", 0)) or 0)
    merged["end_ms"] = max(
        int(previous.get("end_ms", 0) or 0),
        int(current.get("end_ms", 0) or 0),
    )

    if "eou_probability" not in merged and "eou_probability" in previous:
        merged["eou_probability"] = previous["eou_probability"]

    previous_words = previous.get("words") or []
    current_words = current.get("words") or []
    if previous_words or current_words:
        merged["words"] = [*previous_words, *current_words]

    previous_topics = previous.get("topics") or []
    current_topics = current.get("topics") or []
    if previous_topics or current_topics:
        merged["topics"] = list(dict.fromkeys([*previous_topics, *current_topics]))

    previous_entities = previous.get("entities") or []
    current_entities = current.get("entities") or []
    if previous_entities or current_entities:
        merged["entities"] = [*previous_entities, *current_entities]

    return merged


@dataclass
class PendingTranscriptFinalizer:
    language: str
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("vox.conversation.session"))
    pending: dict[str, Any] | None = None

    def remember_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.pending is None:
            self.pending = dict(payload)
        else:
            self.pending = coalesce_transcript_payload(self.pending, payload)
        return self.pending

    def remember(self, transcript: StreamTranscript) -> dict[str, Any]:
        return self.remember_payload(transcript_done_payload(transcript, language=self.language))

    def pending_text(self, default: str = "") -> str:
        if self.pending is None:
            return default
        return str(self.pending.get("transcript", default))

    def clear(self) -> None:
        self.pending = None

    def pop(self) -> dict[str, Any] | None:
        payload = self.pending
        self.pending = None
        return payload

    def log(self, payload: dict[str, Any]) -> None:
        self.logger.info(
            "conversation final transcript emitted text=%r start_ms=%s end_ms=%s "
            "eou_probability=%s topics=%d entities=%d words=%d",
            str(payload.get("transcript") or ""),
            payload.get("start_ms"),
            payload.get("end_ms"),
            payload.get("eou_probability"),
            len(payload.get("topics") or ()),
            len(payload.get("entities") or ()),
            len(payload.get("words") or ()),
        )
