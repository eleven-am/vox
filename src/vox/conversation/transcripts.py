from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.audio.pipeline import AudioChunk
from vox.conversation.types import TimerKey, TurnEvent, TurnEventType, TurnPolicy
from vox.speech_context.types import speech_context_payload
from vox.streaming.types import TARGET_SAMPLE_RATE, StreamTranscript, samples_to_ms

WIRE_TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"
WIRE_TURN_EOU_PREDICTED = "turn.eou.predicted"
TRANSCRIPT_REVISION_SIMILARITY = 0.78
TRANSCRIPT_CONTINUATION_COMMIT_MS = 650
TRANSCRIPT_LOW_EOU_MAX_EXTENSION_MS = 150


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
    if transcript.speech_context is not None:
        payload["speech_context"] = speech_context_payload(transcript.speech_context)
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
    merged.pop("speech_context", None)
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
    pending_audio: list[AudioChunk] = field(default_factory=list)
    pending_turn_audio: dict[int, AudioChunk] = field(default_factory=dict)

    def remember_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.pending is None:
            self.pending = dict(payload)
        else:
            self.pending = coalesce_transcript_payload(self.pending, payload)
        return self.pending

    def remember(self, transcript: StreamTranscript) -> dict[str, Any]:
        payload = transcript_done_payload(transcript, language=self.language)
        audio = self._audio_chunk(transcript)
        if self.pending is None:
            self.pending = dict(payload)
            self.pending_audio = [audio] if audio is not None else []
            return self.pending

        previous_text = str(self.pending.get("transcript") or "")
        if is_transcript_revision(previous_text, transcript.text):
            previous_norm = normalise_transcript_text(previous_text)
            current_norm = normalise_transcript_text(transcript.text)
            if len(current_norm) >= len(previous_norm):
                self.pending = dict(payload)
                self.pending_audio = [audio] if audio is not None else []
            return self.pending

        self.pending = coalesce_transcript_payload(self.pending, payload)
        if audio is not None:
            self.pending_audio.append(audio)
        return self.pending

    def pending_text(self, default: str = "") -> str:
        if self.pending is None:
            return default
        return str(self.pending.get("transcript", default))

    def remember_turn_audio(
        self,
        *,
        utterance_id: int,
        audio: NDArray[np.float32] | None,
        start_ms: int,
    ) -> None:
        if utterance_id <= 0 or audio is None or audio.size == 0:
            return
        self.pending_turn_audio[utterance_id] = AudioChunk(
            data=audio,
            sample_rate=TARGET_SAMPLE_RATE,
            duration_ms=samples_to_ms(audio.size),
            offset_ms=start_ms,
        )

    def discard_turn_audio(self, utterance_id: int) -> None:
        self.pending_turn_audio.pop(utterance_id, None)

    def clear(self) -> None:
        self.pending = None
        self.pending_audio.clear()
        self.pending_turn_audio.clear()

    def pop(self) -> dict[str, Any] | None:
        payload, _ = self.pop_with_audio()
        return payload

    def pop_with_audio(self) -> tuple[dict[str, Any] | None, tuple[AudioChunk, ...]]:
        payload = self.pending
        audio = (
            tuple(sorted(self.pending_turn_audio.values(), key=lambda chunk: chunk.offset_ms))
            if self.pending_turn_audio
            else tuple(self.pending_audio)
        )
        self.pending = None
        self.pending_audio.clear()
        self.pending_turn_audio.clear()
        return payload, audio

    @staticmethod
    def _audio_chunk(transcript: StreamTranscript) -> AudioChunk | None:
        if transcript.audio is None:
            return None
        return AudioChunk(
            data=transcript.audio,
            sample_rate=TARGET_SAMPLE_RATE,
            duration_ms=samples_to_ms(transcript.audio.size),
            offset_ms=transcript.start_ms,
        )

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


@dataclass(frozen=True)
class EndpointCommitDelayPolicy:
    """Computes how long endpointing should wait before committing a transcript."""

    max_delay_ms: int
    min_delay_ms: int
    dynamic_endpointing: bool
    continuation_delay_ms: int = TRANSCRIPT_CONTINUATION_COMMIT_MS

    @classmethod
    def from_turn_policy(cls, policy: TurnPolicy) -> EndpointCommitDelayPolicy:
        return cls(
            max_delay_ms=policy.max_endpointing_delay_ms,
            min_delay_ms=policy.min_endpointing_delay_ms,
            dynamic_endpointing=policy.dynamic_endpointing,
        )

    def commit_delay_ms(
        self,
        *,
        recent_pause_ms: list[int] | tuple[int, ...] = (),
        eou_probability: float | None = None,
        eou_threshold: float | None = None,
    ) -> int:
        max_delay_ms = max(0, int(self.max_delay_ms))
        min_delay_ms = max(0, int(self.min_delay_ms))
        continuation_delay_ms = max(
            min_delay_ms,
            max(0, int(self.continuation_delay_ms)),
        )
        if self.dynamic_endpointing and recent_pause_ms:
            avg_pause = sum(recent_pause_ms) / len(recent_pause_ms)
            dynamic_ms = int(avg_pause * 1.25)
            base_ms = min(
                max_delay_ms,
                max(
                    continuation_delay_ms,
                    dynamic_ms,
                ),
            )
        else:
            base_ms = min(max_delay_ms, continuation_delay_ms)

        if eou_probability is None or eou_threshold is None:
            return base_ms
        if eou_probability < eou_threshold:
            if eou_threshold <= 0.0:
                return base_ms
            incompletion = min(
                1.0,
                max(0.0, (eou_threshold - eou_probability) / eou_threshold),
            )
            available_extension_ms = max(0, max_delay_ms - base_ms)
            extension_ms = min(
                TRANSCRIPT_LOW_EOU_MAX_EXTENSION_MS,
                available_extension_ms,
            )
            return round(base_ms + incompletion * extension_ms)

        floor_ms = min(min_delay_ms, base_ms)
        if eou_threshold >= 1.0:
            return floor_ms
        confidence = min(1.0, max(0.0, (eou_probability - eou_threshold) / (1.0 - eou_threshold)))
        return int(base_ms - confidence * (base_ms - floor_ms))


def eou_indicates_incomplete_turn(
    eou_probability: float | None,
    *,
    threshold: float,
) -> bool:
    return eou_probability is not None and eou_probability < threshold


@dataclass
class EndpointPauseHistory:
    max_items: int = 8
    pauses_ms: list[int] = field(default_factory=list)

    def record_since(self, stopped_at: float | None, *, now: float | None = None) -> None:
        if stopped_at is None:
            return
        current_time = time.monotonic() if now is None else now
        pause_ms = max(0, int((current_time - stopped_at) * 1000))
        self.pauses_ms.append(pause_ms)
        self.pauses_ms = self.pauses_ms[-max(1, int(self.max_items)) :]

    def values(self) -> tuple[int, ...]:
        return tuple(self.pauses_ms)


@dataclass(frozen=True)
class FinalTranscriptDecision:
    commit_delay_ms: int
    defer_commit: bool
    eou_complete: bool
    eou_event: dict[str, Any] | None


def final_transcript_decision(
    transcript: StreamTranscript,
    *,
    endpoint_timer_active: bool,
    commit_delay_policy: EndpointCommitDelayPolicy,
    recent_pause_ms: list[int] | tuple[int, ...],
    eou_threshold: float,
    turn_detector: str,
) -> FinalTranscriptDecision:
    eou_probability = float(transcript.eou_probability) if transcript.eou_probability is not None else None
    commit_delay_ms = (
        commit_delay_policy.commit_delay_ms(
            recent_pause_ms=recent_pause_ms,
            eou_probability=eou_probability,
            eou_threshold=eou_threshold,
        )
        if endpoint_timer_active
        else 0
    )
    defer_commit = endpoint_timer_active and commit_delay_ms > 0
    eou_complete = eou_probability is not None and eou_probability >= eou_threshold

    if eou_probability is None:
        return FinalTranscriptDecision(
            commit_delay_ms=commit_delay_ms,
            defer_commit=defer_commit,
            eou_complete=False,
            eou_event=None,
        )

    return FinalTranscriptDecision(
        commit_delay_ms=commit_delay_ms,
        defer_commit=defer_commit,
        eou_complete=eou_complete,
        eou_event={
            "type": WIRE_TURN_EOU_PREDICTED,
            "probability": eou_probability,
            "threshold": eou_threshold,
            "decision": "complete" if eou_complete else "incomplete",
            "action": "wait" if defer_commit else "commit",
            "delay_ms": commit_delay_ms,
            "turn_detector": turn_detector,
            "start_ms": transcript.start_ms,
            "end_ms": transcript.end_ms,
        },
    )


def should_wait_for_pending_final_transcript(
    event: TurnEvent,
    *,
    awaiting_final_transcript: bool,
    awaiting_started_at: float,
    max_endpointing_delay_ms: int,
    now: float | None = None,
) -> bool:
    if event.type != TurnEventType.TIMER_ELAPSED:
        return False
    if event.payload.get("key") != TimerKey.ENDPOINTING.value:
        return False
    if not awaiting_final_transcript:
        return False
    if awaiting_started_at <= 0.0:
        return True

    current_time = time.monotonic() if now is None else now
    max_wait_ms = max(
        max_endpointing_delay_ms,
        TRANSCRIPT_CONTINUATION_COMMIT_MS,
    )
    waited_ms = int((current_time - awaiting_started_at) * 1000)
    return waited_ms < max_wait_ms
