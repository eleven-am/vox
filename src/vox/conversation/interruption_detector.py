"""Stateful, content-independent interruption candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vox.conversation.interrupt import (
    InterruptClassifier,
    looks_like_self_echo,
    transcript_duration_ms,
    transcript_word_count,
)
from vox.conversation.types import TurnPolicy
from vox.streaming.types import StreamTranscript


class InterruptionDecisionAction(StrEnum):
    DEFER = "defer"
    CONFIRM = "confirm"
    REJECT = "reject"


class InterruptionCandidateStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class InterruptionDecision:
    action: InterruptionDecisionAction
    reason: str
    candidate_id: int | None
    vad_active_ms: int = 0
    transcript: str | None = None
    newly_decided: bool = False
    retry_after_ms: int = 0


@dataclass
class InterruptionCandidate:
    candidate_id: int
    utterance_id: int
    response_id: str | None
    started_at: float
    speech_started_ms: int
    assistant_text: str
    last_observed_at: float
    speech_active: bool = True
    stopped_at: float | None = None
    speech_stopped_ms: int | None = None
    cumulative_transcript: str = ""
    partial_revisions: int = 0
    latest_partial_duration_ms: int = 0
    final_transcript: str = ""
    final_eou_probability: float | None = None
    status: InterruptionCandidateStatus = InterruptionCandidateStatus.PENDING
    decision_reason: str | None = None

    def vad_active_ms(self, now: float) -> int:
        end = self.stopped_at if self.stopped_at is not None else now
        return max(0, int((end - self.started_at) * 1000))


@runtime_checkable
class InterruptDetector(Protocol):
    """Owns one interruption candidate from VAD start through final STT."""

    def confirm_window_ms(self, base_ms: int, last_eou_probability: float | None) -> int: ...

    def wants_partials(self) -> bool: ...

    def begin(
        self,
        *,
        utterance_id: int,
        response_id: str | None,
        started_at: float,
        speech_started_ms: int,
        assistant_text: str,
    ) -> InterruptionCandidate: ...

    def mark_speech_stopped(
        self,
        *,
        utterance_id: int,
        stopped_at: float,
        speech_stopped_ms: int,
        expects_transcript: bool,
    ) -> InterruptionDecision: ...

    async def observe_partial(
        self,
        transcript: StreamTranscript,
        *,
        cumulative_transcript: str,
        assistant_text: str,
        output_echo: bool,
        now: float,
    ) -> InterruptionDecision: ...

    async def observe_final(
        self,
        transcript: StreamTranscript,
        *,
        assistant_text: str,
        output_echo: bool,
        audio: NDArray[np.float32] | None,
        sample_rate: int,
        now: float,
    ) -> InterruptionDecision: ...

    async def evaluate_timeout(
        self,
        *,
        assistant_text: str,
        output_echo: bool,
        aec_warmup_remaining_ms: int,
        audio: NDArray[np.float32] | None,
        sample_rate: int,
        last_eou_probability: float | None,
        now: float,
    ) -> InterruptionDecision: ...

    def current(self) -> InterruptionCandidate | None: ...

    def finish(self, utterance_id: int) -> None: ...

    def reset(self) -> None: ...


class EvidenceBasedInterruptDetector:
    """Combines VAD, acoustic, transcript, EOU, and echo evidence.

    The detector owns candidate identity and terminal disposition. The supplied
    classifier remains the replaceable acoustic decision boundary, preserving
    compatibility with custom classifiers while keeping session orchestration
    independent of the default rules.
    """

    def __init__(self, *, policy: TurnPolicy, classifier: InterruptClassifier) -> None:
        self._policy = policy
        self._classifier = classifier
        self._candidate: InterruptionCandidate | None = None
        self._candidate_sequence = 0

    def confirm_window_ms(self, base_ms: int, last_eou_probability: float | None) -> int:
        return self._classifier.confirm_window_ms(base_ms, last_eou_probability)

    def wants_partials(self) -> bool:
        return bool(self._policy.partial_interrupts)

    def begin(
        self,
        *,
        utterance_id: int,
        response_id: str | None,
        started_at: float,
        speech_started_ms: int,
        assistant_text: str,
    ) -> InterruptionCandidate:
        current = self._candidate
        if current is not None and self._matches(current, utterance_id):
            return current

        self._candidate_sequence += 1
        self._candidate = InterruptionCandidate(
            candidate_id=self._candidate_sequence,
            utterance_id=utterance_id,
            response_id=response_id,
            started_at=started_at,
            speech_started_ms=speech_started_ms,
            assistant_text=assistant_text,
            last_observed_at=started_at,
        )
        return self._candidate

    def mark_speech_stopped(
        self,
        *,
        utterance_id: int,
        stopped_at: float,
        speech_stopped_ms: int,
        expects_transcript: bool,
    ) -> InterruptionDecision:
        candidate = self._candidate_for(utterance_id)
        if candidate is None:
            return self._defer("speech_stopped_without_candidate")
        if candidate.status is not InterruptionCandidateStatus.PENDING:
            return self._terminal(candidate)

        candidate.speech_active = False
        candidate.stopped_at = stopped_at
        candidate.last_observed_at = stopped_at
        candidate.speech_stopped_ms = speech_stopped_ms
        if not expects_transcript:
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "no_transcript")
        return self._defer("awaiting_final_transcript", candidate)

    async def observe_partial(
        self,
        transcript: StreamTranscript,
        *,
        cumulative_transcript: str,
        assistant_text: str,
        output_echo: bool,
        now: float,
    ) -> InterruptionDecision:
        candidate = self._candidate_for(transcript.utterance_id)
        if candidate is None:
            return self._defer("stale_partial")
        if candidate.status is not InterruptionCandidateStatus.PENDING:
            return self._terminal(candidate)

        text = cumulative_transcript.strip() or transcript.text.strip()
        candidate.last_observed_at = now
        if text and text != candidate.cumulative_transcript:
            candidate.partial_revisions += 1
            candidate.cumulative_transcript = text
        candidate.latest_partial_duration_ms = max(
            candidate.latest_partial_duration_ms,
            transcript_duration_ms(transcript),
            candidate.vad_active_ms(now),
        )
        candidate.assistant_text = assistant_text

        if self._is_self_echo(text, assistant_text):
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "self_echo_transcript")

        if text and self._classifier.should_short_circuit(text):
            return self._decide(
                candidate,
                InterruptionDecisionAction.CONFIRM,
                "custom_classifier_partial",
            )

        if self._strong_transcript(text, candidate.latest_partial_duration_ms):
            return self._decide(candidate, InterruptionDecisionAction.CONFIRM, "stable_partial")

        if output_echo:
            return self._defer("output_echo_needs_transcript", candidate)
        return self._defer("partial_needs_more_evidence", candidate)

    async def observe_final(
        self,
        transcript: StreamTranscript,
        *,
        assistant_text: str,
        output_echo: bool,
        audio: NDArray[np.float32] | None,
        sample_rate: int,
        now: float,
    ) -> InterruptionDecision:
        candidate = self._candidate_for(transcript.utterance_id)
        if candidate is None:
            return self._defer("stale_final")
        if candidate.status is not InterruptionCandidateStatus.PENDING:
            return self._terminal(candidate)

        text = transcript.text.strip()
        candidate.last_observed_at = now
        candidate.final_transcript = text
        candidate.final_eou_probability = transcript.eou_probability
        candidate.assistant_text = assistant_text
        duration_ms = max(
            transcript_duration_ms(transcript),
            candidate.latest_partial_duration_ms,
            candidate.vad_active_ms(now),
        )

        if not text:
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "empty_final")
        if self._is_self_echo(text, assistant_text):
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "self_echo_transcript")

        word_count = transcript_word_count(text)
        strong_text = self._strong_transcript(text, duration_ms)
        if output_echo and not strong_text:
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "output_echo")

        acoustic_speech = await self._classifier.is_real_interrupt(
            audio,
            text,
            transcript.eou_probability,
            duration_ms,
            sample_rate,
        )
        eou_support = transcript.eou_probability is not None and transcript.eou_probability >= 0.5
        if strong_text:
            if acoustic_speech or candidate.partial_revisions > 0 or eou_support:
                return self._decide(
                    candidate,
                    InterruptionDecisionAction.CONFIRM,
                    "supported_final_transcript",
                )
            return self._decide(
                candidate,
                InterruptionDecisionAction.REJECT,
                "final_transcript_without_support",
            )

        if word_count == 1:
            eou = transcript.eou_probability
            if eou is not None and eou < 0.35:
                return self._decide(
                    candidate,
                    InterruptionDecisionAction.REJECT,
                    "isolated_low_eou_final",
                )
            stable_single_word = candidate.partial_revisions > 0
            high_eou = eou is not None and eou >= 0.5
            if acoustic_speech and (stable_single_word or high_eou or eou is None):
                return self._decide(
                    candidate,
                    InterruptionDecisionAction.CONFIRM,
                    "supported_single_word_final",
                )
            return self._decide(
                candidate,
                InterruptionDecisionAction.REJECT,
                "isolated_final_without_support",
            )

        return self._decide(
            candidate,
            InterruptionDecisionAction.REJECT,
            "insufficient_final_evidence",
        )

    async def evaluate_timeout(
        self,
        *,
        assistant_text: str,
        output_echo: bool,
        aec_warmup_remaining_ms: int,
        audio: NDArray[np.float32] | None,
        sample_rate: int,
        last_eou_probability: float | None,
        now: float,
    ) -> InterruptionDecision:
        candidate = self._candidate
        if candidate is None:
            return self._defer("timeout_without_candidate")
        if candidate.status is not InterruptionCandidateStatus.PENDING:
            return self._terminal(candidate)

        candidate.assistant_text = assistant_text
        candidate.last_observed_at = now
        vad_active_ms = max(candidate.vad_active_ms(now), candidate.latest_partial_duration_ms)
        candidate_age_ms = max(0, int((now - candidate.started_at) * 1000))
        evidence_wait_ms = max(1, int(self._policy.false_interruption_timeout_ms))
        text = candidate.cumulative_transcript.strip()
        if self._is_self_echo(text, assistant_text):
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "self_echo_transcript")
        if self._strong_transcript(text, vad_active_ms):
            return self._decide(candidate, InterruptionDecisionAction.CONFIRM, "stable_partial")
        if output_echo and candidate_age_ms < evidence_wait_ms:
            return self._defer(
                "output_echo_waiting_for_transcript",
                candidate,
                retry_after_ms=evidence_wait_ms - candidate_age_ms,
            )
        if output_echo:
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "output_echo_timeout")
        if aec_warmup_remaining_ms > 0 and not text and candidate_age_ms < evidence_wait_ms:
            return self._defer(
                "aec_warmup_waiting_for_transcript",
                candidate,
                retry_after_ms=min(
                    max(1, int(aec_warmup_remaining_ms)),
                    evidence_wait_ms - candidate_age_ms,
                ),
            )

        try:
            acoustic_speech = await self._classifier.is_real_interrupt(
                audio,
                text or None,
                last_eou_probability,
                vad_active_ms,
                sample_rate,
            )
        except Exception:
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "classifier_error")

        if acoustic_speech:
            return self._decide(candidate, InterruptionDecisionAction.CONFIRM, "acoustic_speech")
        return self._decide(candidate, InterruptionDecisionAction.REJECT, "insufficient_acoustic_evidence")

    def current(self) -> InterruptionCandidate | None:
        if self._candidate is None:
            return None
        return replace(self._candidate)

    def finish(self, utterance_id: int) -> None:
        if self._candidate_for(utterance_id) is not None:
            self._candidate = None

    def reset(self) -> None:
        self._candidate = None

    def _candidate_for(self, utterance_id: int) -> InterruptionCandidate | None:
        candidate = self._candidate
        if candidate is None or not self._matches(candidate, utterance_id):
            return None
        return candidate

    @staticmethod
    def _matches(candidate: InterruptionCandidate, utterance_id: int) -> bool:
        return utterance_id > 0 and candidate.utterance_id == utterance_id

    def _strong_transcript(self, text: str, duration_ms: int) -> bool:
        required_words = max(2, int(self._policy.speaking_interrupt_min_words))
        return (
            transcript_word_count(text) >= required_words
            and duration_ms >= self._policy.speaking_interrupt_min_duration_ms
        )

    def _is_self_echo(self, text: str, assistant_text: str) -> bool:
        return looks_like_self_echo(
            text,
            assistant_text,
            min_words=self._policy.self_echo_min_words,
            min_overlap=self._policy.self_echo_min_overlap,
        )

    def _decide(
        self,
        candidate: InterruptionCandidate,
        action: InterruptionDecisionAction,
        reason: str,
    ) -> InterruptionDecision:
        candidate.status = (
            InterruptionCandidateStatus.CONFIRMED
            if action is InterruptionDecisionAction.CONFIRM
            else InterruptionCandidateStatus.REJECTED
        )
        candidate.decision_reason = reason
        return InterruptionDecision(
            action=action,
            reason=reason,
            candidate_id=candidate.candidate_id,
            vad_active_ms=max(
                candidate.vad_active_ms(candidate.last_observed_at),
                candidate.latest_partial_duration_ms,
            ),
            transcript=(candidate.final_transcript or candidate.cumulative_transcript or None),
            newly_decided=True,
        )

    def _terminal(self, candidate: InterruptionCandidate) -> InterruptionDecision:
        action = (
            InterruptionDecisionAction.CONFIRM
            if candidate.status is InterruptionCandidateStatus.CONFIRMED
            else InterruptionDecisionAction.REJECT
        )
        return InterruptionDecision(
            action=action,
            reason=candidate.decision_reason or candidate.status.value,
            candidate_id=candidate.candidate_id,
            vad_active_ms=max(
                candidate.vad_active_ms(candidate.last_observed_at),
                candidate.latest_partial_duration_ms,
            ),
            transcript=(candidate.final_transcript or candidate.cumulative_transcript or None),
            newly_decided=False,
        )

    @staticmethod
    def _defer(
        reason: str,
        candidate: InterruptionCandidate | None = None,
        *,
        retry_after_ms: int = 0,
    ) -> InterruptionDecision:
        return InterruptionDecision(
            action=InterruptionDecisionAction.DEFER,
            reason=reason,
            candidate_id=candidate.candidate_id if candidate is not None else None,
            vad_active_ms=(candidate.vad_active_ms(candidate.last_observed_at) if candidate is not None else 0),
            transcript=(candidate.cumulative_transcript or None) if candidate is not None else None,
            retry_after_ms=max(0, int(retry_after_ms)),
        )
