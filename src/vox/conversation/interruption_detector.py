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


def candidate_timer_arming_ms(
    *,
    confirm_window_ms: int,
    false_interruption_timeout_ms: int,
    echo_exposed: bool,
    evidence_distrust_remaining_ms: int,
) -> int:
    timeout_ms = max(1, int(false_interruption_timeout_ms))
    if echo_exposed:
        return timeout_ms
    confirm_ms = max(1, int(confirm_window_ms))
    distrust_ms = max(0, int(evidence_distrust_remaining_ms))
    return min(max(confirm_ms, distrust_ms + confirm_ms), timeout_ms)


class InterruptionDecisionAction(StrEnum):
    DEFER = "defer"
    PROVISIONAL_REJECT = "provisional_reject"
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


@dataclass
class InterruptionCandidate:
    candidate_id: int
    utterance_id: int
    started_at: float
    assistant_text: str
    last_observed_at: float
    stopped_at: float | None = None
    cumulative_transcript: str = ""
    partial_revisions: int = 0
    latest_partial_duration_ms: int = 0
    final_transcript: str = ""
    status: InterruptionCandidateStatus = InterruptionCandidateStatus.PENDING
    decision_reason: str | None = None
    provisional_rejection_reason: str | None = None

    def vad_active_ms(self, now: float) -> int:
        end = self.stopped_at if self.stopped_at is not None else now
        return max(0, int((end - self.started_at) * 1000))


@runtime_checkable
class InterruptDetector(Protocol):
    """Owns one interruption candidate from VAD start through final STT."""

    def confirm_window_ms(self, base_ms: int, last_eou_probability: float | None) -> int: ...

    def wants_partials(self) -> bool: ...

    def is_self_echo(self, text: str, assistant_text: str) -> bool: ...

    def begin(
        self,
        *,
        utterance_id: int,
        started_at: float,
        assistant_text: str,
    ) -> InterruptionCandidate: ...

    def mark_speech_stopped(
        self,
        *,
        utterance_id: int,
        stopped_at: float,
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
        started_at: float,
        assistant_text: str,
    ) -> InterruptionCandidate:
        current = self._candidate
        if current is not None and self._matches(current, utterance_id):
            return current

        self._candidate_sequence += 1
        self._candidate = InterruptionCandidate(
            candidate_id=self._candidate_sequence,
            utterance_id=utterance_id,
            started_at=started_at,
            assistant_text=assistant_text,
            last_observed_at=started_at,
        )
        return self._candidate

    def mark_speech_stopped(
        self,
        *,
        utterance_id: int,
        stopped_at: float,
        expects_transcript: bool,
    ) -> InterruptionDecision:
        candidate = self._candidate_for(utterance_id)
        if candidate is None:
            return self._defer("speech_stopped_without_candidate")
        if candidate.status is not InterruptionCandidateStatus.PENDING:
            return self._defer("already_decided", candidate)

        candidate.stopped_at = stopped_at
        candidate.last_observed_at = stopped_at
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
            return self._defer("already_decided", candidate)

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

        if self.is_self_echo(text, assistant_text):
            return self._provisionally_reject(candidate, "self_echo_transcript")

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
            return self._defer("already_decided", candidate)

        text = transcript.text.strip()
        candidate.last_observed_at = now
        candidate.final_transcript = text
        candidate.assistant_text = assistant_text
        duration_ms = max(
            transcript_duration_ms(transcript),
            candidate.latest_partial_duration_ms,
            candidate.vad_active_ms(now),
        )

        if not text:
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "empty_final")
        if self.is_self_echo(text, assistant_text):
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
        if not self._candidate_is_current_and_pending(candidate):
            return self._defer("stale_final_decision", candidate)
        supported, reason = self._final_is_supported(
            strong_text=strong_text,
            word_count=word_count,
            acoustic_speech=acoustic_speech,
            partial_revisions=candidate.partial_revisions,
            eou_probability=transcript.eou_probability,
        )
        action = InterruptionDecisionAction.CONFIRM if supported else InterruptionDecisionAction.REJECT
        return self._decide(candidate, action, reason)

    @staticmethod
    def _final_is_supported(
        *,
        strong_text: bool,
        word_count: int,
        acoustic_speech: bool,
        partial_revisions: int,
        eou_probability: float | None,
    ) -> tuple[bool, str]:
        revised = partial_revisions > 0
        eou_support = eou_probability is not None and eou_probability >= 0.5
        if strong_text:
            if acoustic_speech or revised or eou_support:
                return True, "supported_final_transcript"
            return False, "final_transcript_without_support"
        if word_count == 1:
            if eou_probability is not None and eou_probability < 0.35:
                return False, "isolated_low_eou_final"
            if acoustic_speech and (revised or eou_support or eou_probability is None):
                return True, "supported_single_word_final"
            return False, "isolated_final_without_support"
        return False, "insufficient_final_evidence"

    async def evaluate_timeout(
        self,
        *,
        assistant_text: str,
        output_echo: bool,
        audio: NDArray[np.float32] | None,
        sample_rate: int,
        last_eou_probability: float | None,
        now: float,
    ) -> InterruptionDecision:
        candidate = self._candidate
        if candidate is None:
            return InterruptionDecision(
                action=InterruptionDecisionAction.REJECT,
                reason="timeout_without_candidate",
                candidate_id=None,
            )
        if candidate.status is not InterruptionCandidateStatus.PENDING:
            return self._defer("already_decided", candidate)

        candidate.assistant_text = assistant_text
        candidate.last_observed_at = now
        vad_active_ms = max(candidate.vad_active_ms(now), candidate.latest_partial_duration_ms)
        text = candidate.cumulative_transcript.strip()
        if self.is_self_echo(text, assistant_text):
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "self_echo_transcript")
        if self._strong_transcript(text, vad_active_ms):
            return self._decide(candidate, InterruptionDecisionAction.CONFIRM, "stable_partial")
        if output_echo:
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "output_echo_timeout")

        try:
            acoustic_speech = await self._classifier.is_real_interrupt(
                audio,
                text or None,
                last_eou_probability,
                vad_active_ms,
                sample_rate,
            )
        except Exception:
            if not self._candidate_is_current_and_pending(candidate):
                return self._defer("stale_timeout_decision", candidate)
            return self._decide(candidate, InterruptionDecisionAction.REJECT, "classifier_error")

        if not self._candidate_is_current_and_pending(candidate):
            return self._defer("stale_timeout_decision", candidate)
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

    def _candidate_is_current_and_pending(self, candidate: InterruptionCandidate) -> bool:
        return self._candidate is candidate and candidate.status is InterruptionCandidateStatus.PENDING

    @staticmethod
    def _matches(candidate: InterruptionCandidate, utterance_id: int) -> bool:
        return utterance_id > 0 and candidate.utterance_id == utterance_id

    def _strong_transcript(self, text: str, duration_ms: int) -> bool:
        required_words = max(2, int(self._policy.speaking_interrupt_min_words))
        return (
            transcript_word_count(text) >= required_words
            and duration_ms >= self._policy.speaking_interrupt_min_duration_ms
        )

    def is_self_echo(self, text: str, assistant_text: str) -> bool:
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
        )

    def _provisionally_reject(
        self,
        candidate: InterruptionCandidate,
        reason: str,
    ) -> InterruptionDecision:
        if candidate.provisional_rejection_reason == reason:
            return self._defer("provisional_rejection_already_observed", candidate)
        candidate.provisional_rejection_reason = reason
        return InterruptionDecision(
            action=InterruptionDecisionAction.PROVISIONAL_REJECT,
            reason=reason,
            candidate_id=candidate.candidate_id,
            vad_active_ms=max(
                candidate.vad_active_ms(candidate.last_observed_at),
                candidate.latest_partial_duration_ms,
            ),
            transcript=candidate.cumulative_transcript or None,
        )

    @staticmethod
    def _defer(
        reason: str,
        candidate: InterruptionCandidate | None = None,
    ) -> InterruptionDecision:
        return InterruptionDecision(
            action=InterruptionDecisionAction.DEFER,
            reason=reason,
            candidate_id=candidate.candidate_id if candidate is not None else None,
            vad_active_ms=(candidate.vad_active_ms(candidate.last_observed_at) if candidate is not None else 0),
            transcript=(candidate.cumulative_transcript or None) if candidate is not None else None,
        )
