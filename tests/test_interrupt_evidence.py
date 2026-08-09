from __future__ import annotations

import asyncio

import numpy as np
import pytest

from vox.conversation.interrupt import HeuristicInterruptClassifier
from vox.conversation.interruption_detector import (
    EvidenceBasedInterruptDetector,
    InterruptionCandidateStatus,
    InterruptionDecisionAction,
    candidate_evidence_deadline_ms,
)
from vox.conversation.types import TurnPolicy
from vox.streaming.types import StreamTranscript

SAMPLE_RATE = 16_000


class _GatedClassifier:
    def __init__(self, *, result: bool) -> None:
        self.result = result
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def confirm_window_ms(self, base_ms: int, _last_eou_probability: float | None) -> int:
        return base_ms

    def should_short_circuit(self, _transcript: str) -> bool:
        return False

    async def is_real_interrupt(self, *_args) -> bool:
        self.started.set()
        await self.release.wait()
        return self.result


def _voice(duration_ms: int, *, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(duration_ms * SAMPLE_RATE // 1000) / SAMPLE_RATE
    return (amplitude * (np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t))).astype(np.float32)


def _noise(duration_ms: int, *, seed: int = 0) -> np.ndarray:
    return (
        np.random.default_rng(seed)
        .normal(
            0,
            0.15,
            duration_ms * SAMPLE_RATE // 1000,
        )
        .astype(np.float32)
    )


def _detector(**policy_overrides) -> EvidenceBasedInterruptDetector:
    policy = TurnPolicy(
        speaking_interrupt_min_duration_ms=420,
        speaking_interrupt_min_words=2,
        **policy_overrides,
    )
    return EvidenceBasedInterruptDetector(
        policy=policy,
        classifier=HeuristicInterruptClassifier(),
    )


def _begin(detector: EvidenceBasedInterruptDetector, *, utterance_id: int = 1) -> None:
    detector.begin(
        utterance_id=utterance_id,
        started_at=10.0,
        assistant_text="The assistant is still speaking.",
    )


def test_timer_arming_uses_one_evidence_deadline() -> None:
    assert (
        candidate_evidence_deadline_ms(
            false_interruption_timeout_ms=2000,
        )
        == 2000
    )


def test_timer_arming_is_bounded_by_evidence_timeout() -> None:
    assert (
        candidate_evidence_deadline_ms(
            false_interruption_timeout_ms=600,
        )
        == 600
    )


def test_timer_arming_clamps_non_positive_timeout() -> None:
    assert (
        candidate_evidence_deadline_ms(
            false_interruption_timeout_ms=0,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_output_echo_candidate_confirms_on_late_genuine_final() -> None:
    detector = _detector()
    _begin(detector)

    detector.mark_speech_stopped(utterance_id=1, stopped_at=10.5, expects_transcript=True)
    decision = await detector.observe_final(
        StreamTranscript(
            text="No wait I need to change that.",
            audio_duration_ms=500,
            eou_probability=0.8,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=True,
        audio=_voice(500),
        sample_rate=SAMPLE_RATE,
        now=10.5,
    )

    assert decision.action is InterruptionDecisionAction.CONFIRM
    assert decision.reason == "supported_final_transcript"


@pytest.mark.asyncio
async def test_cumulative_partial_confirms_without_keyword_semantics() -> None:
    detector = _detector()
    _begin(detector)

    first = await detector.observe_partial(
        StreamTranscript(text="I need", is_partial=True, audio_duration_ms=250, utterance_id=1),
        cumulative_transcript="I need",
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        now=10.25,
    )
    second = await detector.observe_partial(
        StreamTranscript(text="to add something", is_partial=True, audio_duration_ms=700, utterance_id=1),
        cumulative_transcript="I need to add something",
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        now=10.70,
    )

    assert first.action is InterruptionDecisionAction.DEFER
    assert second.action is InterruptionDecisionAction.CONFIRM
    assert second.reason == "stable_partial"


@pytest.mark.asyncio
async def test_isolated_low_eou_final_is_rejected_even_when_stt_returns_a_word() -> None:
    detector = _detector()
    _begin(detector)

    decision = await detector.observe_final(
        StreamTranscript(
            text="Anyway.",
            audio_duration_ms=691,
            eou_probability=0.00005,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_noise(691, seed=1),
        sample_rate=SAMPLE_RATE,
        now=10.691,
    )

    assert decision.action is InterruptionDecisionAction.REJECT
    assert decision.reason == "isolated_low_eou_final"


@pytest.mark.asyncio
async def test_natural_single_word_final_with_eou_and_voice_is_preserved() -> None:
    detector = _detector()
    _begin(detector)

    decision = await detector.observe_final(
        StreamTranscript(
            text="No.",
            audio_duration_ms=650,
            eou_probability=0.92,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(650),
        sample_rate=SAMPLE_RATE,
        now=10.65,
    )

    assert decision.action is InterruptionDecisionAction.CONFIRM
    assert decision.reason == "supported_single_word_final"


@pytest.mark.asyncio
async def test_multiword_final_confirms_even_when_user_is_not_at_eou() -> None:
    detector = _detector()
    _begin(detector)

    decision = await detector.observe_final(
        StreamTranscript(
            text="What I mean is",
            audio_duration_ms=1100,
            eou_probability=0.08,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(1100),
        sample_rate=SAMPLE_RATE,
        now=11.1,
    )

    assert decision.action is InterruptionDecisionAction.CONFIRM
    assert decision.reason == "supported_final_transcript"


@pytest.mark.asyncio
async def test_multiword_noise_hallucination_needs_independent_support() -> None:
    detector = _detector()
    _begin(detector)

    decision = await detector.observe_final(
        StreamTranscript(
            text="Okay anyway",
            audio_duration_ms=800,
            eou_probability=0.01,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_noise(800, seed=11),
        sample_rate=SAMPLE_RATE,
        now=10.8,
    )

    assert decision.action is InterruptionDecisionAction.REJECT
    assert decision.reason == "final_transcript_without_support"


@pytest.mark.asyncio
async def test_timeout_never_confirms_without_transcript_evidence() -> None:
    voice_detector = _detector()
    _begin(voice_detector)
    voice = await voice_detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(600),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=10.6,
    )

    noise_detector = _detector()
    _begin(noise_detector)
    noise = await noise_detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_noise(600, seed=4),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=10.6,
    )

    assert voice.action is InterruptionDecisionAction.PROVISIONAL_REJECT
    assert voice.reason == "no_transcript_timeout"
    assert noise.action is InterruptionDecisionAction.PROVISIONAL_REJECT
    assert noise.reason == "no_transcript_timeout"
    assert voice_detector.current().status is InterruptionCandidateStatus.PENDING
    assert noise_detector.current().status is InterruptionCandidateStatus.PENDING


@pytest.mark.asyncio
async def test_sustained_speech_confirms_when_transcript_evidence_arrives() -> None:
    detector = _detector()
    _begin(detector)

    decision = await detector.observe_partial(
        StreamTranscript(
            text="Please stop",
            is_partial=True,
            audio_duration_ms=700,
            utterance_id=1,
        ),
        cumulative_transcript="Please stop",
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        now=10.7,
    )

    assert decision.action is InterruptionDecisionAction.CONFIRM
    assert decision.reason == "stable_partial"


@pytest.mark.asyncio
async def test_self_echo_and_output_echo_reject_without_cancelling() -> None:
    detector = _detector()
    _begin(detector)
    self_echo = await detector.observe_partial(
        StreamTranscript(
            text="assistant is still speaking",
            is_partial=True,
            audio_duration_ms=800,
            utterance_id=1,
        ),
        cumulative_transcript="assistant is still speaking",
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        now=10.8,
    )
    assert self_echo.action is InterruptionDecisionAction.PROVISIONAL_REJECT

    output_detector = _detector()
    _begin(output_detector)
    output_echo = await output_detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=True,
        audio=_voice(600),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=10.6,
    )
    assert output_echo.action is InterruptionDecisionAction.PROVISIONAL_REJECT
    assert output_echo.reason == "no_transcript_timeout"

    repeated = await output_detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=True,
        audio=_voice(2100),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=12.1,
    )
    assert repeated.action is InterruptionDecisionAction.DEFER
    assert repeated.reason == "provisional_rejection_already_observed"


@pytest.mark.asyncio
async def test_stale_final_cannot_confirm_a_new_candidate() -> None:
    detector = _detector()
    _begin(detector, utterance_id=1)
    detector.begin(
        utterance_id=2,
        started_at=11.0,
        assistant_text="The assistant is still speaking.",
    )

    decision = await detector.observe_final(
        StreamTranscript(
            text="This belongs to the old candidate",
            audio_duration_ms=900,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(900),
        sample_rate=SAMPLE_RATE,
        now=11.2,
    )

    assert decision.action is InterruptionDecisionAction.DEFER
    assert decision.reason == "stale_final"
    assert detector.current().utterance_id == 2


@pytest.mark.asyncio
async def test_final_classifier_result_cannot_decide_replacement_candidate() -> None:
    classifier = _GatedClassifier(result=True)
    detector = EvidenceBasedInterruptDetector(
        policy=TurnPolicy(
            speaking_interrupt_min_duration_ms=420,
            speaking_interrupt_min_words=2,
        ),
        classifier=classifier,
    )
    _begin(detector, utterance_id=1)

    pending = asyncio.create_task(
        detector.observe_final(
            StreamTranscript(
                text="Please let me finish",
                audio_duration_ms=900,
                eou_probability=0.9,
                utterance_id=1,
            ),
            assistant_text="The assistant is still speaking.",
            output_echo=False,
            audio=_voice(900),
            sample_rate=SAMPLE_RATE,
            now=10.9,
        )
    )
    await classifier.started.wait()
    replacement = detector.begin(
        utterance_id=2,
        started_at=11.0,
        assistant_text="The assistant is still speaking.",
    )
    classifier.release.set()

    decision = await pending

    assert decision.action is InterruptionDecisionAction.DEFER
    assert detector.current() == replacement


@pytest.mark.asyncio
async def test_timeout_is_provisional_until_empty_final_terminally_rejects() -> None:
    detector = _detector()
    _begin(detector)

    timeout = await detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(600),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=10.6,
    )
    final = await detector.observe_final(
        StreamTranscript(
            text="",
            audio_duration_ms=600,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(600),
        sample_rate=SAMPLE_RATE,
        now=10.7,
    )
    candidate = detector.current()

    assert timeout.action is InterruptionDecisionAction.PROVISIONAL_REJECT
    assert timeout.reason == "no_transcript_timeout"
    assert final.action is InterruptionDecisionAction.REJECT
    assert final.reason == "empty_final"
    assert candidate is not None
    assert candidate.decision_reason == "empty_final"


@pytest.mark.asyncio
async def test_supported_final_can_confirm_after_provisional_timeout() -> None:
    detector = _detector()
    _begin(detector)

    timeout = await detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(1200),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=11.2,
    )
    final = await detector.observe_final(
        StreamTranscript(
            text="Please let me finish",
            audio_duration_ms=1600,
            eou_probability=0.8,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(1600),
        sample_rate=SAMPLE_RATE,
        now=11.6,
    )

    assert timeout.action is InterruptionDecisionAction.PROVISIONAL_REJECT
    assert final.action is InterruptionDecisionAction.CONFIRM
    assert final.reason == "supported_final_transcript"


@pytest.mark.asyncio
async def test_timeout_defers_while_final_classifier_is_in_flight() -> None:
    classifier = _GatedClassifier(result=True)
    detector = EvidenceBasedInterruptDetector(
        policy=TurnPolicy(
            speaking_interrupt_min_duration_ms=420,
            speaking_interrupt_min_words=2,
        ),
        classifier=classifier,
    )
    _begin(detector)
    pending_final = asyncio.create_task(
        detector.observe_final(
            StreamTranscript(
                text="Please let me finish",
                audio_duration_ms=900,
                eou_probability=0.9,
                utterance_id=1,
            ),
            assistant_text="The assistant is still speaking.",
            output_echo=False,
            audio=_voice(900),
            sample_rate=SAMPLE_RATE,
            now=10.9,
        )
    )
    await classifier.started.wait()

    timeout = await detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(900),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=10.9,
    )
    classifier.release.set()
    final = await pending_final

    assert timeout.action is InterruptionDecisionAction.DEFER
    assert timeout.reason == "final_in_flight"
    assert final.action is InterruptionDecisionAction.CONFIRM


@pytest.mark.asyncio
async def test_genuine_final_supersedes_partial_self_echo_for_same_utterance() -> None:
    detector = _detector()
    _begin(detector)

    partial = await detector.observe_partial(
        StreamTranscript(
            text="assistant is still speaking",
            is_partial=True,
            audio_duration_ms=450,
            utterance_id=1,
        ),
        cumulative_transcript="assistant is still speaking",
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        now=10.45,
    )
    final = await detector.observe_final(
        StreamTranscript(
            text="No wait I need to change that.",
            audio_duration_ms=900,
            eou_probability=0.9,
            utterance_id=1,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(900),
        sample_rate=SAMPLE_RATE,
        now=10.9,
    )

    assert partial.action is not InterruptionDecisionAction.CONFIRM
    assert final.action is InterruptionDecisionAction.CONFIRM
    assert detector.current().decision_reason == "supported_final_transcript"


@pytest.mark.asyncio
async def test_unidentified_final_cannot_match_an_active_candidate() -> None:
    detector = _detector()
    _begin(detector, utterance_id=2)

    decision = await detector.observe_final(
        StreamTranscript(
            text="This final has no candidate identity",
            audio_duration_ms=900,
            utterance_id=0,
        ),
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        audio=_voice(900),
        sample_rate=SAMPLE_RATE,
        now=10.9,
    )

    assert decision.action is InterruptionDecisionAction.DEFER
    assert decision.reason == "stale_final"
    assert detector.current().utterance_id == 2


def test_speech_stop_without_transcript_rejects_candidate() -> None:
    detector = _detector()
    _begin(detector)

    decision = detector.mark_speech_stopped(
        utterance_id=1,
        stopped_at=10.4,
        expects_transcript=False,
    )

    assert decision.action is InterruptionDecisionAction.REJECT
    assert decision.reason == "no_transcript"
