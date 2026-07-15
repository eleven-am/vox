from __future__ import annotations

import numpy as np
import pytest

from vox.conversation.interrupt import HeuristicInterruptClassifier
from vox.conversation.interruption_detector import (
    EvidenceBasedInterruptDetector,
    InterruptionDecisionAction,
)
from vox.conversation.types import TurnPolicy
from vox.streaming.types import StreamTranscript

SAMPLE_RATE = 16_000


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
        response_id="resp_1",
        started_at=10.0,
        speech_started_ms=1000,
        assistant_text="The assistant is still speaking.",
    )


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
async def test_timeout_uses_acoustic_speech_likeness_not_energy_alone() -> None:
    voice_detector = _detector()
    _begin(voice_detector)
    voice = await voice_detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=False,
        aec_warmup_remaining_ms=0,
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
        aec_warmup_remaining_ms=0,
        audio=_noise(600, seed=4),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=10.6,
    )

    assert voice.action is InterruptionDecisionAction.CONFIRM
    assert noise.action is InterruptionDecisionAction.REJECT


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
    assert self_echo.action is InterruptionDecisionAction.REJECT

    output_detector = _detector()
    _begin(output_detector)
    output_echo = await output_detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=True,
        aec_warmup_remaining_ms=0,
        audio=_voice(600),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=10.6,
    )
    assert output_echo.action is InterruptionDecisionAction.DEFER
    assert output_echo.reason == "output_echo_waiting_for_transcript"
    assert output_echo.retry_after_ms > 0

    timed_out = await output_detector.evaluate_timeout(
        assistant_text="The assistant is still speaking.",
        output_echo=True,
        aec_warmup_remaining_ms=0,
        audio=_voice(2100),
        sample_rate=SAMPLE_RATE,
        last_eou_probability=None,
        now=12.1,
    )
    assert timed_out.action is InterruptionDecisionAction.REJECT
    assert timed_out.reason == "output_echo_timeout"


@pytest.mark.asyncio
async def test_stale_final_cannot_confirm_a_new_candidate() -> None:
    detector = _detector()
    _begin(detector, utterance_id=1)
    detector.begin(
        utterance_id=2,
        response_id="resp_1",
        started_at=11.0,
        speech_started_ms=2000,
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
        speech_stopped_ms=1400,
        expects_transcript=False,
    )

    assert decision.action is InterruptionDecisionAction.REJECT
    assert decision.reason == "no_transcript"
