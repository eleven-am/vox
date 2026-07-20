from __future__ import annotations

import pytest

from vox.conversation.speech_guard import (
    RESUME_STABILITY,
    RESUME_STABILITY_MS,
    TTS_START_WARMUP,
    TTS_TAIL,
    AssistantSpeechGuard,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_start_warmup_contribution_suppresses_interrupt_evidence() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    assert not guard.suppresses_interrupt_evidence(clock.now)
    assert guard.contribution_until(TTS_START_WARMUP) == 0.0

    guard.arm(TTS_START_WARMUP, 250)

    assert guard.suppresses_interrupt_evidence(clock.now)
    assert guard.contribution_until(TTS_START_WARMUP) == 10.25

    clock.advance(0.251)

    assert not guard.suppresses_interrupt_evidence(clock.now)


def test_resume_stability_contribution_suppresses_interrupt_evidence() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.arm(RESUME_STABILITY, 100)

    assert guard.suppresses_interrupt_evidence(clock.now)

    clock.advance(0.101)

    assert not guard.suppresses_interrupt_evidence(clock.now)


def test_contributions_combine_by_max() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.arm(TTS_START_WARMUP, 100)
    guard.arm(RESUME_STABILITY, 300)

    clock.advance(0.2)

    assert guard.suppresses_interrupt_evidence(clock.now)

    clock.advance(0.2)

    assert not guard.suppresses_interrupt_evidence(clock.now)


def test_zero_duration_clears_a_contribution() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.arm(TTS_START_WARMUP, 250)
    guard.arm(TTS_START_WARMUP, 0)

    assert guard.contribution_until(TTS_START_WARMUP) == 0.0
    assert not guard.suppresses_interrupt_evidence(clock.now)


def test_unknown_contribution_is_rejected() -> None:
    guard = AssistantSpeechGuard(clock=FakeClock())

    with pytest.raises(ValueError, match="unknown distrust contribution"):
        guard.arm("mystery_window", 100)


def test_transcript_trust_tracks_active_speech_and_tail() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.mark_speech_started()

    assert guard.speech_active
    assert guard.suppresses_transcript_trust(clock.now)

    guard.mark_speech_ended(500)

    assert not guard.speech_active
    assert guard.suppresses_transcript_trust(clock.now)

    clock.advance(0.501)

    assert not guard.suppresses_transcript_trust(clock.now)


def test_tail_is_only_armed_when_speech_was_active() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.mark_speech_ended(500)

    assert guard.contribution_until(TTS_TAIL) == 0.0
    assert not guard.suppresses_transcript_trust(clock.now)


def test_tail_does_not_suppress_interrupt_evidence() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.mark_speech_started()
    guard.mark_speech_ended(1500)

    assert guard.suppresses_transcript_trust(clock.now)
    assert not guard.suppresses_interrupt_evidence(clock.now)


def test_resume_stability_constant_is_production_magnitude() -> None:
    assert RESUME_STABILITY_MS == 150


def test_evidence_distrust_remaining_tracks_longest_window() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    assert guard.interrupt_evidence_distrust_remaining_ms(clock.now) == 0

    guard.arm(TTS_START_WARMUP, 100)
    guard.arm(RESUME_STABILITY, 300)

    assert guard.interrupt_evidence_distrust_remaining_ms(clock.now) == 300

    clock.advance(0.2)

    assert guard.interrupt_evidence_distrust_remaining_ms(clock.now) == 100

    clock.advance(0.2)

    assert guard.interrupt_evidence_distrust_remaining_ms(clock.now) == 0


def test_evidence_windows_do_not_suppress_transcript_trust() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.arm(TTS_START_WARMUP, 500)
    guard.arm(RESUME_STABILITY, 500)

    assert guard.suppresses_interrupt_evidence(clock.now)
    assert not guard.suppresses_transcript_trust(clock.now)
