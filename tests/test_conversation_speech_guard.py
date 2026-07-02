from __future__ import annotations

from vox.conversation.speech_guard import AssistantSpeechGuard


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_aec_warmup_uses_injected_clock() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    assert not guard.aec_warmup_active()
    assert guard.aec_warmup_until == 0.0

    guard.arm_aec_warmup(250)

    assert guard.aec_warmup_active()
    assert guard.aec_warmup_until == 10.25

    clock.advance(0.251)

    assert not guard.aec_warmup_active()


def test_zero_aec_warmup_does_not_arm() -> None:
    guard = AssistantSpeechGuard(clock=FakeClock())

    guard.arm_aec_warmup(0)

    assert guard.aec_warmup_until == 0.0
    assert not guard.aec_warmup_active()


def test_self_echo_window_tracks_active_speech_and_end_cooldown() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.mark_speech_started()

    assert guard.speech_active
    assert guard.in_self_echo_window(0)

    guard.mark_speech_ended()

    assert not guard.speech_active
    assert guard.in_self_echo_window(500)

    clock.advance(0.501)

    assert not guard.in_self_echo_window(500)


def test_flutter_cooldown_uses_injected_clock() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.start_flutter_cooldown(100)

    assert guard.flutter_cooldown_active()

    clock.advance(0.101)

    assert not guard.flutter_cooldown_active()


def test_zero_flutter_cooldown_clears_active_window() -> None:
    clock = FakeClock()
    guard = AssistantSpeechGuard(clock=clock)

    guard.start_flutter_cooldown(100)
    assert guard.flutter_cooldown_active()

    guard.start_flutter_cooldown(0)

    assert not guard.flutter_cooldown_active()
