"""Assistant speech timing guards for conversation turn handling."""

from __future__ import annotations

import time
from collections.abc import Callable


class AssistantSpeechGuard:
    """Tracks assistant playout timing used to reject self-echo as user speech."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._flutter_cooldown_until = 0.0
        self._aec_warmup_until = 0.0
        self._speech_active = False
        self._speech_end_monotonic = 0.0

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def aec_warmup_until(self) -> float:
        return self._aec_warmup_until

    def mark_speech_started(self) -> None:
        self._speech_active = True

    def mark_speech_ended(self) -> None:
        if self._speech_active:
            self._speech_end_monotonic = self._clock()
        self._speech_active = False

    def arm_aec_warmup(self, duration_ms: int) -> None:
        if duration_ms <= 0:
            return
        self._aec_warmup_until = self._clock() + duration_ms / 1000.0

    def aec_warmup_active(self) -> bool:
        if self._aec_warmup_until <= 0.0:
            return False
        return self._clock() < self._aec_warmup_until

    def start_flutter_cooldown(self, duration_ms: int) -> None:
        if duration_ms <= 0:
            self._flutter_cooldown_until = 0.0
            return
        self._flutter_cooldown_until = self._clock() + duration_ms / 1000.0

    def flutter_cooldown_active(self) -> bool:
        if self._flutter_cooldown_until <= 0.0:
            return False
        return self._clock() < self._flutter_cooldown_until

    def in_self_echo_window(self, cooldown_ms: int) -> bool:
        if self._speech_active:
            return True
        cooldown_s = cooldown_ms / 1000.0
        if cooldown_s <= 0.0:
            return False
        return self._clock() < self._speech_end_monotonic + cooldown_s
