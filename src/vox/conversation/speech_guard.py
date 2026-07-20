from __future__ import annotations

import time
from collections.abc import Callable

TTS_START_WARMUP = "tts_start_warmup"
TTS_TAIL = "tts_tail"
RESUME_STABILITY = "resume_stability"
RESUME_STABILITY_MS = 150

_EVIDENCE_CONTRIBUTIONS = (TTS_START_WARMUP, RESUME_STABILITY)
_TRANSCRIPT_CONTRIBUTIONS = (TTS_TAIL,)


class AssistantSpeechGuard:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._distrust_until: dict[str, float] = {
            TTS_START_WARMUP: 0.0,
            TTS_TAIL: 0.0,
            RESUME_STABILITY: 0.0,
        }
        self._speech_active = False

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def contribution_until(self, name: str) -> float:
        return self._distrust_until[name]

    def mark_speech_started(self) -> None:
        self._speech_active = True

    def mark_speech_ended(self, tail_ms: int) -> None:
        if self._speech_active:
            self.arm(TTS_TAIL, tail_ms)
        self._speech_active = False

    def arm(self, name: str, duration_ms: int) -> None:
        if name not in self._distrust_until:
            raise ValueError(f"unknown distrust contribution: {name}")
        if duration_ms <= 0:
            self._distrust_until[name] = 0.0
            return
        self._distrust_until[name] = self._clock() + duration_ms / 1000.0

    def suppresses_interrupt_evidence(self, now: float) -> bool:
        return now < max(self._distrust_until[name] for name in _EVIDENCE_CONTRIBUTIONS)

    def interrupt_evidence_distrust_remaining_ms(self, now: float) -> int:
        until = max(self._distrust_until[name] for name in _EVIDENCE_CONTRIBUTIONS)
        return max(0, int((until - now) * 1000))

    def suppresses_transcript_trust(self, now: float) -> bool:
        if self._speech_active:
            return True
        return now < max(self._distrust_until[name] for name in _TRANSCRIPT_CONTRIBUTIONS)
