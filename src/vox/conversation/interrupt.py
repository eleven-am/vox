"""Pluggable barge-in classifier.

Role: given signals that accumulated while the state machine is in PAUSED, decide
(a) how long the PAUSED window should be before confirming an interrupt, and
(b) whether the interrupt is "real" versus a backchannel / cough.

The default `HeuristicInterruptClassifier` is rules-based: it uses the last
user turn's EOU probability to shrink or grow the confirm window, but always
confirms at window expiry. This matches the industry norm (naive duration) with
a small efficiency win from EOU context.

Future implementations can plug in an acoustic CNN for (b) to distinguish
backchannels from real interrupts. The state machine and session layer don't
need to change — only the classifier does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class InterruptClassifier(Protocol):
    """Decides confirm-window duration and whether an interrupt is real."""

    def confirm_window_ms(
        self,
        base_ms: int,
        last_eou_probability: float | None,
    ) -> int:
        """Return the PAUSED confirm window (ms) for this barge-in attempt.

        base_ms: TurnPolicy.min_interrupt_duration_ms (policy default).
        last_eou_probability: EOU probability of the last committed user turn,
          or None if no turn has occurred yet.
        """
        ...

    async def is_real_interrupt(
        self,
        audio_since_paused: NDArray[np.float32] | None,
        partial_transcript: str | None,
        last_eou_probability: float | None,
        vad_active_duration_ms: int,
    ) -> bool:
        """Called when the confirm timer fires. Return True to confirm the
        interrupt, False to treat as a backchannel and resume TTS.

        partial_transcript: STT text of the audio collected during the PAUSED
          window, or None if STT wasn't available (timed out, errored, or no
          audio). Caller supplies it as-is; the classifier is responsible
          for any normalisation appropriate to its language / domain.
        """
        ...


@dataclass
class HeuristicInterruptClassifier:
    """Default classifier: EOU-modulated window + tail-RMS backchannel filter.

    `confirm_window_ms` modulates the PAUSED wait time using EOU context:
      * bot was clearly at a turn boundary (high last-EOU) → shorter window
        (snappy interrupt on clean turn boundaries)
      * bot was mid-thought (low last-EOU) → longer window (skeptical)

    `is_real_interrupt` runs when the confirm timer fires. Its job is to catch
    backchannels ("mhmm", "uh-huh") that keep Silero VAD "active" because of its
    silence-padding lag even though the user's voice actually decayed quickly.
    The heuristic: check the last `tail_check_ms` of the PAUSED-window audio —
    if RMS is below `backchannel_rms_threshold`, the user already went quiet,
    so treat the trigger as a backchannel and resume TTS.

    When `audio_since_paused` is None (caller can't provide audio), falls back
    to a duration-only rule: reject when the VAD burst is shorter than
    `min_real_interrupt_ms`.
    """

    high_eou_threshold: float = 0.7
    low_eou_threshold: float = 0.3
    high_eou_multiplier: float = 0.5
    low_eou_multiplier: float = 1.5
    min_window_ms: int = 100




    tail_check_ms: int = 100
    backchannel_rms_threshold: float = 0.015
    min_real_interrupt_ms: int = 400






    interrupt_keywords: frozenset[str] = field(default_factory=frozenset)

    def confirm_window_ms(
        self,
        base_ms: int,
        last_eou_probability: float | None,
    ) -> int:
        if last_eou_probability is None:
            return base_ms
        if last_eou_probability >= self.high_eou_threshold:
            scaled = int(base_ms * self.high_eou_multiplier)
        elif last_eou_probability < self.low_eou_threshold:
            scaled = int(base_ms * self.low_eou_multiplier)
        else:
            scaled = base_ms
        return max(scaled, self.min_window_ms)

    async def is_real_interrupt(
        self,
        audio_since_paused: NDArray[np.float32] | None,
        partial_transcript: str | None,
        last_eou_probability: float | None,
        vad_active_duration_ms: int,
    ) -> bool:



        if partial_transcript and self.interrupt_keywords:
            normalised = partial_transcript.strip().lower()
            if normalised:
                for keyword in self.interrupt_keywords:
                    if keyword in normalised:
                        return True

        if audio_since_paused is not None and audio_since_paused.size > 0:


            sample_rate = _assumed_sample_rate(audio_since_paused.size, vad_active_duration_ms)
            tail_samples = min(
                audio_since_paused.size,
                max(1, self.tail_check_ms * sample_rate // 1000),
            )
            tail = audio_since_paused[-tail_samples:]
            rms = float(np.sqrt(np.mean(tail * tail))) if tail.size else 0.0
            if rms < self.backchannel_rms_threshold:
                return False
            return True


        if vad_active_duration_ms < self.min_real_interrupt_ms:
            return False
        return True


def _assumed_sample_rate(n_samples: int, duration_ms: int) -> int:
    """Infer the audio's sample rate from its size + expected duration.

    Falls back to 16 kHz (Vox's TARGET_SAMPLE_RATE) when duration is zero or
    too small to be informative. This lets the classifier work without an
    explicit sample-rate parameter — we compute the tail-window size from what
    the session actually collected.
    """
    if duration_ms <= 0:
        return 16_000
    rate = int(round(n_samples / (duration_ms / 1000.0)))

    if rate < 8_000:
        return 16_000
    if rate > 48_000:
        return 48_000
    return rate
