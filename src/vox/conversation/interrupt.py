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

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vox.conversation.types import TurnPolicy
from vox.streaming.types import StreamTranscript

DEFAULT_INTERRUPT_KEYWORDS_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset({"stop", "wait", "hold on", "pause", "cancel", "nevermind", "never mind"}),
    "fr": frozenset({"arrête", "arretez", "arrêtez", "attends", "attendez", "pause", "annule", "annulez"}),
    "es": frozenset({"para", "pare", "espera", "alto", "pausa", "cancela", "cancele"}),
    "de": frozenset({"stopp", "stop", "warte", "warten", "halt", "pause", "abbrechen"}),
    "it": frozenset({"ferma", "fermati", "fermo", "aspetta", "pausa", "annulla"}),
    "pt": frozenset({"para", "pare", "espera", "pausa", "cancela", "cancele"}),
    "nl": frozenset({"stop", "wacht", "pauze", "annuleer"}),
    "ar": frozenset({"توقف", "انتظر", "إلغاء"}),
    "hi": frozenset({"रुको", "रुकिए", "ठहरो", "रद्द"}),
}

_WORD_RE = re.compile(r"[a-z0-9']+")


def looks_like_self_echo(
    transcript: str | None,
    assistant_text: str | None,
    *,
    min_words: int = 3,
    min_overlap: float = 0.7,
) -> bool:
    """Return True when mic transcript appears to be leaked assistant playback.

    This is not acoustic echo cancellation. It is a server-side guard for the
    loudspeaker case where AEC misses and STT transcribes Vox's own TTS.
    """
    if not transcript or not assistant_text:
        return False

    heard = _WORD_RE.findall(transcript.lower())
    spoken = _WORD_RE.findall(assistant_text.lower())
    if len(heard) < min_words or len(spoken) < min_words:
        return False

    spoken_text = " ".join(spoken)
    heard_text = " ".join(heard)
    if heard_text in spoken_text:
        return True

    spoken_set = set(spoken)
    overlap = sum(1 for word in heard if word in spoken_set) / len(heard)
    return overlap >= min_overlap


def transcript_word_count(text: str | None) -> int:
    if not text:
        return 0
    return len([word for word in text.strip().split() if word])


def transcript_duration_ms(transcript: StreamTranscript | None) -> int:
    if transcript is None:
        return 0
    if transcript.audio_duration_ms > 0:
        return int(transcript.audio_duration_ms)
    if transcript.end_ms > transcript.start_ms:
        return int(transcript.end_ms - transcript.start_ms)
    return 0


@dataclass(frozen=True)
class PartialInterruptEvidence:
    """Policy for deciding whether partial STT is enough to confirm barge-in."""

    min_interrupt_duration_ms: int
    speaking_interrupt_min_words: int
    self_echo_min_words: int
    self_echo_min_overlap: float

    @classmethod
    def from_turn_policy(cls, policy: TurnPolicy) -> PartialInterruptEvidence:
        return cls(
            min_interrupt_duration_ms=policy.min_interrupt_duration_ms,
            speaking_interrupt_min_words=policy.speaking_interrupt_min_words,
            self_echo_min_words=policy.self_echo_min_words,
            self_echo_min_overlap=policy.self_echo_min_overlap,
        )

    def is_strong(
        self,
        transcript: StreamTranscript | None,
        *,
        assistant_text: str | None,
    ) -> bool:
        if transcript is None:
            return False
        text = transcript.text.strip()
        if not text:
            return False
        if assistant_text and looks_like_self_echo(
            text,
            assistant_text,
            min_words=self.self_echo_min_words,
            min_overlap=self.self_echo_min_overlap,
        ):
            return False
        if transcript_word_count(text) < self.speaking_interrupt_min_words:
            return False
        duration_ms = transcript_duration_ms(transcript)
        return not (duration_ms > 0 and duration_ms < self.min_interrupt_duration_ms)


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

    def wants_short_circuit(self) -> bool:
        ...

    def should_short_circuit(self, partial_transcript: str | None) -> bool:
        ...

    async def is_real_interrupt(
        self,
        audio_since_paused: NDArray[np.float32] | None,
        partial_transcript: str | None,
        last_eou_probability: float | None,
        vad_active_duration_ms: int,
        sample_rate: int,
    ) -> bool:
        """Called when the confirm timer fires. Return True to confirm the
        interrupt, False to treat as a backchannel and resume TTS.

        partial_transcript: STT text of the audio collected during the PAUSED
          window, or None if STT wasn't available (timed out, errored, or no
          audio). Caller supplies it as-is; the classifier is responsible
          for any normalisation appropriate to its language / domain.
        sample_rate: rate of audio_since_paused in Hz; required so the
          classifier can size its tail window in samples without inferring.
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
    high_eou_multiplier: float = 0.35
    low_eou_multiplier: float = 1.25
    min_window_ms: int = 75




    tail_check_ms: int = 80
    backchannel_rms_threshold: float = 0.01
    min_real_interrupt_ms: int = 180
    min_interrupt_words: int = 0




    interrupt_keywords: frozenset[str] = field(default_factory=frozenset)
    language: str | None = None

    def __post_init__(self) -> None:
        if not self.interrupt_keywords and self.language:
            defaults = DEFAULT_INTERRUPT_KEYWORDS_BY_LANG.get(self.language.lower())
            if defaults:
                self.interrupt_keywords = defaults

    def wants_short_circuit(self) -> bool:
        return bool(self.interrupt_keywords)

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

    def should_short_circuit(self, partial_transcript: str | None) -> bool:
        if not partial_transcript or not self.interrupt_keywords:
            return False
        normalised = partial_transcript.strip().lower()
        if not normalised:
            return False
        return any(keyword in normalised for keyword in self.interrupt_keywords)

    async def is_real_interrupt(
        self,
        audio_since_paused: NDArray[np.float32] | None,
        partial_transcript: str | None,
        last_eou_probability: float | None,
        vad_active_duration_ms: int,
        sample_rate: int,
    ) -> bool:



        if self.should_short_circuit(partial_transcript):
            return True

        if self.min_interrupt_words > 0 and partial_transcript is not None:
            words = [word for word in partial_transcript.strip().split() if word]
            if len(words) < self.min_interrupt_words:
                return False

        if audio_since_paused is not None and audio_since_paused.size > 0:


            tail_samples = min(
                audio_since_paused.size,
                max(1, self.tail_check_ms * sample_rate // 1000),
            )
            tail = audio_since_paused[-tail_samples:]
            rms = float(np.sqrt(np.mean(tail * tail))) if tail.size else 0.0
            return not rms < self.backchannel_rms_threshold


        return not vad_active_duration_ms < self.min_real_interrupt_ms
