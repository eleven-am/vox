"""Pluggable acoustic evidence for WebRTC interruption candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vox.streaming.types import StreamTranscript

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def looks_like_self_echo(
    transcript: str | None,
    assistant_text: str | None,
    *,
    min_words: int = 3,
    min_overlap: float = 0.7,
) -> bool:
    """Return True when mic transcript appears to be leaked assistant playback."""
    if not transcript or not assistant_text:
        return False

    heard = _WORD_RE.findall(transcript.casefold())
    spoken = _WORD_RE.findall(assistant_text.casefold())
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
    return len(_WORD_RE.findall(text))


def transcript_duration_ms(transcript: StreamTranscript | None) -> int:
    if transcript is None:
        return 0
    if transcript.audio_duration_ms > 0:
        return int(transcript.audio_duration_ms)
    if transcript.end_ms > transcript.start_ms:
        return int(transcript.end_ms - transcript.start_ms)
    return 0


@dataclass(frozen=True)
class AcousticInterruptEvidence:
    """Cheap speech-likeness features computed from an interruption audio tail."""

    duration_ms: int
    rms: float
    tail_rms: float
    active_frame_ratio: float
    voiced_frame_ratio: float
    spectral_flatness: float
    crest_factor: float

    def is_speech_like(
        self,
        *,
        min_duration_ms: int,
        min_rms: float,
        min_tail_rms: float,
        min_active_frame_ratio: float,
        min_voiced_frame_ratio: float,
        max_spectral_flatness: float,
        max_crest_factor: float,
    ) -> bool:
        return (
            self.duration_ms >= min_duration_ms
            and self.rms >= min_rms
            and self.tail_rms >= min_tail_rms
            and self.active_frame_ratio >= min_active_frame_ratio
            and self.voiced_frame_ratio >= min_voiced_frame_ratio
            and self.spectral_flatness <= max_spectral_flatness
            and self.crest_factor <= max_crest_factor
        )


def analyze_interrupt_audio(
    audio: NDArray[np.float32] | None,
    sample_rate: int,
    *,
    tail_check_ms: int = 80,
    analysis_window_ms: int = 1200,
) -> AcousticInterruptEvidence:
    if audio is None or audio.size == 0 or sample_rate <= 0:
        return AcousticInterruptEvidence(0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    full_signal = np.nan_to_num(np.asarray(audio, dtype=np.float32), copy=False)
    duration_ms = int(full_signal.size * 1000 / sample_rate)
    analysis_samples = max(1, analysis_window_ms * sample_rate // 1000)
    signal = full_signal[-analysis_samples:]
    rms = _rms(signal)
    tail_samples = min(signal.size, max(1, tail_check_ms * sample_rate // 1000))
    tail_rms = _rms(signal[-tail_samples:])
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    crest_factor = peak / max(rms, 1e-8)

    frame_size = max(32, int(sample_rate * 0.04))
    hop_size = max(16, frame_size // 2)
    frames = _audio_frames(signal, frame_size, hop_size)
    if not frames:
        frames = [signal]

    frame_rms = np.asarray([_rms(frame) for frame in frames], dtype=np.float32)
    active_threshold = max(0.002, float(np.max(frame_rms)) * 0.18)
    active = [frame for frame, value in zip(frames, frame_rms, strict=True) if value >= active_threshold]
    active_frame_ratio = len(active) / len(frames)
    if not active:
        return AcousticInterruptEvidence(
            duration_ms,
            rms,
            tail_rms,
            active_frame_ratio,
            0.0,
            1.0,
            crest_factor,
        )

    periodicities = [_frame_periodicity(frame, sample_rate) for frame in active]
    voiced_frame_ratio = sum(value >= 0.32 for value in periodicities) / len(periodicities)
    flatness = float(np.median([_spectral_flatness(frame) for frame in active]))
    return AcousticInterruptEvidence(
        duration_ms=duration_ms,
        rms=rms,
        tail_rms=tail_rms,
        active_frame_ratio=active_frame_ratio,
        voiced_frame_ratio=voiced_frame_ratio,
        spectral_flatness=flatness,
        crest_factor=crest_factor,
    )


def _audio_frames(
    audio: NDArray[np.float32],
    frame_size: int,
    hop_size: int,
) -> list[NDArray[np.float32]]:
    if audio.size <= frame_size:
        return [audio] if audio.size else []
    return [
        audio[start:start + frame_size]
        for start in range(0, audio.size - frame_size + 1, hop_size)
    ]


def _rms(audio: NDArray[np.float32]) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))


def _frame_periodicity(frame: NDArray[np.float32], sample_rate: int) -> float:
    centered = np.asarray(frame, dtype=np.float32) - float(np.mean(frame))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-10:
        return 0.0

    min_lag = max(1, sample_rate // 400)
    max_lag = min(centered.size - 2, sample_rate // 70)
    if max_lag <= min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > 1e-10:
            best = max(best, float(np.dot(left, right)) / denominator)
    return max(0.0, min(1.0, best))


def _spectral_flatness(frame: NDArray[np.float32]) -> float:
    if frame.size < 2:
        return 1.0
    windowed = frame * np.hanning(frame.size).astype(np.float32)
    power = np.square(np.abs(np.fft.rfft(windowed)), dtype=np.float64) + 1e-12
    geometric_mean = float(np.exp(np.mean(np.log(power))))
    arithmetic_mean = float(np.mean(power))
    return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 1.0


@runtime_checkable
class InterruptClassifier(Protocol):
    """Pluggable acoustic boundary used by the interruption detector."""

    def confirm_window_ms(
        self,
        base_ms: int,
        last_eou_probability: float | None,
    ) -> int: ...

    def wants_short_circuit(self) -> bool: ...

    def should_short_circuit(self, partial_transcript: str | None) -> bool: ...

    async def is_real_interrupt(
        self,
        audio_since_paused: NDArray[np.float32] | None,
        partial_transcript: str | None,
        last_eou_probability: float | None,
        vad_active_duration_ms: int,
        sample_rate: int,
    ) -> bool: ...


@dataclass
class HeuristicInterruptClassifier:
    """Content-independent acoustic classifier for the default detector.

    Explicit keyword sets remain supported for custom deployments, but Vox no
    longer derives semantic interruption words from the session language.
    """

    high_eou_threshold: float = 0.7
    low_eou_threshold: float = 0.3
    high_eou_multiplier: float = 0.35
    low_eou_multiplier: float = 1.25
    min_window_ms: int = 75
    tail_check_ms: int = 80
    min_real_interrupt_ms: int = 180
    min_interrupt_words: int = 0
    min_rms: float = 0.0025
    min_tail_rms: float = 0.0025
    min_active_frame_ratio: float = 0.55
    min_voiced_frame_ratio: float = 0.20
    max_spectral_flatness: float = 0.70
    max_crest_factor: float = 12.0
    interrupt_keywords: frozenset[str] = field(default_factory=frozenset)
    language: str | None = None

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
        normalised = partial_transcript.casefold().strip()
        return bool(normalised) and any(
            keyword.casefold() in normalised for keyword in self.interrupt_keywords
        )

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

        if (
            self.min_interrupt_words > 0
            and partial_transcript is not None
            and transcript_word_count(partial_transcript) < self.min_interrupt_words
        ):
            return False

        evidence = analyze_interrupt_audio(
            audio_since_paused,
            sample_rate,
            tail_check_ms=self.tail_check_ms,
        )
        if evidence.duration_ms == 0:
            return False
        supported_eou = (
            last_eou_probability is not None
            and last_eou_probability >= 0.5
        )
        return evidence.is_speech_like(
            min_duration_ms=max(self.min_real_interrupt_ms, min(vad_active_duration_ms, 250)),
            min_rms=self.min_rms,
            min_tail_rms=0.0 if supported_eou else self.min_tail_rms,
            min_active_frame_ratio=self.min_active_frame_ratio,
            min_voiced_frame_ratio=self.min_voiced_frame_ratio,
            max_spectral_flatness=self.max_spectral_flatness,
            max_crest_factor=self.max_crest_factor,
        )
