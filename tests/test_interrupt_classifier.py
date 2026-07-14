from __future__ import annotations

import numpy as np
import pytest

from vox.conversation import HeuristicInterruptClassifier, InterruptClassifier
from vox.conversation.interrupt import (
    analyze_interrupt_audio,
    looks_like_self_echo,
    transcript_duration_ms,
    transcript_word_count,
)
from vox.streaming.types import StreamTranscript

SAMPLE_RATE = 16_000


def _voice(duration_ms: int, *, amplitude: float = 0.1, frequency: float = 220) -> np.ndarray:
    t = np.arange(duration_ms * SAMPLE_RATE // 1000) / SAMPLE_RATE
    fundamental = np.sin(2 * np.pi * frequency * t)
    harmonic = 0.35 * np.sin(2 * np.pi * frequency * 2 * t)
    envelope = 0.65 + 0.35 * np.sin(2 * np.pi * 4 * t) ** 2
    return (amplitude * envelope * (fundamental + harmonic)).astype(np.float32)


def _noise(duration_ms: int, *, amplitude: float = 0.1, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        0,
        amplitude,
        duration_ms * SAMPLE_RATE // 1000,
    ).astype(np.float32)


class TestConfirmWindow:
    def test_eou_context_only_modulates_the_existing_window(self) -> None:
        classifier = HeuristicInterruptClassifier()

        assert classifier.confirm_window_ms(250, 0.9) == 87
        assert classifier.confirm_window_ms(250, 0.1) == 312
        assert classifier.confirm_window_ms(250, None) == 250


class TestContentIndependentDefaults:
    def test_language_does_not_install_semantic_shortcuts(self) -> None:
        classifier = HeuristicInterruptClassifier(language="en")

        assert classifier.interrupt_keywords == frozenset()
        assert not classifier.wants_short_circuit()
        assert not classifier.should_short_circuit("stop")

    def test_explicit_operator_keywords_remain_opt_in(self) -> None:
        classifier = HeuristicInterruptClassifier(interrupt_keywords=frozenset({"halt"}))

        assert classifier.wants_short_circuit()
        assert classifier.should_short_circuit("please HALT now")


class TestAcousticEvidence:
    @pytest.mark.asyncio
    async def test_sustained_voice_like_audio_is_accepted(self) -> None:
        classifier = HeuristicInterruptClassifier()
        audio = _voice(600)

        assert await classifier.is_real_interrupt(audio, None, None, 600, SAMPLE_RATE)

    @pytest.mark.asyncio
    async def test_sustained_handling_noise_is_not_accepted_from_energy_alone(self) -> None:
        classifier = HeuristicInterruptClassifier()
        audio = _noise(600, amplitude=0.15, seed=4)

        assert not await classifier.is_real_interrupt(audio, None, None, 600, SAMPLE_RATE)

    @pytest.mark.asyncio
    async def test_impulse_is_rejected(self) -> None:
        classifier = HeuristicInterruptClassifier()
        audio = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        audio[100:180] = 0.8

        assert not await classifier.is_real_interrupt(audio, None, None, 500, SAMPLE_RATE)

    @pytest.mark.asyncio
    async def test_backchannel_with_quiet_tail_is_rejected(self) -> None:
        classifier = HeuristicInterruptClassifier()
        audio = np.concatenate([_voice(180, amplitude=0.08), np.zeros(5120, dtype=np.float32)])

        assert not await classifier.is_real_interrupt(audio, "mhmm", None, 500, SAMPLE_RATE)

    @pytest.mark.asyncio
    async def test_high_eou_single_word_can_tolerate_vad_trailing_silence(self) -> None:
        classifier = HeuristicInterruptClassifier()
        audio = np.concatenate([_voice(400, amplitude=0.08), np.zeros(1600, dtype=np.float32)])

        assert await classifier.is_real_interrupt(audio, "hold", 0.9, 500, SAMPLE_RATE)

    @pytest.mark.asyncio
    async def test_missing_audio_is_not_promoted_by_duration(self) -> None:
        classifier = HeuristicInterruptClassifier()

        assert not await classifier.is_real_interrupt(None, None, None, 2000, SAMPLE_RATE)

    def test_feature_report_distinguishes_voice_from_broadband_noise(self) -> None:
        voice = analyze_interrupt_audio(_voice(600), SAMPLE_RATE)
        noise = analyze_interrupt_audio(_noise(600, seed=9), SAMPLE_RATE)

        assert voice.voiced_frame_ratio > noise.voiced_frame_ratio
        assert voice.spectral_flatness < noise.spectral_flatness

    def test_long_utterance_analysis_is_bounded_to_recent_audio(self) -> None:
        recent_voice = _voice(1200)
        long_audio = np.concatenate([_noise(13_800, seed=12), recent_voice])

        recent = analyze_interrupt_audio(recent_voice, SAMPLE_RATE)
        long = analyze_interrupt_audio(long_audio, SAMPLE_RATE)

        assert long.duration_ms == 15_000
        assert long.rms == pytest.approx(recent.rms)
        assert long.active_frame_ratio == pytest.approx(recent.active_frame_ratio)
        assert long.voiced_frame_ratio == pytest.approx(recent.voiced_frame_ratio)
        assert long.spectral_flatness == pytest.approx(recent.spectral_flatness)


class TestTranscriptUtilities:
    def test_unicode_self_echo(self) -> None:
        assert looks_like_self_echo(
            "الموعد غدا عند الظهر",
            "حسنا، الموعد غدا عند الظهر بالتأكيد.",
        )

    def test_unrelated_text_is_not_echo(self) -> None:
        assert not looks_like_self_echo(
            "I need to change the appointment",
            "The appointment is tomorrow at noon",
        )

    def test_word_count_and_duration(self) -> None:
        transcript = StreamTranscript(start_ms=100, end_ms=900, audio_duration_ms=450)

        assert transcript_word_count("  hello,   there ") == 2
        assert transcript_duration_ms(transcript) == 450


def test_default_classifier_implements_public_protocol() -> None:
    assert isinstance(HeuristicInterruptClassifier(), InterruptClassifier)
