from __future__ import annotations

import numpy as np
import pytest

from vox.conversation import HeuristicInterruptClassifier, InterruptClassifier


class TestConfirmWindowMs:
    def test_default_tuning_is_aggressive(self):
        c = HeuristicInterruptClassifier()
        assert c.confirm_window_ms(base_ms=250, last_eou_probability=0.9) == 87
        assert c.confirm_window_ms(base_ms=250, last_eou_probability=0.1) == 312

    def test_no_eou_returns_base(self):
        c = HeuristicInterruptClassifier()
        assert c.confirm_window_ms(base_ms=300, last_eou_probability=None) == 300

    def test_high_eou_shrinks_window(self):
        c = HeuristicInterruptClassifier(high_eou_threshold=0.7, high_eou_multiplier=0.5)
        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.9) == 150

    def test_low_eou_grows_window(self):
        c = HeuristicInterruptClassifier(low_eou_threshold=0.3, low_eou_multiplier=1.5)
        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.1) == 450

    def test_mid_range_eou_returns_base(self):
        c = HeuristicInterruptClassifier()
        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.5) == 300

    def test_min_window_floor(self):
        c = HeuristicInterruptClassifier(
            high_eou_multiplier=0.01,
            min_window_ms=100,
        )
        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.9) == 100

    def test_custom_thresholds_applied(self):
        c = HeuristicInterruptClassifier(
            high_eou_threshold=0.9,
            high_eou_multiplier=0.4,
        )

        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.8) == 300
        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.95) == 120

    def test_exactly_at_high_threshold(self):
        c = HeuristicInterruptClassifier(high_eou_threshold=0.7)

        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.7) < 300

    def test_exactly_at_low_threshold(self):
        c = HeuristicInterruptClassifier(low_eou_threshold=0.3)

        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.3) == 300
        assert c.confirm_window_ms(base_ms=300, last_eou_probability=0.29) > 300


class TestIsRealInterruptWithoutAudio:
    """No audio supplied → duration-only fallback (vad_active_ms vs threshold)."""

    @pytest.mark.asyncio
    async def test_default_duration_threshold_is_shorter(self):
        c = HeuristicInterruptClassifier()
        assert await c.is_real_interrupt(None, None, None, 179) is False
        assert await c.is_real_interrupt(None, None, None, 180) is True

    @pytest.mark.asyncio
    async def test_short_duration_rejected(self):
        c = HeuristicInterruptClassifier(min_real_interrupt_ms=400)
        assert await c.is_real_interrupt(None, None, None, 300) is False

    @pytest.mark.asyncio
    async def test_long_duration_accepted(self):
        c = HeuristicInterruptClassifier(min_real_interrupt_ms=400)
        assert await c.is_real_interrupt(None, None, None, 500) is True

    @pytest.mark.asyncio
    async def test_threshold_boundary(self):
        c = HeuristicInterruptClassifier(min_real_interrupt_ms=400)
        assert await c.is_real_interrupt(None, None, None, 399) is False
        assert await c.is_real_interrupt(None, None, None, 400) is True


class TestIsRealInterruptWithAudio:
    """Audio supplied → RMS of the tail decides; quiet tail = backchannel."""

    @pytest.mark.asyncio
    async def test_quiet_tail_rejected_as_backchannel(self):
        """Simulates 'mhmm': short voice burst, then silence. RMS near the end
        is low → classifier says backchannel, resume TTS."""
        c = HeuristicInterruptClassifier(
            tail_check_ms=200, backchannel_rms_threshold=0.015,
        )
        sr = 16_000
        voice_samples = int(0.25 * sr)
        silence_samples = int(0.35 * sr)
        audio = np.concatenate([
            (0.1 * np.sin(2 * np.pi * 220 * np.arange(voice_samples) / sr)).astype(np.float32),
            np.zeros(silence_samples, dtype=np.float32),
        ])
        duration_ms = int(1000 * audio.size / sr)
        assert await c.is_real_interrupt(audio, None, None, duration_ms) is False

    @pytest.mark.asyncio
    async def test_sustained_voice_accepted(self):
        """Simulates real interrupt: voice continues through the whole window."""
        c = HeuristicInterruptClassifier(
            tail_check_ms=200, backchannel_rms_threshold=0.015,
        )
        sr = 16_000
        samples = int(0.6 * sr)
        audio = (
            0.15 * np.sin(2 * np.pi * 220 * np.arange(samples) / sr)
        ).astype(np.float32)
        assert await c.is_real_interrupt(audio, None, None, 600) is True

    @pytest.mark.asyncio
    async def test_tail_shorter_than_tail_check_still_works(self):
        """If the buffer is shorter than tail_check_ms, still evaluates against what's there."""
        c = HeuristicInterruptClassifier(
            tail_check_ms=500, backchannel_rms_threshold=0.015,
        )
        sr = 16_000
        samples = int(0.1 * sr)
        audio = np.zeros(samples, dtype=np.float32)
        assert await c.is_real_interrupt(audio, None, None, 100) is False

    @pytest.mark.asyncio
    async def test_custom_threshold_respected(self):
        """Raising the threshold makes the classifier stricter (more rejects)."""
        c_strict = HeuristicInterruptClassifier(
            tail_check_ms=200, backchannel_rms_threshold=0.2,
        )
        sr = 16_000
        samples = int(0.4 * sr)
        audio = (
            0.1 * np.sin(2 * np.pi * 220 * np.arange(samples) / sr)
        ).astype(np.float32)

        assert await c_strict.is_real_interrupt(audio, None, None, 400) is False


class TestKeywordOverride:
    """`interrupt_keywords` — opt-in, language-neutral (empty by default)."""

    @pytest.mark.asyncio
    async def test_default_keyword_set_is_empty(self):
        """Out of the box the classifier does NOT interpret any words as
        interrupt intent — operators must supply their own set.
        """
        c = HeuristicInterruptClassifier()
        assert c.interrupt_keywords == frozenset()

    @pytest.mark.asyncio
    async def test_keyword_hit_overrides_quiet_tail(self):
        """Even with a silent-tail audio clip (would normally reject as
        backchannel), a keyword hit in the transcript confirms the interrupt.
        Proves the transcript path short-circuits the RMS check.
        """
        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"stop"}),
        )
        sr = 16_000
        audio = np.concatenate([
            (0.1 * np.sin(2 * np.pi * 220 * np.arange(int(0.1 * sr)) / sr)).astype(np.float32),
            np.zeros(int(0.4 * sr), dtype=np.float32),
        ])
        assert await c.is_real_interrupt(audio, "please stop talking", None, 500) is True

    @pytest.mark.asyncio
    async def test_empty_keyword_set_does_not_match(self):
        """With the default empty set, a transcript that happens to contain
        "stop" should still go through the audio-based check (rejected here
        because the tail is silence).
        """
        c = HeuristicInterruptClassifier()
        sr = 16_000
        audio = np.concatenate([
            (0.1 * np.sin(2 * np.pi * 220 * np.arange(int(0.1 * sr)) / sr)).astype(np.float32),
            np.zeros(int(0.4 * sr), dtype=np.float32),
        ])
        assert await c.is_real_interrupt(audio, "please stop talking", None, 500) is False

    @pytest.mark.asyncio
    async def test_keyword_substring_match(self):
        """Keyword check is substring, not token — 'halt' hits 'haltet euch'."""
        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"halt"}),
        )
        assert await c.is_real_interrupt(None, "haltet euch zurueck", None, 100) is True

    @pytest.mark.asyncio
    async def test_keyword_case_insensitive(self):
        """Keyword compare is lowercase — 'STOP' and 'Stop' both match."""
        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"stop"}),
        )
        assert await c.is_real_interrupt(None, "STOP", None, 100) is True
        assert await c.is_real_interrupt(None, "Stop please", None, 100) is True

    @pytest.mark.asyncio
    async def test_keyword_language_neutral_non_latin(self):
        """Operators can supply keywords in any script — CJK, Devanagari, etc.
        Proves no hardcoded language assumptions in the classifier.
        """

        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"やめて", "停"}),
        )
        assert await c.is_real_interrupt(None, "やめて", None, 100) is True
        assert await c.is_real_interrupt(None, "停下来", None, 100) is True


        assert await c.is_real_interrupt(None, "こんにちは", None, 100) is False

    @pytest.mark.asyncio
    async def test_keyword_no_partial_transcript_falls_through(self):
        """When STT timed out (partial_transcript=None), the keyword check is
        silently skipped and the audio/duration paths decide.
        """
        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"stop"}),
            min_real_interrupt_ms=400,
        )

        assert await c.is_real_interrupt(None, None, None, 300) is False

        assert await c.is_real_interrupt(None, None, None, 500) is True


class TestProtocolSurface:
    def test_heuristic_implements_protocol(self):
        c = HeuristicInterruptClassifier()
        assert isinstance(c, InterruptClassifier)

    @pytest.mark.asyncio
    async def test_custom_classifier_can_replace_default(self):
        class AlwaysBackchannel:
            def confirm_window_ms(self, base_ms, last_eou_probability):
                return 250

            async def is_real_interrupt(self, audio, partial_transcript, eou, duration_ms):
                return False


        assert isinstance(AlwaysBackchannel(), InterruptClassifier)
        c = AlwaysBackchannel()
        assert c.confirm_window_ms(300, 0.5) == 250
        assert await c.is_real_interrupt(None, None, None, 500) is False
