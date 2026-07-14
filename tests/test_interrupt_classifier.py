from __future__ import annotations

import numpy as np
import pytest

from vox.conversation import HeuristicInterruptClassifier, InterruptClassifier
from vox.conversation.interrupt import (
    DEFAULT_INTERRUPT_KEYWORDS_BY_LANG,
    InterruptCandidateAction,
    PartialInterruptEvidence,
    evaluate_interrupt_candidate_gate,
    looks_like_self_echo,
)
from vox.conversation.types import TurnPolicy
from vox.streaming.types import StreamTranscript


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
        assert await c.is_real_interrupt(None, None, None, 179, 16_000) is False
        assert await c.is_real_interrupt(None, None, None, 180, 16_000) is True

    @pytest.mark.asyncio
    async def test_short_duration_rejected(self):
        c = HeuristicInterruptClassifier(min_real_interrupt_ms=400)
        assert await c.is_real_interrupt(None, None, None, 300, 16_000) is False

    @pytest.mark.asyncio
    async def test_long_duration_accepted(self):
        c = HeuristicInterruptClassifier(min_real_interrupt_ms=400)
        assert await c.is_real_interrupt(None, None, None, 500, 16_000) is True

    @pytest.mark.asyncio
    async def test_threshold_boundary(self):
        c = HeuristicInterruptClassifier(min_real_interrupt_ms=400)
        assert await c.is_real_interrupt(None, None, None, 399, 16_000) is False
        assert await c.is_real_interrupt(None, None, None, 400, 16_000) is True


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
        assert await c.is_real_interrupt(audio, None, None, duration_ms, sr) is False

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
        assert await c.is_real_interrupt(audio, None, None, 600, sr) is True

    @pytest.mark.asyncio
    async def test_tail_shorter_than_tail_check_still_works(self):
        """If the buffer is shorter than tail_check_ms, still evaluates against what's there."""
        c = HeuristicInterruptClassifier(
            tail_check_ms=500, backchannel_rms_threshold=0.015,
        )
        sr = 16_000
        samples = int(0.1 * sr)
        audio = np.zeros(samples, dtype=np.float32)
        assert await c.is_real_interrupt(audio, None, None, 100, sr) is False

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

        assert await c_strict.is_real_interrupt(audio, None, None, 400, sr) is False


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
        assert await c.is_real_interrupt(audio, "please stop talking", None, 500, sr) is True

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
        assert await c.is_real_interrupt(audio, "please stop talking", None, 500, sr) is False

    @pytest.mark.asyncio
    async def test_keyword_substring_match(self):
        """Keyword check is substring, not token — 'halt' hits 'haltet euch'."""
        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"halt"}),
        )
        assert await c.is_real_interrupt(None, "haltet euch zurueck", None, 100, 16_000) is True

    @pytest.mark.asyncio
    async def test_keyword_case_insensitive(self):
        """Keyword compare is lowercase — 'STOP' and 'Stop' both match."""
        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"stop"}),
        )
        assert await c.is_real_interrupt(None, "STOP", None, 100, 16_000) is True
        assert await c.is_real_interrupt(None, "Stop please", None, 100, 16_000) is True

    @pytest.mark.asyncio
    async def test_keyword_language_neutral_non_latin(self):
        """Operators can supply keywords in any script — CJK, Devanagari, etc.
        Proves no hardcoded language assumptions in the classifier.
        """

        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"やめて", "停"}),
        )
        assert await c.is_real_interrupt(None, "やめて", None, 100, 16_000) is True
        assert await c.is_real_interrupt(None, "停下来", None, 100, 16_000) is True


        assert await c.is_real_interrupt(None, "こんにちは", None, 100, 16_000) is False

    @pytest.mark.asyncio
    async def test_keyword_no_partial_transcript_falls_through(self):
        """When STT timed out (partial_transcript=None), the keyword check is
        silently skipped and the audio/duration paths decide.
        """
        c = HeuristicInterruptClassifier(
            interrupt_keywords=frozenset({"stop"}),
            min_real_interrupt_ms=400,
        )

        assert await c.is_real_interrupt(None, None, None, 300, 16_000) is False

        assert await c.is_real_interrupt(None, None, None, 500, 16_000) is True


class TestSelfEchoHeuristic:
    def test_contiguous_assistant_phrase_matches(self):
        assert looks_like_self_echo(
            "the appointment is tomorrow",
            "Sure. The appointment is tomorrow at noon.",
        )

    def test_high_word_overlap_matches(self):
        assert looks_like_self_echo(
            "appointment tomorrow noon",
            "Sure. The appointment is tomorrow at noon.",
        )

    def test_short_transcript_does_not_match(self):
        assert not looks_like_self_echo("sure", "Sure. The appointment is tomorrow at noon.")

    def test_unrelated_user_interrupt_does_not_match(self):
        assert not looks_like_self_echo(
            "wait I need to change that",
            "Sure. The appointment is tomorrow at noon.",
        )

    def test_non_latin_transcript_matches_assistant_echo(self):
        assert looks_like_self_echo(
            "الموعد غدا عند الظهر",
            "حسنا، الموعد غدا عند الظهر بالتأكيد.",
            min_words=3,
        )


class TestInterruptCandidateGate:
    def test_rejects_partial_transcript_self_echo(self):
        policy = TurnPolicy(self_echo_min_words=3, self_echo_min_overlap=0.7)
        decision = evaluate_interrupt_candidate_gate(
            partial=StreamTranscript(text="the appointment is tomorrow", start_ms=0, end_ms=800),
            active_assistant_text="Sure. The appointment is tomorrow at noon.",
            policy=policy,
            evidence=PartialInterruptEvidence.from_turn_policy(policy),
            is_interrupt_keyword=False,
            output_echo=False,
            vad_active_ms=500,
        )

        assert decision.action is InterruptCandidateAction.REJECT
        assert decision.false_positive_reason == "self_echo_transcript"
        assert decision.speech_stopped_reason == "self_echo"

    def test_strong_partial_overrides_acoustic_echo_for_double_talk(self):
        policy = TurnPolicy(speaking_interrupt_min_words=1)
        decision = evaluate_interrupt_candidate_gate(
            partial=StreamTranscript(
                text="please stop now",
                start_ms=0,
                end_ms=1200,
                audio_duration_ms=1200,
            ),
            active_assistant_text="assistant text",
            policy=policy,
            evidence=PartialInterruptEvidence.from_turn_policy(policy),
            is_interrupt_keyword=True,
            output_echo=True,
            vad_active_ms=1200,
        )

        assert decision.action is InterruptCandidateAction.CONFIRM_FROM_PARTIAL

    def test_rejects_acoustic_output_echo_without_transcript_evidence(self):
        policy = TurnPolicy(speaking_interrupt_min_words=2)
        decision = evaluate_interrupt_candidate_gate(
            partial=None,
            active_assistant_text="assistant text",
            policy=policy,
            evidence=PartialInterruptEvidence.from_turn_policy(policy),
            is_interrupt_keyword=False,
            output_echo=True,
            vad_active_ms=500,
        )

        assert decision.action is InterruptCandidateAction.REJECT
        assert decision.false_positive_reason == "output_echo"

    def test_confirms_strong_partial_interrupt_evidence(self):
        policy = TurnPolicy(speaking_interrupt_min_words=2, min_interrupt_duration_ms=250)
        decision = evaluate_interrupt_candidate_gate(
            partial=StreamTranscript(
                text="wait stop now",
                start_ms=0,
                end_ms=800,
                audio_duration_ms=800,
            ),
            active_assistant_text="assistant is still speaking",
            policy=policy,
            evidence=PartialInterruptEvidence.from_turn_policy(policy),
            is_interrupt_keyword=False,
            output_echo=False,
            vad_active_ms=800,
        )

        assert decision.action is InterruptCandidateAction.CONFIRM_FROM_PARTIAL

    def test_rejects_weak_partial_inside_false_interruption_window(self):
        policy = TurnPolicy(
            speaking_interrupt_min_words=2,
            false_interruption_timeout_ms=2000,
        )
        decision = evaluate_interrupt_candidate_gate(
            partial=StreamTranscript(text="uh", start_ms=0, end_ms=300),
            active_assistant_text="assistant is still speaking",
            policy=policy,
            evidence=PartialInterruptEvidence.from_turn_policy(policy),
            is_interrupt_keyword=False,
            output_echo=False,
            vad_active_ms=300,
        )

        assert decision.action is InterruptCandidateAction.REJECT
        assert decision.false_positive_reason == "insufficient_interrupt_evidence"
        assert decision.speech_stopped_reason == "insufficient_interrupt_evidence"

    def test_continues_to_classifier_when_no_pre_gate_applies(self):
        policy = TurnPolicy(speaking_interrupt_min_words=2)
        decision = evaluate_interrupt_candidate_gate(
            partial=None,
            active_assistant_text="assistant is still speaking",
            policy=policy,
            evidence=PartialInterruptEvidence.from_turn_policy(policy),
            is_interrupt_keyword=False,
            output_echo=False,
            vad_active_ms=2200,
        )

        assert decision.action is InterruptCandidateAction.CONTINUE


class TestLanguageDefaults:
    def test_language_loads_default_keywords(self):
        c = HeuristicInterruptClassifier(language="en")
        assert "stop" in c.interrupt_keywords
        assert "wait" in c.interrupt_keywords

    def test_explicit_keywords_override_language_defaults(self):
        custom = frozenset({"abracadabra"})
        c = HeuristicInterruptClassifier(language="en", interrupt_keywords=custom)
        assert c.interrupt_keywords == custom

    def test_no_language_means_empty_keywords(self):
        c = HeuristicInterruptClassifier()
        assert c.interrupt_keywords == frozenset()

    def test_unknown_language_leaves_keywords_empty(self):
        c = HeuristicInterruptClassifier(language="xx")
        assert c.interrupt_keywords == frozenset()

    def test_language_lookup_is_case_insensitive(self):
        c = HeuristicInterruptClassifier(language="EN")
        assert "stop" in c.interrupt_keywords

    def test_all_supported_languages_have_defaults(self):
        for lang in ("en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi"):
            assert lang in DEFAULT_INTERRUPT_KEYWORDS_BY_LANG
            assert len(DEFAULT_INTERRUPT_KEYWORDS_BY_LANG[lang]) > 0


class TestMinimumWords:
    @pytest.mark.asyncio
    async def test_min_interrupt_words_rejects_short_partial(self):
        c = HeuristicInterruptClassifier(min_interrupt_words=2)
        assert await c.is_real_interrupt(None, "wait", None, 500, 16_000) is False
        assert await c.is_real_interrupt(None, "wait please", None, 500, 16_000) is True

    @pytest.mark.asyncio
    async def test_keyword_override_bypasses_min_words(self):
        c = HeuristicInterruptClassifier(
            min_interrupt_words=3,
            interrupt_keywords=frozenset({"stop"}),
        )
        assert await c.is_real_interrupt(None, "stop", None, 100, 16_000) is True


class TestShortCircuit:
    def test_wants_short_circuit_false_without_keywords(self):
        assert HeuristicInterruptClassifier().wants_short_circuit() is False

    def test_wants_short_circuit_true_with_keywords(self):
        c = HeuristicInterruptClassifier(interrupt_keywords=frozenset({"stop"}))
        assert c.wants_short_circuit() is True

    def test_wants_short_circuit_true_via_language_defaults(self):
        c = HeuristicInterruptClassifier(language="en")
        assert c.wants_short_circuit() is True

    def test_should_short_circuit_keyword_match(self):
        c = HeuristicInterruptClassifier(interrupt_keywords=frozenset({"stop"}))
        assert c.should_short_circuit("please stop talking") is True

    def test_should_short_circuit_case_insensitive(self):
        c = HeuristicInterruptClassifier(interrupt_keywords=frozenset({"stop"}))
        assert c.should_short_circuit("STOP NOW") is True

    def test_should_short_circuit_no_match(self):
        c = HeuristicInterruptClassifier(interrupt_keywords=frozenset({"stop"}))
        assert c.should_short_circuit("continue please") is False

    def test_should_short_circuit_no_keywords_no_match(self):
        c = HeuristicInterruptClassifier()
        assert c.should_short_circuit("stop") is False

    def test_should_short_circuit_none_partial(self):
        c = HeuristicInterruptClassifier(interrupt_keywords=frozenset({"stop"}))
        assert c.should_short_circuit(None) is False

    def test_should_short_circuit_empty_partial(self):
        c = HeuristicInterruptClassifier(interrupt_keywords=frozenset({"stop"}))
        assert c.should_short_circuit("") is False
        assert c.should_short_circuit("   ") is False


class TestSampleRateExplicit:
    @pytest.mark.asyncio
    async def test_tail_sized_to_supplied_rate(self):
        c = HeuristicInterruptClassifier(
            tail_check_ms=80, backchannel_rms_threshold=0.015,
        )
        sr = 24_000
        loud_head = (0.5 * np.sin(2 * np.pi * 220 * np.arange(int(0.4 * sr)) / sr)).astype(np.float32)
        silent_tail = np.zeros(int(0.2 * sr), dtype=np.float32)
        audio = np.concatenate([loud_head, silent_tail])
        duration_ms = int(1000 * audio.size / sr)
        assert await c.is_real_interrupt(audio, None, None, duration_ms, sr) is False

    @pytest.mark.asyncio
    async def test_different_rates_change_tail_window(self):
        c = HeuristicInterruptClassifier(
            tail_check_ms=200, backchannel_rms_threshold=0.015,
        )
        sr_ref = 16_000
        voice = (0.2 * np.sin(2 * np.pi * 220 * np.arange(4800) / sr_ref)).astype(np.float32)
        silence = np.zeros(3200, dtype=np.float32)
        audio = np.concatenate([voice, silence])
        assert await c.is_real_interrupt(audio, None, None, 500, 16_000) is False
        assert await c.is_real_interrupt(audio, None, None, 250, 32_000) is True


class TestProtocolSurface:
    def test_heuristic_implements_protocol(self):
        c = HeuristicInterruptClassifier()
        assert isinstance(c, InterruptClassifier)

    @pytest.mark.asyncio
    async def test_custom_classifier_can_replace_default(self):
        class AlwaysBackchannel:
            def confirm_window_ms(self, base_ms, last_eou_probability):
                return 250

            def wants_short_circuit(self):
                return False

            def should_short_circuit(self, partial_transcript):
                return False

            async def is_real_interrupt(self, audio, partial_transcript, eou, duration_ms, sample_rate):
                return False


        assert isinstance(AlwaysBackchannel(), InterruptClassifier)
        c = AlwaysBackchannel()
        assert c.confirm_window_ms(300, 0.5) == 250
        assert await c.is_real_interrupt(None, None, None, 500, 16_000) is False
