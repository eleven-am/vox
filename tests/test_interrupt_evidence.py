from __future__ import annotations

from vox.conversation.interrupt import (
    PartialInterruptEvidence,
    has_recent_interrupt_context,
    interrupt_vad_active_ms,
    transcript_duration_ms,
    transcript_word_count,
)
from vox.conversation.types import TurnPolicy
from vox.streaming.types import StreamTranscript


def _evidence(**policy_kwargs) -> PartialInterruptEvidence:
    policy = {
        "min_interrupt_duration_ms": 300,
        "speaking_interrupt_min_words": 2,
        "self_echo_min_words": 3,
        "self_echo_min_overlap": 0.7,
        **policy_kwargs,
    }
    return PartialInterruptEvidence.from_turn_policy(
        TurnPolicy(**policy)
    )


def test_transcript_duration_prefers_audio_duration() -> None:
    transcript = StreamTranscript(start_ms=100, end_ms=900, audio_duration_ms=450)

    assert transcript_duration_ms(transcript) == 450


def test_transcript_duration_falls_back_to_timestamp_span() -> None:
    transcript = StreamTranscript(start_ms=100, end_ms=900)

    assert transcript_duration_ms(transcript) == 800


def test_transcript_word_count_ignores_blank_words() -> None:
    assert transcript_word_count("  hello   there ") == 2
    assert transcript_word_count(None) == 0
    assert transcript_word_count("   ") == 0


def test_interrupt_vad_active_ms_uses_latest_partial_when_longer() -> None:
    partial = StreamTranscript(start_ms=0, end_ms=900)

    assert interrupt_vad_active_ms(vad_started_at=10.0, latest_partial=partial, now=10.2) == 900


def test_interrupt_vad_active_ms_clamps_negative_vad_age() -> None:
    assert interrupt_vad_active_ms(vad_started_at=12.0, latest_partial=None, now=10.0) == 0


def test_recent_interrupt_context_accepts_timer_or_active_vad() -> None:
    assert has_recent_interrupt_context(
        confirm_timer_active=True,
        vad_started_at=None,
        last_speech_stopped_at=None,
        false_interruption_timeout_ms=2000,
        now=10.0,
    )
    assert has_recent_interrupt_context(
        confirm_timer_active=False,
        vad_started_at=9.0,
        last_speech_stopped_at=None,
        false_interruption_timeout_ms=2000,
        now=10.0,
    )


def test_recent_interrupt_context_uses_speech_stop_timeout() -> None:
    assert has_recent_interrupt_context(
        confirm_timer_active=False,
        vad_started_at=None,
        last_speech_stopped_at=8.1,
        false_interruption_timeout_ms=2000,
        now=10.0,
    )
    assert not has_recent_interrupt_context(
        confirm_timer_active=False,
        vad_started_at=None,
        last_speech_stopped_at=7.9,
        false_interruption_timeout_ms=2000,
        now=10.0,
    )
    assert not has_recent_interrupt_context(
        confirm_timer_active=False,
        vad_started_at=None,
        last_speech_stopped_at=None,
        false_interruption_timeout_ms=2000,
        now=10.0,
    )


def test_strong_partial_requires_configured_word_count() -> None:
    evidence = _evidence(speaking_interrupt_min_words=3)
    transcript = StreamTranscript(text="stop now", start_ms=0, end_ms=800)

    assert not evidence.is_strong(transcript, assistant_text="assistant is speaking")


def test_strong_partial_rejects_short_audio_span() -> None:
    evidence = _evidence(min_interrupt_duration_ms=300)
    transcript = StreamTranscript(text="please stop now", start_ms=0, end_ms=120)

    assert not evidence.is_strong(transcript, assistant_text="assistant is speaking")


def test_strong_partial_rejects_assistant_self_echo() -> None:
    evidence = _evidence()
    transcript = StreamTranscript(text="the answer is forty two", start_ms=0, end_ms=800)

    assert not evidence.is_strong(
        transcript,
        assistant_text="The answer is forty two and here is why.",
    )


def test_strong_partial_accepts_non_echo_with_enough_words_and_duration() -> None:
    evidence = _evidence()
    transcript = StreamTranscript(text="please stop now", start_ms=0, end_ms=800)

    assert evidence.is_strong(transcript, assistant_text="assistant is speaking")
