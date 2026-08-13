import numpy as np

from vox.conversation.spoken_history import (
    ResponseSpokenHistory,
    SpokenHistorySnapshot,
    longest_confident_source_prefix,
    resolve_spoken_text,
)
from vox.streaming.codecs import float32_to_pcm16


def _pcm(duration_s: float, *, sample_rate: int = 16_000, frequency: float = 220.0) -> bytes:
    samples = int(duration_s * sample_rate)
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    return float32_to_pcm16(0.1 * np.sin(2 * np.pi * frequency * timeline))


def test_completed_spans_release_audio_and_preserve_exact_source_text() -> None:
    history = ResponseSpokenHistory(playout_available=True)
    first = history.begin_span("The first sentence.")
    audio = _pcm(0.4)
    history.register_audio(first, audio)
    history.finish_span(first)
    history.observe_playout(audio, 16_000)

    assert history.completed_text() == "The first sentence."
    assert history.retained_audio_bytes == 0
    snapshot = history.snapshot()
    assert snapshot.completed_text == "The first sentence."
    assert snapshot.partial_source_text == ""


def test_partial_span_captures_only_observed_playout() -> None:
    history = ResponseSpokenHistory(playout_available=True)
    span = history.begin_span("The browser retains the audio queue for later playback.")
    audio = _pcm(1.0)
    history.register_audio(span, audio)
    history.observe_playout(audio[: len(audio) // 2], 16_000)

    snapshot = history.snapshot()

    assert snapshot.completed_text == ""
    assert snapshot.partial_source_text == "The browser retains the audio queue for later playback."
    assert snapshot.partial_capture_complete is True
    assert 490 <= snapshot.played_audio_ms <= 500
    assert len(snapshot.partial_audio_pcm16) == len(audio) // 2


def test_partial_capture_has_a_hard_byte_limit() -> None:
    history = ResponseSpokenHistory(playout_available=True, max_partial_capture_bytes=64)
    span = history.begin_span("A long spoken span.")
    audio = _pcm(0.1)
    history.register_audio(span, audio)
    history.observe_playout(audio, 16_000)

    snapshot = history.snapshot()

    assert history.retained_audio_bytes == 0
    assert snapshot.partial_capture_complete is False
    assert snapshot.partial_audio_pcm16 == b""
    assert resolve_spoken_text(snapshot, "A long spoken span").partial_status == "partial_omitted"


def test_unobserved_transport_never_claims_text_was_spoken() -> None:
    history = ResponseSpokenHistory(playout_available=False)
    span = history.begin_span("Generated but not observed.")
    audio = _pcm(0.2)
    history.register_audio(span, audio)
    history.finish_span(span)
    history.observe_playout(audio, 16_000)

    resolution = resolve_spoken_text(history.snapshot(), "Generated but not observed")

    assert resolution.spoken_text == ""
    assert resolution.partial_status == "playout_unavailable"


def test_matcher_returns_only_a_confident_prefix_of_known_source_text() -> None:
    source = "The browser retains the audio queue so that it can resume later."

    assert longest_confident_source_prefix(source, "the browser retains the audio queue") == (
        "The browser retains the audio queue"
    )
    assert longest_confident_source_prefix(source, "a different sentence entirely") == ""
    assert longest_confident_source_prefix(source, "the browser retains the audio hallucination") == (
        "The browser retains the audio"
    )


def test_resolution_combines_completed_spans_with_matched_partial_prefix() -> None:
    snapshot = SpokenHistorySnapshot(
        completed_text="The first sentence.",
        partial_source_text="The browser retains the audio queue for later playback.",
        partial_audio_pcm16=_pcm(0.5),
        sample_rate=16_000,
        playout_available=True,
        partial_capture_complete=True,
        played_audio_ms=500,
    )

    resolution = resolve_spoken_text(snapshot, "The browser retains the audio queue")

    assert resolution.spoken_text == "The first sentence. The browser retains the audio queue"
    assert resolution.partial_status == "matched"
