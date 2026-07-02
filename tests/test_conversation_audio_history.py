from __future__ import annotations

import numpy as np

from vox.conversation.audio_history import ConversationAudioHistory
from vox.streaming.codecs import float32_to_pcm16


def _voice_signal(duration_s: float, amp: float = 0.1, sr: int = 16_000, freq: float = 220) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mic_history_is_bounded() -> None:
    history = ConversationAudioHistory(mic_window_ms=200)
    history.append_mic(_voice_signal(1.0))

    assert history.mic_size == history.mic_max_samples


def test_recent_output_echo_allows_playout_delay() -> None:
    history = ConversationAudioHistory()
    voice = _voice_signal(0.32, amp=0.08, freq=330)
    trailing_audio = np.zeros(int(0.18 * 16_000), dtype=np.float32)

    history.replace_output(np.concatenate([voice, trailing_audio]))
    history.replace_mic(voice)

    assert history.looks_like_current_output_echo()


def test_low_rms_matching_audio_is_not_echo() -> None:
    history = ConversationAudioHistory()
    quiet_voice = _voice_signal(0.32, amp=0.0005, freq=330)

    history.replace_output(quiet_voice)
    history.replace_mic(quiet_voice)

    assert not history.looks_like_current_output_echo()


def test_output_history_accepts_pcm16_at_client_sample_rate() -> None:
    history = ConversationAudioHistory()
    voice = _voice_signal(0.32, amp=0.08, sr=24_000, freq=330)

    history.remember_output_pcm16(float32_to_pcm16(voice), 24_000)
    history.replace_mic(_voice_signal(0.32, amp=0.08, freq=330))

    assert history.output_size > 0
    assert history.looks_like_current_output_echo()
