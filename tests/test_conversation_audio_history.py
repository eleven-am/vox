from __future__ import annotations

import fractions

import av
import numpy as np
from aiortc.codecs.opus import OpusDecoder, OpusEncoder
from aiortc.jitterbuffer import JitterFrame
from av.audio.resampler import AudioResampler

from vox.conversation.audio_history import ConversationAudioHistory
from vox.streaming.codecs import float32_to_pcm16


def _voice_signal(duration_s: float, amp: float = 0.1, sr: int = 16_000, freq: float = 220) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _speech_like_signal(seed: int, duration_s: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample_rate = 16_000
    signal = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    segment_samples = int(0.08 * sample_rate)
    for start in range(0, signal.size, segment_samples):
        size = min(segment_samples, signal.size - start)
        frequency = float(rng.choice([130, 170, 210, 260, 320, 390]))
        amplitude = float(rng.uniform(0.03, 0.12))
        time = np.arange(size) / sample_rate
        envelope = np.sqrt(np.sin(np.pi * np.arange(size) / size))
        signal[start:start + size] = amplitude * envelope * (
            np.sin(2 * np.pi * frequency * time)
            + 0.35 * np.sin(4 * np.pi * frequency * time)
            + 0.15 * np.sin(6 * np.pi * frequency * time)
        )
    return signal


def _opus_transport_pass(audio: np.ndarray) -> np.ndarray:
    source = av.AudioFrame.from_ndarray(
        (audio * 32767).astype(np.int16).reshape(1, -1),
        format="s16",
        layout="mono",
    )
    source.sample_rate = 16_000
    to_transport = AudioResampler(format="s16", layout="mono", rate=48_000)
    transport_samples = np.concatenate(
        [frame.to_ndarray().reshape(-1) for frame in to_transport.resample(source)]
    )

    encoder = OpusEncoder()
    decoder = OpusDecoder()
    decoded: list[av.AudioFrame] = []
    frame_samples = 960
    for start in range(0, transport_samples.size, frame_samples):
        samples = transport_samples[start:start + frame_samples]
        if samples.size < frame_samples:
            samples = np.pad(samples, (0, frame_samples - samples.size))
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = 48_000
        frame.pts = start
        frame.time_base = fractions.Fraction(1, 48_000)
        packets, timestamp = encoder.encode(frame)
        for offset, packet in enumerate(packets):
            decoded.extend(
                decoder.decode(
                    JitterFrame(packet, int(timestamp) + offset * frame_samples)
                )
            )

    from_transport = AudioResampler(format="s16", layout="mono", rate=16_000)
    result = np.concatenate(
        [
            converted.to_ndarray().reshape(-1)
            for frame in decoded
            for converted in from_transport.resample(frame)
        ]
    )
    return result.astype(np.float32) / 32768.0


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


def test_output_echo_survives_two_non_frame_aligned_opus_passes() -> None:
    output = _speech_like_signal(seed=1)
    microphone = _opus_transport_pass(_opus_transport_pass(output))

    history = ConversationAudioHistory()
    history.replace_output(output)
    history.replace_mic(microphone)

    assert history.looks_like_current_output_echo()


def test_unrelated_speech_is_not_misclassified_as_output_echo_after_opus() -> None:
    output = _speech_like_signal(seed=1)
    microphone = _opus_transport_pass(_opus_transport_pass(_speech_like_signal(seed=2)))

    history = ConversationAudioHistory()
    history.replace_output(output)
    history.replace_mic(microphone)

    assert not history.looks_like_current_output_echo()
