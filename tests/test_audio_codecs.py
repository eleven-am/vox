"""Tests for vox.audio.codecs — encoding, decoding, and channel conversion."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.audio.codecs import (
    decode_audio,
    encode_flac,
    encode_pcm,
    encode_wav,
    pcm16_to_float32,
    to_mono,
)





SAMPLE_RATE = 44100


def _sine_wave(freq: float = 440.0, duration: float = 0.1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a short sine wave as float32."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)







def test_encode_wav_decode_roundtrip():
    audio = _sine_wave()
    wav_bytes = encode_wav(audio, SAMPLE_RATE)
    decoded, sr = decode_audio(wav_bytes)

    assert sr == SAMPLE_RATE
    assert decoded.shape == audio.shape
    np.testing.assert_allclose(decoded, audio, atol=1e-6)


def test_decode_wav_with_format_hint_uses_container_detection():
    audio = _sine_wave()
    wav_bytes = encode_wav(audio, SAMPLE_RATE)
    decoded, sr = decode_audio(wav_bytes, format_hint="wav")

    assert sr == SAMPLE_RATE
    assert decoded.shape == audio.shape
    np.testing.assert_allclose(decoded, audio, atol=1e-6)







def test_encode_flac_decode_roundtrip():
    audio = _sine_wave() * 0.5
    flac_bytes = encode_flac(audio, SAMPLE_RATE)
    decoded, sr = decode_audio(flac_bytes)

    assert sr == SAMPLE_RATE
    assert decoded.shape == audio.shape

    np.testing.assert_allclose(decoded, audio, atol=2.0 / 32768)







def test_encode_pcm_pcm16_to_float32_roundtrip():
    audio = _sine_wave() * 0.5
    pcm_bytes = encode_pcm(audio)
    recovered = pcm16_to_float32(pcm_bytes)

    assert recovered.shape == audio.shape

    np.testing.assert_allclose(recovered, audio, atol=2.0 / 32768)







def test_to_mono_passthrough_for_1d():
    mono = _sine_wave()
    result = to_mono(mono)
    assert result.ndim == 1
    np.testing.assert_array_equal(result, mono)


def test_to_mono_averages_stereo():
    left = _sine_wave(440.0)
    right = _sine_wave(880.0)
    stereo = np.stack([left, right], axis=1)

    mono = to_mono(stereo)
    expected = (left + right) / 2.0

    assert mono.ndim == 1
    assert mono.shape[0] == left.shape[0]
    np.testing.assert_allclose(mono, expected, atol=1e-6)







def test_decode_audio_invalid_bytes_raises():
    with pytest.raises((RuntimeError, Exception)):
        decode_audio(b"this is not audio data at all")







def _mock_pydub_module():
    """Create a mock pydub module that works even when pydub can't import (Python 3.13+)."""
    mock_pydub = MagicMock()
    mock_AudioSegment = MagicMock()
    mock_pydub.AudioSegment = mock_AudioSegment
    return mock_pydub, mock_AudioSegment


def test_decode_audio_pydub_fallback():
    """When soundfile raises SoundFileError, decode_audio should fall back to pydub."""
    import soundfile as sf

    fake_segment = MagicMock()
    fake_segment.frame_rate = 16000
    fake_segment.channels = 1
    fake_segment.get_array_of_samples.return_value = [0, 1000, -1000, 500]

    mock_pydub, mock_AudioSegment = _mock_pydub_module()
    mock_AudioSegment.from_file.return_value = fake_segment

    with (
        patch("soundfile.read", side_effect=sf.SoundFileError("unsupported format")),
        patch.dict(sys.modules, {"pydub": mock_pydub, "pydub.AudioSegment": mock_AudioSegment}),
    ):
        audio, sr = decode_audio(b"\xff\xfb\x90\x00fake-mp3-data")

    assert sr == 16000
    assert audio.dtype == np.float32
    assert audio.shape == (4,)
    np.testing.assert_allclose(audio, np.array([0, 1000, -1000, 500], dtype=np.float32) / 32768.0, atol=1e-6)


def test_decode_audio_pydub_fallback_stereo():
    """Pydub fallback should correctly reshape interleaved stereo samples."""
    import soundfile as sf

    fake_segment = MagicMock()
    fake_segment.frame_rate = 44100
    fake_segment.channels = 2
    fake_segment.get_array_of_samples.return_value = [100, 200, 300, 400]

    mock_pydub, mock_AudioSegment = _mock_pydub_module()
    mock_AudioSegment.from_file.return_value = fake_segment

    with (
        patch("soundfile.read", side_effect=sf.SoundFileError("bad")),
        patch.dict(sys.modules, {"pydub": mock_pydub, "pydub.AudioSegment": mock_AudioSegment}),
    ):
        audio, sr = decode_audio(b"\x00" * 10)

    assert sr == 44100
    assert audio.shape == (2, 2)


def test_decode_audio_both_fail_raises_runtime_error():
    """When both soundfile and pydub fail, a RuntimeError with both messages should be raised."""
    import soundfile as sf

    mock_pydub, mock_AudioSegment = _mock_pydub_module()
    mock_AudioSegment.from_file.side_effect = ValueError("pydub broke")

    with (
        patch("soundfile.read", side_effect=sf.SoundFileError("sf broke")),
        patch.dict(sys.modules, {"pydub": mock_pydub, "pydub.AudioSegment": mock_AudioSegment}),
        pytest.raises(RuntimeError, match="sf broke") as exc_info,
    ):
        decode_audio(b"\x00" * 10)
    assert "pydub broke" in str(exc_info.value)
