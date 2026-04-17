"""Tests for vox.audio.pipeline — STT preparation and output encoding."""

from __future__ import annotations

import numpy as np
import pytest

from vox.audio.codecs import decode_audio, encode_wav
from vox.audio.pipeline import get_content_type, prepare_for_output, prepare_for_stt





SOURCE_RATE = 44100
TARGET_STT_RATE = 16000


def _sine_wav_bytes(
    freq: float = 440.0,
    duration: float = 0.25,
    sr: int = SOURCE_RATE,
    channels: int = 1,
) -> bytes:
    """Return WAV bytes for a short sine tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)
    if channels == 2:
        audio = np.stack([audio, audio], axis=1).astype(np.float32)
    return encode_wav(audio, sr)







class TestPrepareForSTT:
    def test_prepare_for_stt_resamples_to_target_rate(self):
        wav = _sine_wav_bytes(sr=SOURCE_RATE, duration=0.5)
        result = prepare_for_stt(wav, target_rate=TARGET_STT_RATE)

        expected_len = int(0.5 * TARGET_STT_RATE)

        assert abs(result.shape[0] - expected_len) <= 2

    def test_prepare_for_stt_converts_to_mono(self):
        wav = _sine_wav_bytes(channels=2)
        result = prepare_for_stt(wav, target_rate=TARGET_STT_RATE)
        assert result.ndim == 1

    def test_prepare_for_stt_normalizes_peak(self):
        wav = _sine_wav_bytes()
        result = prepare_for_stt(wav, target_rate=TARGET_STT_RATE)
        peak = np.max(np.abs(result))
        assert peak == pytest.approx(1.0, abs=1e-5)

    def test_prepare_for_stt_silent_audio_no_crash(self):

        silence = np.zeros(4410, dtype=np.float32)
        wav = encode_wav(silence, SOURCE_RATE)
        result = prepare_for_stt(wav, target_rate=TARGET_STT_RATE)

        assert result.ndim == 1
        assert np.max(np.abs(result)) == pytest.approx(0.0, abs=1e-7)







class TestPrepareForOutput:
    _audio = np.sin(
        2 * np.pi * 440 * np.linspace(0, 0.1, int(0.1 * 24000), endpoint=False, dtype=np.float32)
    ).astype(np.float32)

    def test_prepare_for_output_wav(self):
        data, mime = prepare_for_output(self._audio, 24000, "wav")
        assert mime == "audio/wav"
        assert len(data) > 0

        decoded, sr = decode_audio(data)
        assert sr == 24000

    def test_prepare_for_output_flac(self):
        data, mime = prepare_for_output(self._audio, 24000, "flac")
        assert mime == "audio/flac"
        assert len(data) > 0
        decoded, sr = decode_audio(data)
        assert sr == 24000

    def test_prepare_for_output_pcm(self):
        data, mime = prepare_for_output(self._audio, 24000, "pcm")
        assert mime == "audio/L16"

        assert len(data) == self._audio.shape[0] * 2

    def test_prepare_for_output_mp3_returns_mp3_bytes(self):
        data, mime = prepare_for_output(self._audio, 24000, "mp3")
        assert mime == "audio/mpeg"
        assert len(data) > 500
        assert data[0] == 0xFF
        assert data[1] & 0xE0 == 0xE0

    def test_prepare_for_output_opus_returns_ogg_opus(self):
        data, mime = prepare_for_output(self._audio, 24000, "opus")
        assert mime == "audio/opus"
        assert data[:4] == b"OggS"

    def test_prepare_for_output_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported output format"):
            prepare_for_output(self._audio, 24000, "aac")







class TestGetContentType:
    @pytest.mark.parametrize(
        "fmt, expected",
        [
            ("wav", "audio/wav"),
            ("flac", "audio/flac"),
            ("pcm", "audio/L16"),
            ("mp3", "audio/mpeg"),
            ("opus", "audio/opus"),
        ],
    )
    def test_get_content_type_known_formats(self, fmt: str, expected: str):
        assert get_content_type(fmt) == expected

    def test_get_content_type_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            get_content_type("aac")
