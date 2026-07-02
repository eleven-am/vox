from __future__ import annotations

import numpy as np
import pytest

from vox.streaming.opus import OpusStreamEncoder

try:
    import opuslib

    opuslib.Encoder(48_000, 1, "audio")
    _OPUS_AVAILABLE = True
except Exception:
    _OPUS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _OPUS_AVAILABLE, reason="native opus library not installed"
)


def test_flush_drains_resampler_tail_when_resampling():
    encoder = OpusStreamEncoder(source_rate=24_000)
    t = np.linspace(0, 1, 24_000, endpoint=False)
    pcm = (np.sin(2 * np.pi * 220 * t) * 12_000).astype(np.int16).tobytes()
    encoder.encode(pcm)

    calls: list[int] = []
    original = encoder._resampler.flush

    def spy():
        calls.append(1)
        return original()

    encoder._resampler.flush = spy
    frames = encoder.flush()

    assert calls, "flush() must drain the stateful resampler so trailing audio is not dropped"
    assert all(isinstance(frame, bytes) for frame in frames)


def test_flush_without_resampling_still_pads_final_frame():
    encoder = OpusStreamEncoder(source_rate=48_000)
    pcm = np.zeros(48_000 // 100, dtype=np.int16).tobytes()
    encoder.encode(pcm)
    frames = encoder.flush()
    assert all(isinstance(frame, bytes) for frame in frames)
