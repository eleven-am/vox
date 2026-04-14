from __future__ import annotations

import numpy as np
import soxr
from numpy.typing import NDArray

from vox.streaming.types import TARGET_SAMPLE_RATE


def pcm16_to_float32(pcm_bytes: bytes) -> NDArray[np.float32]:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: NDArray[np.float32]) -> bytes:
    clamped = np.clip(audio, -1.0, 1.0)
    return (clamped * 32767).astype(np.int16).tobytes()


def resample_audio(audio: NDArray[np.float32], source_rate: int, target_rate: int) -> NDArray[np.float32]:
    if source_rate == target_rate:
        return audio
    return soxr.resample(audio, source_rate, target_rate).astype(np.float32)
