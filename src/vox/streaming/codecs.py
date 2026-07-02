from __future__ import annotations

import numpy as np
import soxr
from numpy.typing import NDArray


def pcm16_to_float32(pcm_bytes: bytes) -> NDArray[np.float32]:
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: NDArray[np.float32]) -> bytes:
    clamped = np.clip(audio, -1.0, 1.0)
    return (clamped * 32767).astype(np.int16).tobytes()


def resample_audio(audio: NDArray[np.float32], source_rate: int, target_rate: int) -> NDArray[np.float32]:
    if source_rate == target_rate:
        return audio
    return soxr.resample(audio, source_rate, target_rate).astype(np.float32)


class StreamResampler:
    """Stateful resampler for a single continuous audio stream.

    Carries filter state across chunks so concatenating the outputs matches a
    single one-shot resample of the concatenated input — no per-chunk seams or
    sample-count drift. Recreates its filter if the source rate changes.
    """

    def __init__(self, target_rate: int) -> None:
        self._target_rate = int(target_rate)
        self._source_rate: int | None = None
        self._stream: soxr.ResampleStream | None = None

    def process(self, audio: NDArray[np.float32], source_rate: int) -> NDArray[np.float32]:
        source_rate = int(source_rate)
        if source_rate == self._target_rate:
            return audio.astype(np.float32, copy=False)
        if self._stream is None or source_rate != self._source_rate:
            self._source_rate = source_rate
            self._stream = soxr.ResampleStream(
                source_rate, self._target_rate, 1, dtype="float32", quality="HQ"
            )
        return self._stream.resample_chunk(audio.astype(np.float32, copy=False)).astype(np.float32)

    def flush(self) -> NDArray[np.float32]:
        """Emit the resampler's held filter tail and reset for a fresh stream."""
        if self._stream is None:
            return np.array([], dtype=np.float32)
        tail = self._stream.resample_chunk(
            np.array([], dtype=np.float32), last=True
        ).astype(np.float32)
        self._stream = None
        self._source_rate = None
        return tail
