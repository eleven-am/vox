from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class AudioRingBuffer:
    __slots__ = ("_buffer", "_write_pos", "_length")

    def __init__(self, max_samples: int) -> None:
        self._buffer = np.zeros(max_samples, dtype=np.float32)
        self._write_pos = 0
        self._length = 0

    def append(self, audio: NDArray[np.float32]) -> None:
        n = len(audio)
        if n == 0:
            return

        if n >= len(self._buffer):
            self._buffer[:] = audio[-len(self._buffer):]
            self._write_pos = 0
            self._length = len(self._buffer)
            return

        end_pos = self._write_pos + n
        if end_pos <= len(self._buffer):
            self._buffer[self._write_pos:end_pos] = audio
        else:
            first_part = len(self._buffer) - self._write_pos
            self._buffer[self._write_pos:] = audio[:first_part]
            self._buffer[:n - first_part] = audio[first_part:]

        self._write_pos = end_pos % len(self._buffer)
        self._length = min(self._length + n, len(self._buffer))

    def get_last_n(self, n: int) -> NDArray[np.float32]:
        n = min(n, self._length)
        if n == 0:
            return np.array([], dtype=np.float32)

        end_pos = self._write_pos
        start_pos = (end_pos - n) % len(self._buffer)

        if start_pos < end_pos:
            return self._buffer[start_pos:end_pos].copy()
        return np.concatenate([self._buffer[start_pos:], self._buffer[:end_pos]])

    def get_all(self) -> NDArray[np.float32]:
        if self._length == 0:
            return np.array([], dtype=np.float32)
        if self._length < len(self._buffer):
            return self._buffer[:self._length].copy()
        start_pos = self._write_pos
        if start_pos == 0:
            return self._buffer.copy()
        return np.concatenate([self._buffer[start_pos:], self._buffer[:start_pos]])

    def get_slice(self, start_sample: int, end_sample: int) -> NDArray[np.float32]:
        start_sample = max(0, min(start_sample, self._length))
        end_sample = max(0, min(end_sample, self._length))
        if start_sample >= end_sample:
            return np.array([], dtype=np.float32)

        oldest_pos = (self._write_pos - self._length) % len(self._buffer)
        actual_start = (oldest_pos + start_sample) % len(self._buffer)
        actual_end = (oldest_pos + end_sample) % len(self._buffer)

        if actual_start < actual_end:
            return self._buffer[actual_start:actual_end].copy()
        return np.concatenate([self._buffer[actual_start:], self._buffer[:actual_end]])

    def __len__(self) -> int:
        return self._length

    def clear(self) -> None:
        self._write_pos = 0
        self._length = 0
