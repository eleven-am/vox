from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

OPUS_SAMPLE_RATE = 48_000
OPUS_FRAME_MS = 20
OPUS_FRAME_SAMPLES = OPUS_SAMPLE_RATE * OPUS_FRAME_MS // 1000


class OpusStreamDecoder:

    def __init__(self, sample_rate: int = OPUS_SAMPLE_RATE, channels: int = 1) -> None:
        import opuslib

        self.sample_rate = sample_rate
        self.channels = channels
        self.decoder = opuslib.Decoder(sample_rate, channels)
        self.frame_size = sample_rate * OPUS_FRAME_MS // 1000
        self._pending: bytes = b""

    def decode_frame(self, opus_data: bytes) -> NDArray[np.float32]:
        pcm = self.decoder.decode(opus_data, self.frame_size)
        samples = np.frombuffer(pcm, dtype=np.int16)

        if self.channels == 2:
            samples = self._stereo_to_mono(samples)

        return samples.astype(np.float32) / 32768.0

    def _stereo_to_mono(self, samples: np.ndarray) -> np.ndarray:
        left = samples[0::2]
        right = samples[1::2]
        return ((left.astype(np.float32) + right.astype(np.float32)) / 2.0).astype(np.int16)

    def reset(self) -> None:
        import opuslib

        self.decoder = opuslib.Decoder(self.sample_rate, self.channels)
        self._pending = b""

    def flush(self) -> NDArray[np.float32]:
        return np.array([], dtype=np.float32)


class OpusStreamEncoder:

    def __init__(self, source_rate: int = 24_000, target_rate: int = OPUS_SAMPLE_RATE, channels: int = 1) -> None:
        import opuslib

        self.source_rate = source_rate
        self.target_rate = target_rate
        self.channels = channels
        self._frame_samples = target_rate * OPUS_FRAME_MS // 1000
        self._buffer = np.array([], dtype=np.int16)
        self._encoder = opuslib.Encoder(target_rate, channels, "audio")

    def encode(self, pcm16: bytes) -> list[bytes]:
        from vox.streaming.codecs import resample_audio

        samples = np.frombuffer(pcm16, dtype=np.int16)

        if self.source_rate != self.target_rate:
            float_audio = samples.astype(np.float32) / 32768.0
            resampled = resample_audio(float_audio, self.source_rate, self.target_rate)
            samples = (resampled * 32767).astype(np.int16)

        self._buffer = np.concatenate([self._buffer, samples])

        frames: list[bytes] = []
        while len(self._buffer) >= self._frame_samples:
            frame_data = self._buffer[:self._frame_samples]
            self._buffer = self._buffer[self._frame_samples:]
            encoded = self._encoder.encode(frame_data.tobytes(), self._frame_samples)
            frames.append(encoded)

        return frames

    def flush(self) -> list[bytes]:
        if len(self._buffer) == 0:
            return []

        padded = np.zeros(self._frame_samples, dtype=np.int16)
        padded[:len(self._buffer)] = self._buffer
        self._buffer = np.array([], dtype=np.int16)

        encoded = self._encoder.encode(padded.tobytes(), self._frame_samples)
        return [encoded]

    def close(self) -> None:
        self._buffer = np.array([], dtype=np.int16)
