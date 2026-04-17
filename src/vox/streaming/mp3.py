from __future__ import annotations

import numpy as np


class Mp3StreamEncoder:
    """Streaming MP3 encoder.

    Accepts PCM16 bytes chunk-by-chunk, emits raw MP3 frames on each call
    (lameenc buffers internally until it has a full frame). On `flush()`,
    drains any remaining samples.
    """

    def __init__(
        self,
        source_rate: int = 24_000,
        *,
        bitrate: int = 128,
        channels: int = 1,
        quality: int = 2,
    ) -> None:
        import lameenc

        self.source_rate = source_rate
        self.channels = channels
        self._encoder = lameenc.Encoder()
        self._encoder.set_bit_rate(bitrate)
        self._encoder.set_in_sample_rate(source_rate)
        self._encoder.set_channels(channels)
        self._encoder.set_quality(quality)
        self._closed = False

    def encode(self, pcm16: bytes) -> bytes:
        if self._closed:
            return b""
        if not pcm16:
            return b""

        return self._encoder.encode(pcm16) or b""

    def flush(self) -> bytes:
        if self._closed:
            return b""
        self._closed = True
        return self._encoder.flush() or b""

    def close(self) -> None:
        self._closed = True
