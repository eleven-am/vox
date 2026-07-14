from __future__ import annotations

import hashlib
import math
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioMetrics:
    bytes: int
    sha256: str
    duration_s: float | None
    sample_rate: int | None
    channels: int | None
    sample_width: int | None
    peak: float | None
    rms: float | None
    silent: bool


def pcm_stats(frames: bytes, *, sample_width: int) -> tuple[float | None, float | None]:
    if not frames or sample_width not in {1, 2, 4}:
        return None, None

    if sample_width == 1:
        values = [byte - 128 for byte in frames]
        normalizer = 128.0
    else:
        values = [
            int.from_bytes(frames[index:index + sample_width], byteorder="little", signed=True)
            for index in range(0, len(frames) - sample_width + 1, sample_width)
        ]
        normalizer = float(2 ** (8 * sample_width - 1))

    if not values:
        return None, None
    peak = max(abs(value) for value in values) / normalizer
    rms = math.sqrt(sum(value * value for value in values) / len(values)) / normalizer
    return peak, rms


def inspect_audio(path: Path) -> AudioMetrics | None:
    if not path.exists():
        return None

    data = path.read_bytes()
    duration_s: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sample_width: int | None = None
    peak: float | None = None
    rms: float | None = None
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            duration_s = frame_count / sample_rate if sample_rate else None
            peak, rms = pcm_stats(wav.readframes(frame_count), sample_width=sample_width)
    except (EOFError, wave.Error):
        pass

    return AudioMetrics(
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        peak=peak,
        rms=rms,
        silent=bool(rms is not None and rms < 0.0001),
    )
