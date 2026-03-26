"""Audio resampling using soxr."""

from __future__ import annotations

import numpy as np
import soxr
from numpy.typing import NDArray


def resample(
    audio: NDArray[np.float32],
    source_rate: int,
    target_rate: int,
) -> NDArray[np.float32]:
    """Resample audio from source_rate to target_rate.

    Returns the audio unchanged if rates already match.
    """
    if source_rate == target_rate:
        return audio
    return soxr.resample(audio, source_rate, target_rate, quality="HQ").astype(
        np.float32
    )
