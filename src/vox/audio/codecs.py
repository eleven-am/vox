"""Audio encoding and decoding utilities.

Supports WAV, MP3, FLAC, OGG, and other formats via soundfile and pydub.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def decode_audio(
    data: bytes,
    format_hint: str | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Decode audio bytes to float32 samples and sample rate.

    Tries soundfile first for lossless/WAV/FLAC, falls back to pydub
    (which uses ffmpeg) for MP3, OGG, and other formats.
    """
    # Try soundfile first (fast, no subprocess)
    try:
        audio, sample_rate = sf.read(
            io.BytesIO(data),
            dtype="float32",
            format=format_hint,
        )
        return audio.astype(np.float32), sample_rate
    except (sf.SoundFileError, sf.SoundFileRuntimeError) as sf_err:
        logger.warning("soundfile failed: %s — falling back to pydub", sf_err)
        _sf_error = sf_err  # save before Python deletes sf_err at block exit

    # Fall back to pydub (uses ffmpeg under the hood)
    from pydub import AudioSegment

    buffer = io.BytesIO(data)
    try:
        if format_hint:
            segment = AudioSegment.from_file(buffer, format=format_hint)
        else:
            segment = AudioSegment.from_file(buffer)
    except Exception as pydub_err:
        raise RuntimeError(
            f"Failed to decode audio: soundfile error: {_sf_error}; pydub error: {pydub_err}"
        ) from pydub_err

    sample_rate = segment.frame_rate
    channels = segment.channels
    samples = np.array(segment.get_array_of_samples(), dtype=np.int16)

    # pydub interleaves channels; reshape if stereo+
    if channels > 1:
        samples = samples.reshape(-1, channels)

    audio = samples.astype(np.float32) / 32768.0
    return audio, sample_rate


def encode_wav(audio: NDArray[np.float32], sample_rate: int) -> bytes:
    """Encode float32 audio to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def encode_flac(audio: NDArray[np.float32], sample_rate: int) -> bytes:
    """Encode float32 audio to FLAC bytes."""
    buf = io.BytesIO()
    # FLAC requires integer samples; convert to 16-bit
    pcm16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    sf.write(buf, pcm16, samplerate=sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def encode_pcm(audio: NDArray[np.float32]) -> bytes:
    """Convert float32 audio to int16 little-endian PCM bytes."""
    pcm16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    return pcm16.tobytes()


def pcm16_to_float32(data: bytes) -> NDArray[np.float32]:
    """Convert int16 little-endian PCM bytes to float32 array."""
    pcm_array = np.frombuffer(data, dtype=np.int16)
    return pcm_array.astype(np.float32) / 32768.0


def to_mono(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert multi-channel audio to mono by averaging channels.

    Returns the input unchanged if already mono (1-D array).
    """
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1).astype(np.float32)
