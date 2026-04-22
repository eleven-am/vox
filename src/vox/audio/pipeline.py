"""High-level audio pipeline combining decoding, resampling, and encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vox.audio.codecs import (
    decode_audio,
    encode_flac,
    encode_mp3,
    encode_opus,
    encode_pcm,
    encode_wav,
    to_mono,
)
from vox.audio.resampler import resample

_CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "pcm": "audio/L16",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
}

STT_CHUNK_DURATION_MS = 5 * 60 * 1000


@dataclass(frozen=True)
class AudioChunk:
    """A bounded slice of decoded STT-ready audio with its position in the original timeline."""

    data: NDArray[np.float32]
    sample_rate: int
    duration_ms: int
    offset_ms: int


def prepare_for_stt(
    data: bytes,
    target_rate: int = 16000,
    format_hint: str | None = None,
) -> NDArray[np.float32]:
    """Decode raw audio bytes into a mono float32 array ready for STT.

    Pipeline: decode -> mono -> resample -> normalize to [-1, 1].
    """
    audio, source_rate = decode_audio(data, format_hint=format_hint)

    audio = to_mono(audio)
    audio = resample(audio, source_rate, target_rate)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


def chunk_audio(
    audio: NDArray[np.float32],
    sample_rate: int,
    chunk_duration_ms: int = STT_CHUNK_DURATION_MS,
) -> list[AudioChunk]:
    """Split a float32 mono waveform into fixed-duration chunks with preserved offsets.

    Audio shorter than ``chunk_duration_ms`` is returned as a single chunk at offset 0.
    """
    if chunk_duration_ms <= 0:
        raise ValueError(f"chunk_duration_ms must be positive, got {chunk_duration_ms}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    total_samples = int(audio.shape[0])
    total_duration_ms = int(total_samples / sample_rate * 1000)

    if total_duration_ms <= chunk_duration_ms:
        return [
            AudioChunk(
                data=audio,
                sample_rate=sample_rate,
                duration_ms=total_duration_ms,
                offset_ms=0,
            )
        ]

    chunk_samples = max(1, int(chunk_duration_ms * sample_rate / 1000))
    chunks: list[AudioChunk] = []
    offset_samples = 0
    while offset_samples < total_samples:
        end_samples = min(offset_samples + chunk_samples, total_samples)
        segment = audio[offset_samples:end_samples]
        segment_duration_ms = int(segment.shape[0] / sample_rate * 1000)
        offset_ms = int(offset_samples / sample_rate * 1000)
        chunks.append(
            AudioChunk(
                data=segment,
                sample_rate=sample_rate,
                duration_ms=segment_duration_ms,
                offset_ms=offset_ms,
            )
        )
        offset_samples = end_samples

    return chunks


def prepare_for_stt_chunks(
    data: bytes,
    target_rate: int = 16000,
    format_hint: str | None = None,
    chunk_duration_ms: int = STT_CHUNK_DURATION_MS,
) -> list[AudioChunk]:
    """Decode bytes and return bounded chunks ready for STT.

    Adapters and model backends have per-call memory limits. Callers that expect to
    transcribe full-length uploads (HTTP, one-shot gRPC) should use this and merge
    the per-chunk results. Streaming paths that already receive bounded buffers
    should keep using :func:`prepare_for_stt`.
    """
    audio = prepare_for_stt(data, target_rate=target_rate, format_hint=format_hint)
    return chunk_audio(audio, sample_rate=target_rate, chunk_duration_ms=chunk_duration_ms)


def prepare_for_output(
    audio: NDArray[np.float32],
    sample_rate: int,
    output_format: str,
) -> tuple[bytes, str]:
    """Encode float32 audio to the requested format.

    Returns (encoded_bytes, mime_content_type).
    Supported formats: wav, flac, pcm, mp3, opus.
    """
    fmt = output_format.lower()

    if fmt == "wav":
        return encode_wav(audio, sample_rate), get_content_type(fmt)

    if fmt == "flac":
        return encode_flac(audio, sample_rate), get_content_type(fmt)

    if fmt == "pcm":
        return encode_pcm(audio), get_content_type(fmt)

    if fmt == "mp3":
        return encode_mp3(audio, sample_rate), get_content_type(fmt)

    if fmt == "opus":
        return encode_opus(audio, sample_rate), get_content_type(fmt)

    raise ValueError(f"Unsupported output format: {fmt!r}")


def get_content_type(format: str) -> str:
    """Map a format string to its MIME content type."""
    fmt = format.lower()
    try:
        return _CONTENT_TYPES[fmt]
    except KeyError:
        raise ValueError(f"Unknown format: {fmt!r}") from None
