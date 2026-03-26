"""High-level audio pipeline combining decoding, resampling, and encoding."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vox.audio.codecs import (
    decode_audio,
    encode_flac,
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
}


class AudioPipeline:
    """Stateless audio pipeline for STT input preparation and output encoding."""

    def prepare_for_stt(
        self,
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

        # Normalize to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        return audio.astype(np.float32)

    def prepare_for_output(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        output_format: str,
    ) -> tuple[bytes, str]:
        """Encode float32 audio to the requested format.

        Returns:
            Tuple of (encoded_bytes, mime_content_type).

        Supported formats: wav, flac, pcm.
        MP3 raises NotImplementedError (requires optional dependency).
        """
        fmt = output_format.lower()

        if fmt == "wav":
            return encode_wav(audio, sample_rate), self.get_content_type(fmt)

        if fmt == "flac":
            return encode_flac(audio, sample_rate), self.get_content_type(fmt)

        if fmt == "pcm":
            return encode_pcm(audio), self.get_content_type(fmt)

        if fmt == "mp3":
            raise NotImplementedError(
                "MP3 encoding requires an optional dependency (e.g. lameenc). "
                "Install it separately and use a streaming MP3 encoder."
            )

        raise ValueError(f"Unsupported output format: {fmt!r}")

    @staticmethod
    def get_content_type(format: str) -> str:
        """Map a format string to its MIME content type."""
        fmt = format.lower()
        try:
            return _CONTENT_TYPES[fmt]
        except KeyError:
            raise ValueError(f"Unknown format: {fmt!r}") from None
