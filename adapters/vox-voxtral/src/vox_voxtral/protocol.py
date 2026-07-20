from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

VOXTRAL_TTS_SAMPLE_RATE = 24_000

OP_SYNTHESIZE = "synthesize"


@dataclass
class SynthesizeRequest:
    text: str
    voice: str

    def payload(self) -> dict[str, Any]:
        return {"op": OP_SYNTHESIZE, "text": self.text, "voice": self.voice}

    @classmethod
    def decode(cls, payload: dict[str, Any]) -> SynthesizeRequest:
        return cls(text=str(payload.get("text", "")), voice=str(payload.get("voice", "")))


@dataclass
class SynthesizeResponse:
    sample_rate: int
    audio_b64: str

    def audio_bytes(self) -> bytes:
        return base64.b64decode(self.audio_b64)

    def payload(self) -> dict[str, Any]:
        return {"sample_rate": self.sample_rate, "audio_b64": self.audio_b64}

    @classmethod
    def from_audio(cls, audio: bytes, sample_rate: int) -> SynthesizeResponse:
        return cls(sample_rate=sample_rate, audio_b64=base64.b64encode(audio).decode("ascii"))

    @classmethod
    def decode(cls, payload: dict[str, Any]) -> SynthesizeResponse:
        return cls(
            sample_rate=int(payload.get("sample_rate", VOXTRAL_TTS_SAMPLE_RATE)),
            audio_b64=payload["audio_b64"],
        )


def extract_audio_chunk(audio_chunk: Any, chunk_idx: int) -> NDArray[np.float32]:
    if isinstance(audio_chunk, list):
        if not audio_chunk:
            return np.asarray([], dtype=np.float32)
        audio_chunk = audio_chunk[chunk_idx] if chunk_idx < len(audio_chunk) else audio_chunk[-1]

    if hasattr(audio_chunk, "detach"):
        audio_chunk = audio_chunk.float().detach().cpu().numpy()
        return np.asarray(audio_chunk, dtype=np.float32)

    if isinstance(audio_chunk, np.ndarray):
        return audio_chunk.astype(np.float32, copy=False)

    return np.asarray(audio_chunk, dtype=np.float32)


def accumulate_chunk(
    audio_array: NDArray[np.float32],
    accumulated_sample: int,
    finished: bool,
) -> tuple[NDArray[np.float32], int]:
    if finished and accumulated_sample and len(audio_array) > accumulated_sample:
        audio_array = audio_array[accumulated_sample:]
    accumulated_sample += len(audio_array)
    return audio_array, accumulated_sample
