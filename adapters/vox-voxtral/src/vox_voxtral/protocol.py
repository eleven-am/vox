from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

VOXTRAL_TTS_SAMPLE_RATE = 24_000

OP_SYNTHESIZE = "synthesize"
OP_SHUTDOWN = "shutdown"

STATUS_READY = "ready"
STATUS_OK = "ok"
STATUS_ERROR = "error"


@dataclass
class SynthesizeRequest:
    op: str
    text: str
    voice: str

    def encode(self) -> str:
        return json.dumps({"op": self.op, "text": self.text, "voice": self.voice})

    @classmethod
    def make(cls, text: str, voice: str) -> "SynthesizeRequest":
        return cls(op=OP_SYNTHESIZE, text=text, voice=voice)


@dataclass
class ShutdownRequest:
    op: str

    def encode(self) -> str:
        return json.dumps({"op": self.op})

    @classmethod
    def make(cls) -> "ShutdownRequest":
        return cls(op=OP_SHUTDOWN)


@dataclass
class ReadyResponse:
    status: str

    @classmethod
    def decode(cls, payload: dict[str, Any]) -> "ReadyResponse":
        return cls(status=payload["status"])


@dataclass
class OkResponse:
    status: str
    sample_rate: int
    audio_b64: str

    def audio_bytes(self) -> bytes:
        return base64.b64decode(self.audio_b64)

    @classmethod
    def decode(cls, payload: dict[str, Any]) -> "OkResponse":
        return cls(
            status=payload["status"],
            sample_rate=int(payload.get("sample_rate", 24_000)),
            audio_b64=payload["audio_b64"],
        )

    @classmethod
    def encode_audio(cls, audio_bytes: bytes, sample_rate: int) -> str:
        return json.dumps(
            {
                "status": STATUS_OK,
                "sample_rate": sample_rate,
                "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            }
        )


@dataclass
class ErrorResponse:
    status: str
    error: str

    @classmethod
    def decode(cls, payload: dict[str, Any]) -> "ErrorResponse":
        return cls(status=payload["status"], error=payload.get("error", "unknown error"))

    @classmethod
    def encode_error(cls, message: str) -> str:
        return json.dumps({"status": STATUS_ERROR, "error": message})


def decode_response(line: str) -> dict[str, Any]:
    return json.loads(line)


def is_ready(payload: dict[str, Any]) -> bool:
    return payload.get("status") == STATUS_READY


def is_ok(payload: dict[str, Any]) -> bool:
    return payload.get("status") == STATUS_OK


def is_error(payload: dict[str, Any]) -> bool:
    return payload.get("status") == STATUS_ERROR


import numpy as np
from numpy.typing import NDArray


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
