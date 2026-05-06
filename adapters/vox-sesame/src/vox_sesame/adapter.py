from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoProcessor

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

SESAME_SAMPLE_RATE = 24_000

DEFAULT_SPEAKER_ID = "0"

DEFAULT_VOICES: list[VoiceInfo] = [
    VoiceInfo(
        id=DEFAULT_SPEAKER_ID,
        name="Speaker 0",
        language="en",
        description="Default CSM speaker id",
    ),
]


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _get_csm_model_class() -> type[Any]:
    try:
        from transformers import CsmForConditionalGeneration

        return CsmForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Sesame CSM 1B requires Transformers >= 4.52.1, which provides "
            "CsmForConditionalGeneration. Spark should run this model through "
            "the native PyTorch/Transformers runtime, not ONNX."
        ) from exc


def _move_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


def _extract_audio_and_sample_rate(output: Any) -> tuple[np.ndarray, int]:
    sample_rate = getattr(output, "sample_rate", SESAME_SAMPLE_RATE)
    audio = getattr(output, "audio", output)

    if isinstance(audio, dict):
        audio = audio.get("audio", audio)

    if isinstance(audio, (list, tuple)):
        audio = audio[0] if audio else np.array([], dtype=np.float32)

    if hasattr(audio, "detach") and hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        audio = audio.detach().cpu().numpy()
    else:
        audio = np.asarray(audio)

    if audio.ndim > 1:
        audio = audio.reshape(-1)

    return audio.astype(np.float32, copy=False), sample_rate


class SesameTTSAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: AutoProcessor | None = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="sesame-tts-torch",
            type=ModelType.TTS,
            architectures=("sesame-tts-torch", "sesame", "csm"),
            default_sample_rate=SESAME_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = device
        dtype = _select_dtype(self._device)

        CsmForConditionalGeneration = _get_csm_model_class()

        logger.info("Loading Sesame CSM model: %s (device=%s, dtype=%s)", self._model_id, self._device, dtype)
        start = time.perf_counter()

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = CsmForConditionalGeneration.from_pretrained(self._model_id, torch_dtype=dtype)
        if hasattr(self._model, "to"):
            self._model = self._model.to(self._device)
        if hasattr(self._model, "eval"):
            self._model.eval()

        elapsed = time.perf_counter() - start
        logger.info("Sesame CSM model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Sesame CSM adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Sesame CSM model is not loaded — call load() first")

        if not text or not text.strip():
            return

        speaker_id = (voice or DEFAULT_SPEAKER_ID).strip() or DEFAULT_SPEAKER_ID

        if (reference_audio is None) ^ (reference_text is None):
            raise ValueError(
                "Sesame CSM requires both reference_audio and reference_text when providing conversational context"
            )

        if reference_audio is not None and reference_text is not None:
            conversation: list[dict[str, Any]] = [
                {
                    "role": speaker_id,
                    "content": [
                        {"type": "text", "text": reference_text},
                        {"type": "audio", "path": reference_audio},
                    ],
                },
                {
                    "role": speaker_id,
                    "content": [{"type": "text", "text": text}],
                },
            ]
        else:
            conversation = [
                {
                    "role": speaker_id,
                    "content": [{"type": "text", "text": text}],
                }
            ]

        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )
        inputs = _move_to_device(inputs, self._device)

        with torch.inference_mode():
            output = self._model.generate(**inputs, output_audio=True)

        audio, sample_rate = _extract_audio_and_sample_rate(output)
        chunk_size = sample_rate * 2
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield SynthesizeChunk(
                audio=chunk.tobytes(),
                sample_rate=sample_rate,
                is_final=False,
            )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=sample_rate,
            is_final=True,
        )

    def list_voices(self) -> list[VoiceInfo]:
        return list(DEFAULT_VOICES)

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 5_000_000_000
