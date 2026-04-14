from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

QWEN_TTS_SAMPLE_RATE = 24_000

SUPPORTED_LANGUAGES = (
    "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "it",
)

_VRAM_ESTIMATES: dict[str, int] = {
    "0.6b": 2_500_000_000,
    "1.7b": 4_000_000_000,
}


def _select_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if device in ("mps", "auto") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _estimate_vram(model_id: str) -> int:
    lower = model_id.lower()
    for size_key, vram in _VRAM_ESTIMATES.items():
        if size_key in lower:
            return vram
    return _VRAM_ESTIMATES["0.6b"]


class Qwen3TTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="qwen3-tts",
            type=ModelType.TTS,
            architectures=("qwen3-tts",),
            default_sample_rate=QWEN_TTS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=True,
            supported_languages=SUPPORTED_LANGUAGES,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = _select_device(device)
        dtype = _select_dtype(self._device)

        logger.info("Loading Qwen3-TTS model: %s (device=%s, dtype=%s)", self._model_id, self._device, dtype)
        start = time.perf_counter()

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=self._device if self._device != "cpu" else None,
        )
        if self._device == "cpu":
            self._model = self._model.to(self._device)

        elapsed = time.perf_counter() - start
        logger.info("Qwen3-TTS model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3-TTS adapter unloaded")

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
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Qwen3-TTS model is not loaded — call load() first")

        if not text or not text.strip():
            return

        prompt_parts = []
        if reference_audio is not None and reference_text:
            prompt_parts.append(f"[reference_text]{reference_text}[/reference_text]")
            prompt_parts.append("[reference_audio]")

        if voice:
            prompt_parts.append(f"[voice]{voice}[/voice]")

        prompt_parts.append(text)
        full_prompt = "\n".join(prompt_parts)

        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        if reference_audio is not None:
            inputs["reference_audio"] = torch.tensor(reference_audio).unsqueeze(0).to(self._model.device)

        with torch.inference_mode():
            output = self._model.generate(**inputs)

        if hasattr(output, "audio"):
            audio = output.audio[0].cpu().numpy().astype(np.float32)
        elif hasattr(output, "waveform"):
            audio = output.waveform[0].cpu().numpy().astype(np.float32)
        else:
            audio = output[0].cpu().numpy().astype(np.float32)

        chunk_size = QWEN_TTS_SAMPLE_RATE * 2
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield SynthesizeChunk(
                audio=chunk.tobytes(),
                sample_rate=QWEN_TTS_SAMPLE_RATE,
                is_final=False,
            )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=QWEN_TTS_SAMPLE_RATE,
            is_final=True,
        )

    def list_voices(self) -> list[VoiceInfo]:
        return []

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        model_id = kwargs.get("_source") or kwargs.get("model_id") or self._model_id
        return _estimate_vram(str(model_id))
