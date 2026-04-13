from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoProcessor, VoxtralTTSModel

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

VOXTRAL_TTS_SAMPLE_RATE = 24_000

PRESET_VOICES: list[VoiceInfo] = [
    VoiceInfo(id="jessica", name="Jessica", language="en", gender="female"),
    VoiceInfo(id="emma", name="Emma", language="en", gender="female"),
    VoiceInfo(id="allison", name="Allison", language="en", gender="female"),
    VoiceInfo(id="james", name="James", language="en", gender="male"),
    VoiceInfo(id="john", name="John", language="en", gender="male"),
    VoiceInfo(id="sophia", name="Sophia", language="fr", gender="female"),
    VoiceInfo(id="pierre", name="Pierre", language="fr", gender="male"),
    VoiceInfo(id="elena", name="Elena", language="es", gender="female"),
    VoiceInfo(id="carlos", name="Carlos", language="es", gender="male"),
    VoiceInfo(id="anna", name="Anna", language="de", gender="female"),
    VoiceInfo(id="marco", name="Marco", language="it", gender="male"),
    VoiceInfo(id="lucia", name="Lucia", language="pt", gender="female"),
    VoiceInfo(id="priya", name="Priya", language="hi", gender="female"),
    VoiceInfo(id="lars", name="Lars", language="nl", gender="male"),
    VoiceInfo(id="fatima", name="Fatima", language="ar", gender="female"),
]

SUPPORTED_LANGUAGES = ("en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi")


def _select_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "auto"):
        if torch.cuda.is_available():
            return "cuda"
    if device in ("mps", "auto"):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


class VoxtralTTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._model: VoxtralTTSModel | None = None
        self._processor: AutoProcessor | None = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="voxtral-tts",
            type=ModelType.TTS,
            architectures=("voxtral-tts",),
            default_sample_rate=VOXTRAL_TTS_SAMPLE_RATE,
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

        logger.info("Loading Voxtral TTS model: %s (device=%s, dtype=%s)", self._model_id, self._device, dtype)
        start = time.perf_counter()

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = VoxtralTTSModel.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            device_map=self._device if self._device != "cpu" else None,
        )
        if self._device == "cpu":
            self._model = self._model.to(self._device)

        elapsed = time.perf_counter() - start
        logger.info("Voxtral TTS model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Voxtral TTS adapter unloaded")

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
            raise RuntimeError("Voxtral TTS model is not loaded — call load() first")

        if not text or not text.strip():
            return

        voice_id = voice or "jessica"
        speed = max(0.5, min(speed, 2.0))

        inputs = self._processor(
            text=text,
            voice=voice_id,
            language=language,
            reference_audio=reference_audio,
            reference_text=reference_text,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(**inputs, speed=speed)

        audio = output.audio[0].cpu().numpy().astype(np.float32)
        sample_rate = output.sample_rate if hasattr(output, "sample_rate") else VOXTRAL_TTS_SAMPLE_RATE

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
        return list(PRESET_VOICES)

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 16_000_000_000
