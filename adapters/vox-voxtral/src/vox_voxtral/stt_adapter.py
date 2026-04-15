from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoProcessor, VoxtralForConditionalGeneration

from vox.core.adapter import STTAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    TranscribeResult,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

VOXTRAL_SAMPLE_RATE = 16_000

SUPPORTED_LANGUAGES = (
    "en", "es", "fr", "pt", "hi", "de", "nl", "it",
    "ru", "zh", "ja", "ko", "ar",
)

_VRAM_ESTIMATES: dict[str, int] = {
    "3b": 9_500_000_000,
    "4b": 12_000_000_000,
    "24b": 48_000_000_000,
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
    return _VRAM_ESTIMATES["3b"]


class VoxtralSTTAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model: VoxtralForConditionalGeneration | None = None
        self._processor: AutoProcessor | None = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="voxtral-stt-torch",
            type=ModelType.STT,
            architectures=("voxtral-stt-torch", "voxtral", "voxtral-mini", "voxtral-small", "voxtral-realtime"),
            default_sample_rate=VOXTRAL_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_word_timestamps=False,
            supports_language_detection=False,
            supported_languages=SUPPORTED_LANGUAGES,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = _select_device(device)
        dtype = _select_dtype(self._device)

        logger.info("Loading Voxtral STT model: %s (device=%s, dtype=%s)", self._model_id, self._device, dtype)
        start = time.perf_counter()

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = VoxtralForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            device_map=self._device if self._device != "cpu" else None,
        )
        if self._device == "cpu":
            self._model = self._model.to(self._device)

        elapsed = time.perf_counter() - start
        logger.info("Voxtral STT model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Voxtral STT adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _prepare_transcription_inputs(
        self,
        audio: NDArray[np.float32],
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        if self._processor is None or self._model is None:
            raise RuntimeError("Voxtral STT model is not loaded — call load() first")

        if language is None:
            # The current Transformers Voxtral processor requires an explicit language for
            # transcription requests, even though the underlying Mistral request schema treats it
            # as optional. Default to English instead of crashing on a missing value.
            language = "en"
            logger.warning(
                "No language provided for Voxtral STT; defaulting to 'en' because the "
                "Transformers Voxtral processor currently requires an explicit language."
            )

        inputs = self._processor.apply_transcription_request(
            audio=[audio],
            model_id=self._model_id,
            language=language,
            sampling_rate=VOXTRAL_SAMPLE_RATE,
            format=["wav"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {
            k: v.to(self._model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> TranscribeResult:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Voxtral STT model is not loaded — call load() first")

        if len(audio) == 0:
            return TranscribeResult(text="", language=language, duration_ms=0, model=self._model_id)

        audio_duration_ms = int(len(audio) / VOXTRAL_SAMPLE_RATE * 1000)

        inputs = self._prepare_transcription_inputs(audio, language=language)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        prompt_len = inputs["input_ids"].shape[1]
        text = self._processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

        segments = ()
        if text:
            segments = (
                TranscriptSegment(
                    text=text,
                    start_ms=0,
                    end_ms=audio_duration_ms,
                    language=language or "en",
                ),
            )

        if not text:
            logger.warning("Empty transcription for %dms audio", audio_duration_ms)
        else:
            logger.info("Transcribed %dms audio: %s", audio_duration_ms, text[:80])

        return TranscribeResult(
            text=text,
            segments=segments,
            language=language or "en",
            duration_ms=audio_duration_ms,
            model=self._model_id,
        )

    def detect_language(self, audio: NDArray[np.float32]) -> str:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Voxtral STT model is not loaded — call load() first")
        raise NotImplementedError(
            "Standalone language detection is not implemented for the Hugging Face Voxtral "
            "transcription path; omit `language` during transcription to let the model auto-detect it."
        )

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        model_id = kwargs.get("_source") or kwargs.get("model_id") or self._model_id
        return _estimate_vram(str(model_id))
