from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import SpeechT5ForSpeechToText, SpeechT5Processor

from vox.core.adapter import STTAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    TranscribeResult,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

SPEECHT5_SAMPLE_RATE = 16_000


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


class SpeechT5STTAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model: SpeechT5ForSpeechToText | None = None
        self._processor: SpeechT5Processor | None = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="speecht5-stt",
            type=ModelType.STT,
            architectures=("speecht5", "speecht5-asr"),
            default_sample_rate=SPEECHT5_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_word_timestamps=False,
            supports_language_detection=False,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = _select_device(device)

        logger.info("Loading SpeechT5 ASR model: %s (device=%s)", self._model_id, self._device)
        start = time.perf_counter()

        self._processor = SpeechT5Processor.from_pretrained(self._model_id)
        self._model = SpeechT5ForSpeechToText.from_pretrained(self._model_id).to(self._device)

        elapsed = time.perf_counter() - start
        logger.info("SpeechT5 ASR model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SpeechT5 STT adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

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
            raise RuntimeError("SpeechT5 ASR model is not loaded — call load() first")

        if len(audio) == 0:
            return TranscribeResult(text="", language="en", duration_ms=0, model=self._model_id)

        audio_duration_ms = int(len(audio) / SPEECHT5_SAMPLE_RATE * 1000)

        inputs = self._processor(
            audio=audio,
            sampling_rate=SPEECHT5_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            predicted_ids = self._model.generate(**inputs, max_length=512)

        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        segments = ()
        if text:
            segments = (
                TranscriptSegment(
                    text=text,
                    start_ms=0,
                    end_ms=audio_duration_ms,
                    language="en",
                ),
            )

        if not text:
            logger.warning("Empty transcription for %dms audio", audio_duration_ms)
        else:
            logger.info("Transcribed %dms audio: %s", audio_duration_ms, text[:80])

        return TranscribeResult(
            text=text,
            segments=segments,
            language="en",
            duration_ms=audio_duration_ms,
            model=self._model_id,
        )

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 320_000_000
