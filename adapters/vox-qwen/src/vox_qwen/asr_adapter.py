from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from vox.core.adapter import STTAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    TranscribeResult,
    TranscriptSegment,
    WordTimestamp,
)

logger = logging.getLogger(__name__)

QWEN_ASR_SAMPLE_RATE = 16_000

SUPPORTED_LANGUAGES = (
    "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "it", "nl",
    "ru", "ar", "hi", "vi", "th", "id", "ms", "pl", "tr", "uk",
    "cs", "ro", "hu", "el", "bg", "sv", "da", "fi", "no", "he",
)

_VRAM_ESTIMATES: dict[str, int] = {
    "0.6b": 1_500_000_000,
    "1.7b": 4_000_000_000,
}


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


def _estimate_vram(model_id: str) -> int:
    lower = model_id.lower()
    for size_key, vram in _VRAM_ESTIMATES.items():
        if size_key in lower:
            return vram
    return _VRAM_ESTIMATES["0.6b"]


class Qwen3ASRAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: AutoProcessor | None = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="qwen3-asr",
            type=ModelType.STT,
            architectures=("qwen3-asr",),
            default_sample_rate=QWEN_ASR_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_word_timestamps=True,
            supports_language_detection=True,
            supported_languages=SUPPORTED_LANGUAGES,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = _select_device(device)
        dtype = _select_dtype(self._device)

        logger.info("Loading Qwen3-ASR model: %s (device=%s, dtype=%s)", self._model_id, self._device, dtype)
        start = time.perf_counter()

        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=self._device if self._device != "cpu" else None,
        )
        if self._device == "cpu":
            self._model = self._model.to(self._device)

        elapsed = time.perf_counter() - start
        logger.info("Qwen3-ASR model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3-ASR adapter unloaded")

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
            raise RuntimeError("Qwen3-ASR model is not loaded — call load() first")

        if len(audio) == 0:
            return TranscribeResult(text="", language=language, duration_ms=0, model=self._model_id)

        audio_duration_ms = int(len(audio) / QWEN_ASR_SAMPLE_RATE * 1000)

        prompt = "<|startoftranscript|>"
        if language:
            prompt += f"<|{language}|>"
        if word_timestamps:
            prompt += "<|timestamps|>"
        prompt += "<|transcribe|>"

        inputs = self._processor(
            audio=audio,
            sampling_rate=QWEN_ASR_SAMPLE_RATE,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        raw_text = self._processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        words = ()
        if word_timestamps:
            words = self._parse_timestamps(raw_text)

        segments = ()
        if text:
            segments = (
                TranscriptSegment(
                    text=text,
                    start_ms=0,
                    end_ms=audio_duration_ms,
                    words=words,
                    language=language,
                ),
            )

        if not text:
            logger.warning("Empty transcription for %dms audio", audio_duration_ms)
        else:
            logger.info("Transcribed %dms audio: %s", audio_duration_ms, text[:80])

        return TranscribeResult(
            text=text,
            segments=segments,
            language=language,
            duration_ms=audio_duration_ms,
            model=self._model_id,
        )

    def _parse_timestamps(self, raw_text: str) -> tuple[WordTimestamp, ...]:
        import re

        pattern = r"<\|(\d+\.?\d*)\|>([^<]+)"
        matches = re.findall(pattern, raw_text)

        words: list[WordTimestamp] = []
        for i, (ts_str, word_text) in enumerate(matches):
            word_text = word_text.strip()
            if not word_text:
                continue
            start_ms = int(float(ts_str) * 1000)
            end_ms = int(float(matches[i + 1][0]) * 1000) if i + 1 < len(matches) else start_ms + 500
            words.append(WordTimestamp(
                word=word_text,
                start_ms=start_ms,
                end_ms=end_ms,
            ))

        return tuple(words)

    def detect_language(self, audio: NDArray[np.float32]) -> str:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Qwen3-ASR model is not loaded — call load() first")

        snippet = audio[:QWEN_ASR_SAMPLE_RATE * 10]
        inputs = self._processor(
            audio=snippet,
            sampling_rate=QWEN_ASR_SAMPLE_RATE,
            text="<|startoftranscript|><|detect_language|>",
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=10)

        raw = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip().lower()
        return raw[:2] if len(raw) >= 2 else "en"

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return _estimate_vram(self._model_id)
