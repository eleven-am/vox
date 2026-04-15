from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import suppress
from importlib.util import find_spec
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray

from vox.core.adapter import STTAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    TranscribeResult,
    TranscriptSegment,
    WordTimestamp,
)
from vox_qwen.runtime import ensure_runtime

logger = logging.getLogger(__name__)

QWEN_ASR_SAMPLE_RATE = 16_000
QWEN_ASR_FORCE_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"

SUPPORTED_LANGUAGES = (
    "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "it", "nl",
    "ru", "ar", "hi", "vi", "th", "id", "ms", "pl", "tr", "uk",
    "cs", "ro", "hu", "el", "bg", "sv", "da", "fi", "no", "he",
)

_QWEN_LANGUAGE_LABELS = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "pl": "Polish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "bg": "Bulgarian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "he": "Hebrew",
}
_QWEN_LANGUAGE_CODES = {label.lower(): code for code, label in _QWEN_LANGUAGE_LABELS.items()}

_VRAM_ESTIMATES: dict[str, int] = {
    "0.6b": 1_500_000_000,
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


def _select_device_map(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return "cpu"


def _estimate_vram(model_id: str) -> int:
    lower = model_id.lower()
    for size_key, vram in _VRAM_ESTIMATES.items():
        if size_key in lower:
            return vram
    return _VRAM_ESTIMATES["0.6b"]


def _supports_flash_attention() -> bool:
    return find_spec("flash_attn") is not None


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    key = language.strip().lower()
    if not key:
        return None
    if key in _QWEN_LANGUAGE_LABELS:
        return _QWEN_LANGUAGE_LABELS[key]
    return language


def _normalize_language_code(language: str | None) -> str | None:
    if language is None:
        return None
    key = language.strip().lower()
    if not key:
        return None
    if key in _QWEN_LANGUAGE_CODES:
        return _QWEN_LANGUAGE_CODES[key]
    if len(key) >= 2:
        return key[:2]
    return None


def _load_qwen_asr_model() -> Any:
    ensure_runtime(
        "qwen-asr",
        "qwen-asr",
        "qwen_asr",
        purge_modules=("accelerate", "transformers", "tokenizers", "qwen_asr"),
    )
    try:
        from qwen_asr import Qwen3ASRModel
        return Qwen3ASRModel
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "Qwen3-ASR requires the qwen-asr runtime package; install qwen-asr in the image"
        ) from exc


def _load_qwen_forced_aligner() -> Any:
    ensure_runtime(
        "qwen-asr",
        "qwen-asr",
        "qwen_asr",
        purge_modules=("accelerate", "transformers", "tokenizers", "qwen_asr"),
    )
    try:
        from qwen_asr import Qwen3ForcedAligner
        return Qwen3ForcedAligner
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "Qwen3-ASR word timestamps require the qwen-asr forced aligner runtime"
        ) from exc


class Qwen3ASRAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._aligner: Any = None
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
        device_map = _select_device_map(self._device)

        logger.info("Loading Qwen3-ASR model: %s (device=%s, dtype=%s)", self._model_id, self._device, dtype)
        start = time.perf_counter()

        Qwen3ASRModel = _load_qwen_asr_model()
        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": dtype,
        }
        if self._device == "cuda" and _supports_flash_attention():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = Qwen3ASRModel.from_pretrained(self._model_id, **model_kwargs)
        self._processor = getattr(self._model, "processor", None)

        elapsed = time.perf_counter() - start
        logger.info("Qwen3-ASR model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._aligner = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3-ASR adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _audio_to_temp_wav(self, audio: NDArray[np.float32]) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        sf.write(path, audio, QWEN_ASR_SAMPLE_RATE)
        return path

    def _extract_words(self, time_stamps: Any) -> tuple[WordTimestamp, ...]:
        words: list[WordTimestamp] = []
        for item in time_stamps or []:
            word = getattr(item, "text", None) or getattr(item, "word", None)
            if not word:
                continue
            start_time = getattr(item, "start_time", None)
            end_time = getattr(item, "end_time", None)
            if start_time is None or end_time is None:
                continue
            words.append(
                WordTimestamp(
                    word=str(word),
                    start_ms=int(float(start_time) * 1000),
                    end_ms=int(float(end_time) * 1000),
                    confidence=getattr(item, "confidence", None),
                )
            )
        return tuple(words)

    def _ensure_aligner(self) -> Any:
        if self._aligner is not None:
            return self._aligner

        Qwen3ForcedAligner = _load_qwen_forced_aligner()
        dtype = _select_dtype(self._device)
        device_map = _select_device_map(self._device)
        aligner_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": dtype,
        }
        if self._device == "cuda" and _supports_flash_attention():
            aligner_kwargs["attn_implementation"] = "flash_attention_2"

        self._aligner = Qwen3ForcedAligner.from_pretrained(QWEN_ASR_FORCE_ALIGNER, **aligner_kwargs)
        return self._aligner

    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> TranscribeResult:
        if not self._loaded or self._model is None:
            raise RuntimeError("Qwen3-ASR model is not loaded — call load() first")

        if len(audio) == 0:
            return TranscribeResult(text="", language=language, duration_ms=0, model=self._model_id)

        audio_duration_ms = int(len(audio) / QWEN_ASR_SAMPLE_RATE * 1000)
        qwen_language = _normalize_language(language)

        temp_path = self._audio_to_temp_wav(audio)
        aligner: Any = None
        align_results: Any = None
        try:
            results = self._model.transcribe(audio=temp_path, language=qwen_language)
            result = results[0] if isinstance(results, (list, tuple)) and results else results
            text = str(getattr(result, "text", "") or "").strip()
            if word_timestamps and text:
                aligner = self._ensure_aligner()
                align_results = aligner.align(audio=temp_path, text=text, language=qwen_language)
        finally:
            with suppress(OSError):
                os.unlink(temp_path)

        result = results[0] if isinstance(results, (list, tuple)) and results else results
        text = getattr(result, "text", "") or ""
        text = str(text).strip()
        language_code = _normalize_language_code(getattr(result, "language", None) or language) or language

        words = ()
        if word_timestamps:
            if text and aligner is not None:
                if isinstance(align_results, (list, tuple)) and align_results:
                    words = self._extract_words(align_results[0])
                elif align_results:
                    words = self._extract_words(align_results)
            elif hasattr(result, "time_stamps"):
                words = self._extract_words(result.time_stamps)

        segments = ()
        if text:
            segments = (
                TranscriptSegment(
                    text=text,
                    start_ms=0,
                    end_ms=audio_duration_ms,
                    words=words,
                    language=language_code,
                ),
            )

        if not text:
            logger.warning("Empty transcription for %dms audio", audio_duration_ms)
        else:
            logger.info("Transcribed %dms audio: %s", audio_duration_ms, text[:80])

        return TranscribeResult(
            text=text,
            segments=segments,
            language=language_code,
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
        if not self._loaded or self._model is None:
            raise RuntimeError("Qwen3-ASR model is not loaded — call load() first")

        snippet = audio[:QWEN_ASR_SAMPLE_RATE * 10]
        temp_path = self._audio_to_temp_wav(snippet)
        try:
            results = self._model.transcribe(audio=temp_path)
        finally:
            with suppress(OSError):
                os.unlink(temp_path)

        result = results[0] if isinstance(results, (list, tuple)) and results else results
        raw = str(getattr(result, "language", "en")).strip().lower()
        return _normalize_language_code(raw) or "en"

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        model_id = kwargs.get("_source") or kwargs.get("model_id") or self._model_id
        return _estimate_vram(str(model_id))
