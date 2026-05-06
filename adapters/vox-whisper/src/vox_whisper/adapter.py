from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000
_RUNTIME_SENTINEL = ".vox-whisper-runtime-ready"
_RUNTIME_DEPENDENCIES = (
    "faster-whisper>=1.2.1,<2.0.0",
    "ctranslate2>=4.7.1,<5.0.0",
)
_VOX_HOME_ENV = "VOX_HOME"

_VRAM_ESTIMATES: dict[str, int] = {
    "large-v3-turbo": 2_500_000_000,
    "large-v3": 4_000_000_000,
    "base.en": 900_000_000,
}


def _select_compute_type(device: str, requested: str | None = None) -> str:
    if requested:
        return requested
    if device == "cuda":
        return "float16"
    return "int8"


def _should_fallback_to_cpu(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "not compiled with cuda support" in message
        or "cuda support" in message
        or "cuda unavailable" in message
        or "no cuda-capable device is detected" in message
    )


def _load_whisper_model_class() -> Any:
    try:
        from faster_whisper import WhisperModel
        return WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "Whisper requires the faster-whisper runtime package; install faster-whisper in the image"
        ) from exc


def _adapter_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_target_dir() -> Path:
    env_home = os.environ.get(_VOX_HOME_ENV)
    vox_home = Path(env_home) if env_home else Path.home() / ".vox"
    return vox_home / "runtime" / "whisper"


def _activate_runtime_path() -> None:
    runtime_dir = _runtime_target_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = str(runtime_dir)
    if runtime_path in sys.path:
        sys.path.remove(runtime_path)
    sys.path.insert(0, runtime_path)
    importlib.invalidate_caches()


def _runtime_module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _ensure_runtime_dependencies() -> None:
    _activate_runtime_path()
    if _runtime_module_available("faster_whisper") and _runtime_module_available("ctranslate2"):
        return

    adapter_root = _runtime_target_dir()
    sentinel = adapter_root / _RUNTIME_SENTINEL
    if sentinel.is_file():
        return

    logger.info("Installing Whisper runtime dependencies into %s", adapter_root)
    adapter_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(adapter_root),
        *_RUNTIME_DEPENDENCIES,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install Whisper runtime dependencies: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    sentinel.touch()


class WhisperAdapter(STTAdapter):
    def __init__(self) -> None:
        self._model: Any = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"
        self._compute_type: str = "int8"
        self._beam_size: int = 5

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="whisper-stt-ct2",
            type=ModelType.STT,
            architectures=("whisper-stt-ct2", "whisper"),
            default_sample_rate=WHISPER_SAMPLE_RATE,
            supported_formats=(ModelFormat.CT2,),
            supports_streaming=False,
            supports_word_timestamps=True,
            supports_language_detection=True,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        _ensure_runtime_dependencies()
        source = kwargs.pop("_source", None)
        self._model_id = str(source or model_path)
        self._device = device
        requested_compute_type = kwargs.pop("compute_type", None)
        self._compute_type = _select_compute_type(self._device, requested_compute_type)
        self._beam_size = int(kwargs.pop("beam_size", 5))

        model_ref = str(Path(model_path)) if Path(model_path).exists() else self._model_id
        logger.info(
            "Loading Whisper model: %s (device=%s, compute_type=%s, beam_size=%s)",
            model_ref,
            self._device,
            self._compute_type,
            self._beam_size,
        )
        start = time.perf_counter()

        WhisperModel = _load_whisper_model_class()
        try:
            self._model = WhisperModel(
                model_ref,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as exc:
            if self._device != "cuda" or not _should_fallback_to_cpu(exc):
                raise
            logger.warning(
                "Whisper CUDA runtime unavailable for %s; falling back to CPU: %s",
                model_ref,
                exc,
            )
            self._device = "cpu"
            self._compute_type = _select_compute_type("cpu", requested_compute_type)
            self._model = WhisperModel(
                model_ref,
                device=self._device,
                compute_type=self._compute_type,
            )

        elapsed = time.perf_counter() - start
        logger.info("Whisper model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        self._model_id = ""
        self._device = "cpu"
        self._compute_type = "int8"
        self._beam_size = 5
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str | None,
        word_timestamps: bool,
        initial_prompt: str | None,
        temperature: float,
    ) -> tuple[list[TranscriptSegment], str | None]:
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded — call load() first")

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return [], language

        segments_iter, info = self._model.transcribe(
            audio,
            beam_size=self._beam_size,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        segments: list[TranscriptSegment] = []
        for segment in segments_iter:
            text = str(segment.text or "").strip()
            words: list[WordTimestamp] = []
            for word in getattr(segment, "words", []) or []:
                word_text = getattr(word, "word", None)
                if not word_text:
                    continue
                start = getattr(word, "start", None)
                end = getattr(word, "end", None)
                if start is None or end is None:
                    continue
                words.append(
                    WordTimestamp(
                        word=str(word_text),
                        start_ms=int(float(start) * 1000),
                        end_ms=int(float(end) * 1000),
                        confidence=getattr(word, "probability", None),
                    )
                )
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_ms=int(float(getattr(segment, "start", 0.0)) * 1000),
                    end_ms=int(float(getattr(segment, "end", 0.0)) * 1000),
                    words=tuple(words),
                    language=getattr(segment, "language", None) or getattr(info, "language", None),
                    confidence=getattr(segment, "avg_logprob", None),
                )
            )

        return segments, getattr(info, "language", None) or language

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
            raise RuntimeError("Whisper model is not loaded — call load() first")

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscribeResult(text="", segments=(), language=language, duration_ms=0, model=self._model_id)

        segments, detected_language = self._transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
        )
        text = " ".join(segment.text for segment in segments if segment.text).strip()
        duration_ms = int(round((audio.size / WHISPER_SAMPLE_RATE) * 1000))
        return TranscribeResult(
            text=text,
            segments=tuple(segments),
            language=detected_language,
            duration_ms=duration_ms,
            model=self._model_id,
        )

    def detect_language(self, audio: NDArray[np.float32]) -> str:
        if not self._loaded or self._model is None:
            raise RuntimeError("Whisper model is not loaded — call load() first")

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""

        _, detected_language = self._transcribe(
            audio,
            language=None,
            word_timestamps=False,
            initial_prompt=None,
            temperature=0.0,
        )
        return detected_language or ""

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        model_id = str(kwargs.get("_source") or kwargs.get("model_id") or self._model_id).lower()
        for key, estimate in _VRAM_ESTIMATES.items():
            if key in model_id:
                return estimate
        return _VRAM_ESTIMATES["large-v3"]
