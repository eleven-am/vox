from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx_asr
import soundfile as sf
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

PARAKEET_SAMPLE_RATE = 16_000
DEFAULT_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"
VOICED_EMPTY_RELOAD_THRESHOLD = 3
VOICED_EMPTY_MIN_DURATION_MS = 1_000
VOICED_EMPTY_MIN_PEAK = 0.02
VOICED_EMPTY_MIN_RMS = 0.003

_ENGLISH_LANGUAGE_CODES = frozenset({
    "en",
    "english",
    "en-us",
    "en-gb",
    "en-au",
    "en-ca",
    "en-nz",
    "en-ie",
    "en-za",
    "en-in",
})


_VRAM_ESTIMATES: dict[str, int] = {
    "0.6b": 1_300_000_000,
    "1.1b": 2_500_000_000,
}


def _normalize_model_id(model_id: str) -> str:
    """Convert a HuggingFace repo ID (e.g. ``nvidia/parakeet-tdt-0.6b-v3``)
    to the ``nemo-`` prefixed form that ``onnx-asr`` expects
    (e.g. ``nemo-parakeet-tdt-0.6b-v3``).

    If the string already starts with ``nemo-`` or has no known prefix it is
    returned unchanged.
    """
    if "/" in model_id:

        _, repo_name = model_id.split("/", 1)
        return f"nemo-{repo_name}"
    return model_id


def _resolve_model_spec(model_path: str, source: str | None) -> tuple[str, str | None]:
    """Resolve the ONNX-ASR model identifier and optional local model directory."""
    path = Path(model_path)
    local_dir = str(path) if path.exists() else None

    if source:
        return _normalize_model_id(source), local_dir

    if local_dir is not None:
        return DEFAULT_MODEL_ID, local_dir

    return _normalize_model_id(model_path), None


def _get_providers(device: str) -> tuple[list[str], str]:
    """Return ONNX Runtime execution providers and the resolved device."""
    if device == "cpu":
        return ["CPUExecutionProvider"], "cpu"

    if device in {"cuda", "auto"}:
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
        except ImportError:
            raise RuntimeError("Parakeet requires onnxruntime to be installed") from None

        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"

        if device == "auto":
            return ["CPUExecutionProvider"], "cpu"

        raise RuntimeError(
            "Parakeet requires CUDAExecutionProvider for non-CPU devices; "
            "CPU fallback is disabled"
        )

    return ["CPUExecutionProvider"], "cpu"


@dataclass
class _Word:
    word: str
    start: float
    end: float


def _tokens_to_words(tokens: list[str], timestamps: list[float]) -> list[_Word]:
    """Merge sub-word tokens into whole words using leading-space heuristics."""
    if not tokens or not timestamps:
        return []

    if len(tokens) != len(timestamps):
        logger.warning(
            "Token/timestamp length mismatch: %d tokens, %d timestamps",
            len(tokens),
            len(timestamps),
        )
        min_len = min(len(tokens), len(timestamps))
        tokens = tokens[:min_len]
        timestamps = timestamps[:min_len]

    words: list[_Word] = []
    current_word = ""
    current_start: float | None = None

    for token, ts in zip(tokens, timestamps, strict=False):
        token_stripped = token.strip()
        if not token_stripped:
            continue

        is_punctuation = len(token_stripped) == 1 and not token_stripped.isalnum()

        if token.startswith(" ") or current_start is None:
            if current_word and current_start is not None:
                words.append(_Word(word=current_word, start=current_start, end=ts))
            current_word = token_stripped
            current_start = ts
        elif is_punctuation:
            current_word += token_stripped
        else:
            current_word += token_stripped

    if current_word and current_start is not None:
        end_time = timestamps[-1] if timestamps else current_start
        words.append(_Word(word=current_word, start=current_start, end=end_time))

    return words


def _estimate_vram(model_id: str) -> int:
    """Return a rough VRAM estimate in bytes based on the model ID."""
    lower = model_id.lower()
    for size_key, vram in _VRAM_ESTIMATES.items():
        if size_key in lower:
            return vram

    return _VRAM_ESTIMATES["0.6b"]


class ParakeetAdapter(STTAdapter):
    """NVIDIA Parakeet STT adapter backed by ``onnx-asr``."""

    def __init__(self) -> None:
        self._model: onnx_asr.adapters.TextResultsAsrAdapter | None = None
        self._model_with_ts: onnx_asr.adapters.TimestampedResultsAsrAdapter | None = None
        self._loaded = False
        self._model_id: str = DEFAULT_MODEL_ID
        self._device: str = "cpu"
        self._lock = threading.RLock()
        self._load_model_path: str | None = None
        self._load_device: str = "cpu"
        self._load_source: str | None = None
        self._consecutive_voiced_empty = 0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="parakeet-stt-onnx",
            type=ModelType.STT,
            architectures=("parakeet-stt-onnx", "parakeet", "parakeet-tdt", "parakeet-ctc"),
            default_sample_rate=PARAKEET_SAMPLE_RATE,
            supported_formats=(ModelFormat.ONNX,),
            supports_streaming=False,
            supports_word_timestamps=True,
            supports_language_detection=False,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        with self._lock:
            if self._loaded:
                return

            source = kwargs.pop("_source", None)
            self._load_model_path = model_path
            self._load_device = device
            self._load_source = source
            self._load_locked()

    def _load_locked(self) -> None:
        if self._load_model_path is None:
            raise RuntimeError("Parakeet load path is not configured")

        providers, resolved_device = _get_providers(self._load_device)
        self._device = resolved_device
        self._model_id, local_dir = _resolve_model_spec(self._load_model_path, self._load_source)
        logger.info("Loading Parakeet ONNX model: %s", self._model_id)
        start = time.perf_counter()

        load_kwargs: dict[str, Any] = {"providers": providers}
        if local_dir is not None:
            load_kwargs["path"] = local_dir

        self._model = onnx_asr.load_model(self._model_id, **load_kwargs)
        self._model_with_ts = self._model.with_timestamps()

        elapsed = time.perf_counter() - start
        logger.info("Parakeet model loaded in %.2fs", elapsed)
        self._loaded = True
        self._consecutive_voiced_empty = 0

    def _reload_locked(self) -> None:
        logger.warning(
            "Reloading Parakeet model after %d consecutive voiced empty transcriptions",
            self._consecutive_voiced_empty,
        )
        self._model = None
        self._model_with_ts = None
        self._loaded = False
        self._load_locked()

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._model_with_ts = None
            self._loaded = False
            self._consecutive_voiced_empty = 0
        logger.info("Parakeet adapter unloaded")

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
        if not self._loaded or self._model is None:
            raise RuntimeError("Parakeet model is not loaded — call load() first")

        if language and language.strip().lower() not in _ENGLISH_LANGUAGE_CODES:
            logger.warning("Parakeet only supports English, ignoring language=%s", language)

        if len(audio) == 0:
            logger.warning("Empty audio buffer, returning empty result")
            return TranscribeResult(text="", language="en", duration_ms=0, model=self._model_id)

        audio_duration_ms = int(len(audio) / PARAKEET_SAMPLE_RATE * 1000)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, PARAKEET_SAMPLE_RATE)
            temp_path = Path(tmp.name)

        try:
            with self._lock:
                text, segments = self._recognize_locked(
                    temp_path,
                    audio_duration_ms=audio_duration_ms,
                    word_timestamps=word_timestamps,
                )
                text = text.strip() if text else ""
                if self._should_reload_after_empty_locked(
                    audio,
                    text=text,
                    audio_duration_ms=audio_duration_ms,
                ):
                    self._reload_locked()
                    text, segments = self._recognize_locked(
                        temp_path,
                        audio_duration_ms=audio_duration_ms,
                        word_timestamps=word_timestamps,
                    )
        finally:
            temp_path.unlink(missing_ok=True)

        text = text.strip() if text else ""

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

    def _recognize_locked(
        self,
        temp_path: Path,
        *,
        audio_duration_ms: int,
        word_timestamps: bool,
    ) -> tuple[str, tuple[TranscriptSegment, ...]]:
        if not self._loaded or self._model is None:
            raise RuntimeError("Parakeet model is not loaded — call load() first")

        if word_timestamps:
            if self._model_with_ts is None:
                raise RuntimeError("Parakeet timestamp model is not loaded")
            result = self._model_with_ts.recognize(str(temp_path))
            text = (result.text or "").strip()
            words = _tokens_to_words(result.tokens, result.timestamps)
            word_ts = tuple(
                WordTimestamp(
                    word=w.word,
                    start_ms=int(w.start * 1000),
                    end_ms=int(w.end * 1000),
                )
                for w in words
            )
            segments = (
                (TranscriptSegment(
                    text=text,
                    start_ms=0,
                    end_ms=audio_duration_ms,
                    words=word_ts,
                    language="en",
                ),)
                if text
                else ()
            )
            return text, segments

        text = (self._model.recognize(str(temp_path)) or "").strip()
        segments = (
            (TranscriptSegment(
                text=text,
                start_ms=0,
                end_ms=audio_duration_ms,
                language="en",
            ),)
            if text
            else ()
        )
        return text, segments

    def _should_reload_after_empty_locked(
        self,
        audio: NDArray[np.float32],
        *,
        text: str,
        audio_duration_ms: int,
    ) -> bool:
        if text:
            self._consecutive_voiced_empty = 0
            return False

        if audio_duration_ms < VOICED_EMPTY_MIN_DURATION_MS or not _looks_voiced(audio):
            return False

        self._consecutive_voiced_empty += 1
        logger.warning(
            "Parakeet returned an empty transcript for voiced %dms audio (%d/%d)",
            audio_duration_ms,
            self._consecutive_voiced_empty,
            VOICED_EMPTY_RELOAD_THRESHOLD,
        )
        return self._consecutive_voiced_empty >= VOICED_EMPTY_RELOAD_THRESHOLD

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        model_id = kwargs.get("_source") or kwargs.get("model_id") or self._model_id or DEFAULT_MODEL_ID
        return _estimate_vram(_normalize_model_id(str(model_id)))


def _looks_voiced(audio: NDArray[np.float32]) -> bool:
    if audio.size == 0:
        return False

    finite_audio = audio[np.isfinite(audio)]
    if finite_audio.size == 0:
        return False

    peak = float(np.max(np.abs(finite_audio)))
    rms = float(np.sqrt(np.mean(np.square(finite_audio, dtype=np.float64))))
    return peak >= VOICED_EMPTY_MIN_PEAK and rms >= VOICED_EMPTY_MIN_RMS
