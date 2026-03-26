from __future__ import annotations

import logging
import tempfile
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

# Rough VRAM estimates by model size token in the model ID.
_VRAM_ESTIMATES: dict[str, int] = {
    "0.6b": 1_300_000_000,   # ~1.3 GB
    "1.1b": 2_500_000_000,   # ~2.5 GB
}


def _normalize_model_id(model_id: str) -> str:
    """Convert a HuggingFace repo ID (e.g. ``nvidia/parakeet-tdt-0.6b-v3``)
    to the ``nemo-`` prefixed form that ``onnx-asr`` expects
    (e.g. ``nemo-parakeet-tdt-0.6b-v3``).

    If the string already starts with ``nemo-`` or has no known prefix it is
    returned unchanged.
    """
    if "/" in model_id:
        # Take the repo name after the slash and add the nemo- prefix.
        _, repo_name = model_id.split("/", 1)
        return f"nemo-{repo_name}"
    return model_id


def _get_providers(device: str) -> list[str]:
    """Return ONNX Runtime execution providers for *device*."""
    if device == "cuda":
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.warning(
                "CUDA requested but CUDAExecutionProvider not available, falling back to CPU"
            )
        except ImportError:
            logger.warning("onnxruntime not installed, falling back to CPU")
        except Exception as e:
            logger.error(f"ONNX provider check failed: {e} — falling back to CPU")
    return ["CPUExecutionProvider"]


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

    for token, ts in zip(tokens, timestamps):
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
    # Conservative fallback for unknown sizes.
    return _VRAM_ESTIMATES["0.6b"]


class ParakeetAdapter(STTAdapter):
    """NVIDIA Parakeet STT adapter backed by ``onnx-asr``."""

    def __init__(self) -> None:
        self._model: onnx_asr.adapters.TextResultsAsrAdapter | None = None
        self._model_with_ts: onnx_asr.adapters.TimestampedResultsAsrAdapter | None = None
        self._loaded = False
        self._model_id: str = DEFAULT_MODEL_ID
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    # STTAdapter interface
    # ------------------------------------------------------------------

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="parakeet",
            type=ModelType.STT,
            architectures=("parakeet", "parakeet-tdt", "parakeet-ctc"),
            default_sample_rate=PARAKEET_SAMPLE_RATE,
            supported_formats=(ModelFormat.ONNX,),
            supports_streaming=False,
            supports_word_timestamps=True,
            supports_language_detection=False,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        self._device = device
        # Prefer the catalog source (HuggingFace repo ID) passed via _source;
        # fall back to model_path for backward compatibility.
        source = kwargs.pop("_source", None)
        self._model_id = _normalize_model_id(source if source else model_path)

        logger.info("Loading Parakeet ONNX model: %s", self._model_id)
        start = time.perf_counter()

        providers = _get_providers(device)
        self._model = onnx_asr.load_model(self._model_id, providers=providers)
        self._model_with_ts = self._model.with_timestamps()

        elapsed = time.perf_counter() - start
        logger.info("Parakeet model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._model_with_ts = None
        self._loaded = False
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

        if language and language not in ("en", "english"):
            logger.warning("Parakeet only supports English, ignoring language=%s", language)

        if len(audio) == 0:
            logger.warning("Empty audio buffer, returning empty result")
            return TranscribeResult(text="", language="en", duration_ms=0, model=self._model_id)

        audio_duration_ms = int(len(audio) / PARAKEET_SAMPLE_RATE * 1000)

        # onnx-asr requires a file path — write audio to a temporary WAV.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, PARAKEET_SAMPLE_RATE)
            temp_path = Path(tmp.name)

        try:
            if word_timestamps:
                result = self._model_with_ts.recognize(str(temp_path))
                text = result.text
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
            else:
                text = self._model.recognize(str(temp_path))
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

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return _estimate_vram(self._model_id)
