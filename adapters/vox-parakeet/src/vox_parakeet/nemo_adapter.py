from __future__ import annotations

import importlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
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
DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_VRAM_BYTES = 2_500_000_000


def _select_device(device: str) -> str:
    torch = _torch_module()
    if device not in ("cuda", "auto"):
        raise RuntimeError(
            "Parakeet NeMo is a CUDA-backed adapter and requires device='cuda' or 'auto'"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Parakeet NeMo requires CUDA; CPU fallback is disabled")
    return "cuda"


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - runtime-image dependent
        raise RuntimeError("Parakeet NeMo requires torch to be installed in the runtime image") from exc


def _load_asr_model_class() -> Any:
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:  # pragma: no cover - runtime-image dependent
        raise RuntimeError(
            "Parakeet NeMo requires nemo-toolkit[asr] to be installed in the runtime image"
        ) from exc

    try:
        return nemo_asr.models.ASRModel
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Parakeet NeMo requires nemo.collections.asr.models.ASRModel"
        ) from exc


def _resolve_model_ref(model_path: str, source: str | None) -> tuple[str, Path | None]:
    path = Path(model_path)
    if path.exists():
        if path.is_file():
            model_ref = source or str(path)
            return model_ref, path if path.suffix == ".nemo" else None

        nemo_files = sorted(path.rglob("*.nemo"))
        if nemo_files:
            return source or str(path), nemo_files[0]

    if source:
        return source, None

    return model_path, None


def _time_stride_seconds(model: Any) -> float:
    cfg = getattr(model, "cfg", None)
    preprocessor = getattr(cfg, "preprocessor", None)
    window_stride = getattr(preprocessor, "window_stride", None)
    if window_stride is None:
        return 0.01
    return float(window_stride) * 8.0


def _extract_text(result: Any) -> str:
    text = getattr(result, "text", result)
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    return str(text or "").strip()


def _extract_timestamp_dict(result: Any) -> dict[str, Any]:
    timestamp = getattr(result, "timestamp", None)
    if isinstance(timestamp, dict):
        return timestamp

    timestep = getattr(result, "timestep", None)
    if isinstance(timestep, dict):
        return timestep

    return {}


def _extract_word_timestamps(result: Any, model: Any) -> list[WordTimestamp]:
    timestamp_dict = _extract_timestamp_dict(result)
    entries = timestamp_dict.get("word") or []
    time_stride = _time_stride_seconds(model)
    words: list[WordTimestamp] = []

    for entry in entries:
        if isinstance(entry, dict):
            word = entry.get("word") or entry.get("char") or entry.get("segment") or ""
            start_offset = entry.get("start_offset", entry.get("start"))
            end_offset = entry.get("end_offset", entry.get("end"))
        else:
            word = getattr(entry, "word", None) or getattr(entry, "char", None) or getattr(entry, "segment", "")
            start_offset = getattr(entry, "start_offset", getattr(entry, "start", None))
            end_offset = getattr(entry, "end_offset", getattr(entry, "end", None))

        if not word or start_offset is None or end_offset is None:
            continue

        words.append(
            WordTimestamp(
                word=str(word),
                start_ms=int(float(start_offset) * time_stride * 1000),
                end_ms=int(float(end_offset) * time_stride * 1000),
            )
        )

    return words


def _to_numpy_audio(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class ParakeetNemoAdapter(STTAdapter):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded = False
        self._model_id: str = DEFAULT_MODEL_ID
        self._device: str = "cuda"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="parakeet-nemo",
            type=ModelType.STT,
            architectures=("parakeet", "parakeet-nemo", "parakeet-tdt"),
            default_sample_rate=PARAKEET_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_word_timestamps=True,
            supports_language_detection=True,
            supported_languages=(),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        self._device = _select_device(device)
        source = kwargs.pop("_source", None)
        self._model_id, checkpoint_path = _resolve_model_ref(model_path, source)
        ASRModel = _load_asr_model_class()

        logger.info("Loading Parakeet NeMo model: %s (device=%s)", self._model_id, self._device)
        start = time.perf_counter()

        if checkpoint_path is not None:
            model = ASRModel.restore_from(restore_path=str(checkpoint_path))
        else:
            model = ASRModel.from_pretrained(model_name=self._model_id)

        if hasattr(model, "to"):
            model = model.to(self._device)
        if hasattr(model, "eval"):
            model.eval()

        self._model = model
        elapsed = time.perf_counter() - start
        logger.info("Parakeet NeMo model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        self._model_id = DEFAULT_MODEL_ID
        self._device = "cuda"
        torch = _torch_module()
        if torch.cuda.is_available() and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        logger.info("Parakeet NeMo adapter unloaded")

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
            raise RuntimeError("Parakeet NeMo model is not loaded — call load() first")

        if initial_prompt:
            logger.warning("Parakeet NeMo does not use initial_prompt; ignoring it")
        if temperature not in (0, 0.0):
            logger.warning("Parakeet NeMo ignores temperature=%s", temperature)
        if language and language not in ("auto", "en", "en-us", "en-gb"):
            logger.warning("Parakeet NeMo auto-detects language; ignoring language=%s", language)

        audio = _to_numpy_audio(audio)
        if audio.size == 0:
            return TranscribeResult(text="", segments=(), language=language, duration_ms=0, model=self._model_id)

        duration_ms = int(len(audio) / PARAKEET_SAMPLE_RATE * 1000)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, PARAKEET_SAMPLE_RATE)
            temp_path = Path(tmp.name)

        try:
            transcribe_kwargs: dict[str, Any] = {"batch_size": 1}
            if word_timestamps:
                transcribe_kwargs["timestamps"] = True
                transcribe_kwargs["return_hypotheses"] = True

            result = self._model.transcribe([str(temp_path)], **transcribe_kwargs)

            if word_timestamps:
                if isinstance(result, tuple):
                    result = result[0]
                hypothesis = result[0] if isinstance(result, (list, tuple)) else result
                text = _extract_text(hypothesis)
                words = tuple(_extract_word_timestamps(hypothesis, self._model))
                segments = (
                    (
                        TranscriptSegment(
                            text=text,
                            start_ms=0,
                            end_ms=duration_ms,
                            words=words,
                            language=language or getattr(hypothesis, "language", None),
                        ),
                    )
                    if text
                    else ()
                )
            else:
                if isinstance(result, tuple):
                    result = result[0]
                text_result = result[0] if isinstance(result, (list, tuple)) else result
                text = _extract_text(text_result)
                segments = (
                    (
                        TranscriptSegment(
                            text=text,
                            start_ms=0,
                            end_ms=duration_ms,
                            language=language or getattr(text_result, "language", None),
                        ),
                    )
                    if text
                    else ()
                )

            detected_source = result[0] if isinstance(result, (list, tuple)) and result else result
            detected_language = language or getattr(detected_source, "language", None)
            return TranscribeResult(
                text=text,
                segments=segments,
                language=detected_language,
                duration_ms=duration_ms,
                model=self._model_id,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return DEFAULT_VRAM_BYTES
