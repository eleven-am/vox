from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox_kokoro.common import (
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    pipeline_lang_code,
    voice_info,
)

logger = logging.getLogger(__name__)


def _runtime_root() -> Path:
    vox_home = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))
    return vox_home / "runtime" / "kokoro"


def _ensure_runtime_path() -> str:
    runtime_path = str(_runtime_root())
    _runtime_root().mkdir(parents=True, exist_ok=True)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    return runtime_path


def _clear_runtime_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "kokoro" or module_name.startswith(
            ("kokoro.", "misaki.", "transformers.", "accelerate.")
        ) or module_name in {"misaki", "transformers", "accelerate"}:
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _patch_numpy_core() -> None:
    try:
        import numpy as np
        import numpy._core as np_core
    except Exception:
        return

    if not hasattr(np_core, "multiarray") and hasattr(np, "core") and hasattr(np.core, "multiarray"):
        np_core.multiarray = np.core.multiarray


def _install_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    installers = [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            runtime_path,
            "kokoro>=0.9.4,<1.0.0",
            "soundfile",
            "accelerate>=1.10.0,<2.0.0",
            "misaki[en]>=0.9.4",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            runtime_path,
            "kokoro>=0.9.4,<1.0.0",
            "soundfile",
            "accelerate>=1.10.0,<2.0.0",
            "misaki[en]>=0.9.4",
        ],
    ]
    for installer in installers:
        try:
            result = subprocess.run(
                installer,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            logger.info("Bootstrapped Kokoro runtime into %s", runtime_path)
            return
        logger.warning("%s failed: %s", " ".join(installer), result.stderr)

    raise RuntimeError(
        "Kokoro Torch backend requires the official 'kokoro' runtime package"
    )


class KokoroTorchAdapter(TTSAdapter):
    """Vox TTS adapter backed by the native Kokoro Torch runtime."""

    def __init__(self) -> None:
        self._model_file: Path | None = None
        self._device: str = "cpu"
        self._pipelines: dict[str, Any] = {}

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="kokoro-tts-torch",
            type=ModelType.TTS,
            architectures=("kokoro-tts-torch", "kokoro-torch"),
            default_sample_rate=SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=SUPPORTED_LANGUAGES,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        model_file = self._resolve_model_file(Path(model_path))
        if model_file is None:
            raise FileNotFoundError(f"No Kokoro Torch model file found in {model_path}")

        self._model_file = model_file
        self._device = self._normalize_device(device)
        self._pipelines.clear()

        # Validate the runtime eagerly so missing Kokoro installs fail at load time.
        self._get_pipeline("a")
        logger.info("Kokoro Torch model loaded from %s (device=%s)", model_file, self._device)

    def unload(self) -> None:
        self._model_file = None
        self._pipelines.clear()
        logger.info("Kokoro Torch model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model_file is not None

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: np.ndarray | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if self._model_file is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not text or not text.strip():
            return

        voice_id = voice or "af_heart"
        lang_code = pipeline_lang_code(voice_id, language)
        pipeline = self._get_pipeline(lang_code)
        generator = pipeline(text, voice=voice_id, speed=speed, split_pattern=r"\n+")

        for _, _, audio in generator:
            yield SynthesizeChunk(
                audio=np.asarray(audio, dtype=np.float32).tobytes(),
                sample_rate=SAMPLE_RATE,
                is_final=False,
            )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=SAMPLE_RATE,
            is_final=True,
        )

    def list_voices(self) -> list[VoiceInfo]:
        if not self._pipelines:
            return []

        pipeline = next(iter(self._pipelines.values()))
        get_voices = getattr(pipeline, "get_voices", None)
        if callable(get_voices):
            return [voice_info(voice_id) for voice_id in get_voices()]

        voices = getattr(pipeline, "voices", None)
        if isinstance(voices, dict):
            return [voice_info(voice_id) for voice_id in voices]
        if isinstance(voices, (list, tuple)):
            return [voice_info(voice_id) for voice_id in voices]
        return []

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 350 * 1024 * 1024

    def _get_pipeline(self, lang_code: str):
        pipeline = self._pipelines.get(lang_code)
        if pipeline is not None:
            return pipeline

        kokoro = self._import_runtime()
        pipeline_cls = getattr(kokoro, "KPipeline", None)
        if pipeline_cls is None:
            raise RuntimeError("kokoro runtime does not expose KPipeline")

        if self._model_file is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pipeline = pipeline_cls(lang_code=lang_code, model=str(self._model_file), device=self._device)
        self._pipelines[lang_code] = pipeline
        return pipeline

    def _import_runtime(self):
        _ensure_runtime_path()
        _patch_numpy_core()
        try:
            return importlib.import_module("kokoro")
        except Exception as exc:
            _install_runtime()
            _clear_runtime_modules()
            _patch_numpy_core()
            try:
                return importlib.import_module("kokoro")
            except Exception:
                raise RuntimeError(
                    "Kokoro Torch backend requires the official 'kokoro' runtime package"
                ) from exc

    def _resolve_model_file(self, model_path: Path) -> Path | None:
        if model_path.is_file():
            return model_path

        candidates = (
            model_path / "kokoro-v1_0.pth",
            model_path / "model.pth",
            model_path / "model.pt",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pth_files = sorted(model_path.glob("*.pth"))
        if len(pth_files) == 1:
            return pth_files[0]
        if len(pth_files) > 1:
            for candidate in pth_files:
                if candidate.name == "kokoro-v1_0.pth":
                    return candidate
            for candidate in pth_files:
                if candidate.name == "model.pth":
                    return candidate
            return pth_files[0]

        pt_files = sorted(model_path.glob("*.pt"))
        if len(pt_files) == 1:
            return pt_files[0]
        return None

    def _normalize_device(self, device: str) -> str:
        requested = (device or "auto").strip().lower()
        if requested == "auto":
            return "cuda" if self._cuda_is_available() else "cpu"
        if requested == "cuda" and not self._cuda_is_available():
            raise RuntimeError("Kokoro Torch backend requires CUDA when device='cuda'")
        return requested

    def _cuda_is_available(self) -> bool:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            return False
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        return bool(is_available()) if callable(is_available) else False
