from __future__ import annotations

import importlib
import logging
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from vox.core.adapter import TTSAdapter
from vox.core.adapter_runtime import (
    activate_runtime_path,
    install_target_runtime_requirements,
    purge_runtime_modules,
)
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox_kokoro.common import (
    OFFICIAL_VOICE_IDS,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    pipeline_lang_code,
    voice_info,
)
from vox_kokoro.phonemizer_compat import patch_espeak_compat

logger = logging.getLogger(__name__)

_EN_CORE_WEB_SM_WHEEL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
_EN_CORE_WEB_SM_MARKERS = (
    "en_core_web_sm",
    "en_core_web_sm-3.8.0.dist-info",
)

_RUNTIME_MODULE_PREFIXES = (
    "kokoro.",
    "misaki.",
    "spacy.",
    "transformers.",
    "accelerate.",
    "huggingface_hub.",
    "tokenizers.",
    "safetensors.",
)
_RUNTIME_MODULE_NAMES = {
    "kokoro",
    "misaki",
    "spacy",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
}


def _runtime_root() -> Path:
    return vox_runtime_root() / "kokoro"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)


def _clear_runtime_modules() -> None:
    purge_runtime_modules(_RUNTIME_MODULE_NAMES)


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
    runtime_root = Path(runtime_path)
    core_runtime_requirements = [
        "kokoro>=0.9.4,<1.0.0",
        "soundfile",
        "misaki[en]>=0.9.4",
        "addict",
        "regex",
        "num2words",
        "loguru",
    ]
    support_runtime_requirements = [
        "espeakng-loader",
        "phonemizer-fork",
        "spacy>=3.8.0,<3.9.0",
    ]
    curated_runtime_requirements = [
        "spacy-curated-transformers>=0.3.0,<0.4.0",
        "curated-transformers>=0.1.0,<0.2.0",
        "curated-tokenizers>=0.0.9,<0.1.0",
        "fsspec>=2023.5.0",
    ]
    transformers_runtime_requirements = [
        "transformers>=4.57.6,<4.58",
        "accelerate>=1.10.0,<2.0.0",
        "huggingface-hub>=0.34,<1.0",
        "tokenizers>=0.22,<0.23.1",
        "safetensors>=0.5,<1.0",
    ]
    if not _install_runtime_requirements(runtime_path, core_runtime_requirements, no_deps=True):
        raise RuntimeError(
            "Kokoro Torch backend requires the official 'kokoro' runtime package"
        )
    if not _install_runtime_requirements(runtime_path, support_runtime_requirements, no_deps=False):
        raise RuntimeError(
            "Kokoro Torch backend requires phonemizer and spaCy runtime dependencies"
        )
    if not _install_runtime_requirements(runtime_path, curated_runtime_requirements, no_deps=True):
        raise RuntimeError(
            "Kokoro Torch backend requires curated transformer runtime dependencies"
        )
    if not _install_runtime_requirements(
        runtime_path,
        transformers_runtime_requirements,
        no_deps=True,
        expected_paths=(
            runtime_root / "transformers",
            runtime_root / "accelerate",
            runtime_root / "huggingface_hub",
            runtime_root / "tokenizers",
            runtime_root / "safetensors",
        ),
    ):
        raise RuntimeError(
            "Kokoro Torch backend requires Hugging Face transformer runtime dependencies"
        )

    if not _spacy_model_installed(Path(runtime_path)) and not _install_runtime_requirements(
        runtime_path, [_EN_CORE_WEB_SM_WHEEL], no_deps=False
    ):
        raise RuntimeError(
            "Kokoro Torch backend requires the English spaCy runtime model"
        )

    if not _spacy_model_installed(Path(runtime_path)):
        raise RuntimeError(
            "Kokoro Torch backend requires the English spaCy runtime model"
        )


def _install_runtime_requirements(
    runtime_path: str,
    requirements: list[str],
    *,
    no_deps: bool,
    expected_paths: tuple[Path, ...] = (),
) -> bool:
    installed = install_target_runtime_requirements(
        runtime_path,
        requirements,
        no_deps=no_deps,
        timeout=1800,
        expected_paths=expected_paths,
        installer_order=("pip", "uv"),
        install_runner=_run_install_command,
        context="Kokoro Torch runtime install",
    )
    if installed:
        logger.info("Installed Kokoro runtime requirements into %s: %s", runtime_path, requirements)
    return installed


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _spacy_model_installed(runtime_root: Path) -> bool:
    return any((runtime_root / marker).exists() for marker in _EN_CORE_WEB_SM_MARKERS)


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
            max_input_chars=250,
        )

    def prepare_runtime(self) -> None:
        self._import_runtime()

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        model_file = self._resolve_model_file(Path(model_path))
        if model_file is None:
            raise FileNotFoundError(f"No Kokoro Torch model file found in {model_path}")

        self._model_file = model_file
        self._device = self._normalize_device(device)
        self._pipelines.clear()


        self._get_pipeline("a")
        logger.info("Kokoro Torch model loaded from %s (device=%s)", model_file, self._device)

    def unload(self) -> None:
        self._model_file = None
        self._pipelines.clear()
        logger.info("Kokoro Torch model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model_file is not None

    def trim(self) -> None:
        self._pipelines = {}

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
        discovered: list[str] = []

        for pipeline in self._pipelines.values():
            get_voices = getattr(pipeline, "get_voices", None)
            if callable(get_voices):
                discovered.extend(str(voice_id) for voice_id in get_voices())
                continue

            voices = getattr(pipeline, "voices", None)
            if isinstance(voices, dict):
                discovered.extend(str(voice_id) for voice_id in voices)
                continue
            if isinstance(voices, (list, tuple)):
                discovered.extend(str(voice_id) for voice_id in voices)

        ordered_voice_ids: list[str] = []
        for voice_id in [*OFFICIAL_VOICE_IDS, *discovered]:
            if voice_id not in ordered_voice_ids:
                ordered_voice_ids.append(voice_id)

        return [voice_info(voice_id) for voice_id in ordered_voice_ids]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 350 * 1024 * 1024

    def _get_pipeline(self, lang_code: str):
        pipeline = self._pipelines.get(lang_code)
        if pipeline is not None:
            return pipeline

        kokoro = self._import_runtime()
        patch_espeak_compat()
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
            logger.warning("Initial Kokoro runtime import failed, bootstrapping runtime: %s", exc)
            _install_runtime()
            _clear_runtime_modules()
            _patch_numpy_core()
            try:
                return importlib.import_module("kokoro")
            except Exception as retry_exc:
                logger.exception("Kokoro runtime import still failing after bootstrap")
                raise RuntimeError(
                    "Kokoro Torch backend requires the official 'kokoro' runtime package"
                ) from retry_exc

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
