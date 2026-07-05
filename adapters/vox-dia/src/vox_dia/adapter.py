from __future__ import annotations

import asyncio
import importlib
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.adapter_runtime import (
    activate_runtime_path,
    install_target_runtime_requirements,
    write_app_fallback_path,
)
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo

logger = logging.getLogger(__name__)

DIA_SAMPLE_RATE = 44_100
_DEFAULT_MAX_NEW_TOKENS = 3_072
_DEFAULT_GUIDANCE_SCALE = 3.0
_DEFAULT_TEMPERATURE = 1.8
_DEFAULT_TOP_P = 0.90
_DEFAULT_TOP_K = 45
_DIA_TRANSFORMERS_SPEC = "git+https://github.com/huggingface/transformers.git@main"
_DIA_RUNTIME_IMPORT = "transformers.models.dia.modeling_dia"
_DIA_RUNTIME_PACKAGES = (
    "sentencepiece>=0.2.0,<0.3",
    "tiktoken>=0.9.0,<1",
)


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Dia requires PyTorch at runtime. Use a Vox image/runtime with torch and CUDA support."
        ) from exc


def _runtime_root() -> Path:
    return vox_runtime_root() / "dia"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_app_fallback_path(runtime_dir)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)



def _load_transformers_runtime() -> tuple[Any, Any]:
    runtime_path = _ensure_runtime_path()
    if not _runtime_has_dia_support():
        _install_transformers_runtime()
        _clear_transformers_modules()
        _ensure_runtime_path()
    if not _runtime_has_dia_support():
        raise RuntimeError(
            "Dia requires Hugging Face Transformers with DiaForConditionalGeneration support. "
            f"The isolated Dia runtime at {runtime_path} could not expose the required symbols."
        )

    from transformers import AutoProcessor, DiaForConditionalGeneration

    return AutoProcessor, DiaForConditionalGeneration


def _clear_transformers_modules() -> None:
    for name in list(sys.modules):
        if name == "transformers" or name.startswith(("transformers.", "sentencepiece", "tiktoken", "regex")):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _install_transformers_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    install_groups = (
        (
            (_DIA_TRANSFORMERS_SPEC,),
            True,
        ),
        (
            _DIA_RUNTIME_PACKAGES,
            False,
        ),
    )

    for requirements, no_deps in install_groups:
        if not install_target_runtime_requirements(
            runtime_path,
            requirements,
            no_deps=no_deps,
            upgrade=not no_deps,
            timeout=900,
            install_runner=_run_install_command,
            context="Dia runtime install",
        ):
            raise RuntimeError(
                "Failed to install Dia runtime from Hugging Face Transformers main branch."
            ) from None
    _clear_transformers_modules()
    _ensure_runtime_path()
    if not _runtime_has_dia_support():
        raise RuntimeError("Dia runtime install did not expose DiaForConditionalGeneration")


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _runtime_has_dia_support() -> bool:
    _ensure_runtime_path()
    try:
        importlib.import_module(_DIA_RUNTIME_IMPORT)
        transformers = importlib.import_module("transformers")
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False
    return hasattr(transformers, "DiaForConditionalGeneration") and hasattr(transformers, "AutoProcessor")


def _decode_dia_output(
    *,
    processor: Any,
    model: Any,
    inputs: Any,
    temp_path: Path,
) -> tuple[np.ndarray, int]:
    torch = _torch_module()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
            guidance_scale=_DEFAULT_GUIDANCE_SCALE,
            temperature=_DEFAULT_TEMPERATURE,
            top_p=_DEFAULT_TOP_P,
            top_k=_DEFAULT_TOP_K,
        )

    decoded = processor.batch_decode(output)
    processor.save_audio(decoded, str(temp_path))

    import soundfile as sf

    audio, sample_rate = sf.read(str(temp_path), dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


class DiaAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._model_id: str = ""
        self._model_ref: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="dia-tts-torch",
            type=ModelType.TTS,
            architectures=("dia-tts-torch", "dia"),
            default_sample_rate=DIA_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_voice_cloning=False,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        path = Path(model_path)
        self._model_ref = str(path) if path.exists() else self._model_id
        self._device = device
        if self._device != "cuda":
            raise RuntimeError(
                "Dia requires a CUDA-capable GPU. CPU execution is not supported by the official runtime."
            )
        _torch_module()

        AutoProcessor, DiaForConditionalGeneration = _load_transformers_runtime()

        logger.info("Loading Dia model: %s (device=%s)", self._model_ref, self._device)
        start = time.perf_counter()

        self._processor = AutoProcessor.from_pretrained(self._model_ref)
        self._model = DiaForConditionalGeneration.from_pretrained(self._model_ref).to(self._device)
        self._model.eval()

        elapsed = time.perf_counter() - start
        logger.info("Dia model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        self._model_ref = ""
        try:
            torch = _torch_module()
        except RuntimeError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Dia adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def prepare_runtime(self) -> None:
        _load_transformers_runtime()

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Dia model is not loaded — call load() first")

        if reference_audio is not None or reference_text is not None:
            raise NotImplementedError(
                "Dia transformers backend does not yet wire the audio-prompt voice cloning path. "
                "Use the official nari-labs/dia runtime if you need reference-audio cloning."
            )

        if not text or not text.strip():
            return

        inputs = self._processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(self._device) if hasattr(inputs, "to") else inputs

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = Path(tmp.name)
            audio, sample_rate = await asyncio.to_thread(
                _decode_dia_output,
                processor=self._processor,
                model=self._model,
                inputs=inputs,
                temp_path=temp_path,
            )

            chunk_size = sample_rate * 2
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                yield SynthesizeChunk(
                    audio=chunk.tobytes(),
                    sample_rate=sample_rate,
                    is_final=False,
                )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        yield SynthesizeChunk(audio=b"", sample_rate=DIA_SAMPLE_RATE, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return []

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 10_000_000_000
