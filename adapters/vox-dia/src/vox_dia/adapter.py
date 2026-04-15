from __future__ import annotations

import importlib
import importlib.util
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo

logger = logging.getLogger(__name__)

DIA_SAMPLE_RATE = 44_100
_DEFAULT_MAX_NEW_TOKENS = 3_072
_DEFAULT_GUIDANCE_SCALE = 3.0
_DEFAULT_TEMPERATURE = 1.8
_DEFAULT_TOP_P = 0.90
_DEFAULT_TOP_K = 45


def _select_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_transformers_runtime() -> tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, DiaForConditionalGeneration
    except ImportError:
        _install_transformers_runtime()
        _clear_transformers_modules()
        try:
            from transformers import AutoProcessor, DiaForConditionalGeneration
        except ImportError as retry_exc:
            raise RuntimeError(
                "Dia requires Hugging Face Transformers with DiaForConditionalGeneration support. "
                "Install the main branch of transformers: "
                "`pip install git+https://github.com/huggingface/transformers.git`."
            ) from retry_exc
        return AutoProcessor, DiaForConditionalGeneration

    return AutoProcessor, DiaForConditionalGeneration


def _clear_transformers_modules() -> None:
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _ensure_pip_available() -> None:
    if importlib.util.find_spec("pip") is not None:
        return

    result = subprocess.run(
        [sys.executable, "-m", "ensurepip", "--default-pip"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to bootstrap pip for Dia's runtime install. "
            f"stderr: {result.stderr.strip()}"
        )


def _install_transformers_runtime() -> None:
    _ensure_pip_available()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/huggingface/transformers.git@main",
            "sentencepiece>=0.2.0",
            "tiktoken>=0.9.0",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install Dia runtime from Hugging Face Transformers main branch. "
            f"stderr: {result.stderr.strip()}"
        ) from None


class DiaAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._model_id: str = ""
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
        self._device = _select_device(device)
        if self._device != "cuda":
            raise RuntimeError(
                "Dia requires a CUDA-capable GPU. CPU execution is not supported by the official runtime."
            )

        AutoProcessor, DiaForConditionalGeneration = _load_transformers_runtime()

        logger.info("Loading Dia model: %s (device=%s)", self._model_id, self._device)
        start = time.perf_counter()

        try:
            self._processor = AutoProcessor.from_pretrained(self._model_id)
        except ValueError as exc:
            message = str(exc)
            if "sentencepiece" not in message and "tiktoken" not in message:
                raise
            self._processor = AutoProcessor.from_pretrained(self._model_id, use_fast=False)
        self._model = DiaForConditionalGeneration.from_pretrained(self._model_id).to(self._device)
        self._model.eval()

        elapsed = time.perf_counter() - start
        logger.info("Dia model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Dia adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

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

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
                guidance_scale=_DEFAULT_GUIDANCE_SCALE,
                temperature=_DEFAULT_TEMPERATURE,
                top_p=_DEFAULT_TOP_P,
                top_k=_DEFAULT_TOP_K,
            )

        decoded = self._processor.batch_decode(output)

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = Path(tmp.name)
            self._processor.save_audio(decoded, str(temp_path))

            import soundfile as sf

            audio, sample_rate = sf.read(str(temp_path), dtype="float32")
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

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
