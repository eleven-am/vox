from __future__ import annotations

import importlib
import logging
import subprocess
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
from vox.operations.errors import InvalidConfigError

logger = logging.getLogger(__name__)

ORPHEUS_SAMPLE_RATE = 24_000
ORPHEUS_RUNTIME_DEPS = ("orpheus-speech==0.1.0",)
DEFAULT_VOICE = "tara"
ORPHEUS_VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zoe", "zac")


def _runtime_root():
    return vox_runtime_root() / "orpheus"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _install_orpheus_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    if not install_target_runtime_requirements(
        runtime_path,
        ORPHEUS_RUNTIME_DEPS,
        timeout=1800,
        install_runner=_run_install_command,
        context="Orpheus runtime install",
    ):
        raise RuntimeError("Failed to install Orpheus runtime package.")


def _clear_orpheus_modules() -> None:
    purge_runtime_modules(("orpheus_tts", "snac", "vllm"))


def _load_orpheus_model_class() -> type[Any]:
    _ensure_runtime_path()
    try:
        module = importlib.import_module("orpheus_tts")
    except ImportError:
        _install_orpheus_runtime()
        _clear_orpheus_modules()
        module = importlib.import_module("orpheus_tts")

    cls = getattr(module, "OrpheusModel", None)
    if cls is None:
        raise RuntimeError("Orpheus runtime is installed, but orpheus_tts.OrpheusModel was not found.")
    return cls


def _select_dtype(device: str) -> Any:
    try:
        import torch
    except ImportError:
        return None
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _require_cuda_device(device: str) -> None:
    if device == "cuda":
        return
    raise RuntimeError(
        "Orpheus requires a Linux x86_64 CUDA runtime. "
        "CPU and Spark/ARM NVIDIA execution are not supported by the "
        "orpheus-speech/vLLM backend."
    )


def _pcm16_bytes_to_float32(pcm: bytes) -> NDArray[np.float32]:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return audio
    return audio / 32768.0


def _audio_array(chunk: Any) -> NDArray[np.float32]:
    if isinstance(chunk, bytes | bytearray):
        return _pcm16_bytes_to_float32(bytes(chunk))
    if hasattr(chunk, "detach"):
        chunk = chunk.detach()
    if hasattr(chunk, "cpu"):
        chunk = chunk.cpu()
    if hasattr(chunk, "numpy"):
        chunk = chunk.numpy()
    audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise RuntimeError("Orpheus produced an empty audio chunk.")
    return audio


class OrpheusAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_id = ""
        self._device = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="orpheus-tts-vllm",
            type=ModelType.TTS,
            architectures=("orpheus-tts-vllm", "orpheus"),
            default_sample_rate=ORPHEUS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._model is not None:
            return

        _require_cuda_device(device)

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = device
        cls = _load_orpheus_model_class()
        dtype = _select_dtype(device)

        logger.info("Loading Orpheus model: %s (device=%s)", self._model_id, self._device)
        try:
            self._model = cls(self._model_id, dtype=dtype)
        except TypeError:
            self._model = cls(self._model_id)

    def unload(self) -> None:
        self._model = None
        self._model_id = ""
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def prepare_runtime(self) -> None:
        _load_orpheus_model_class()

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> None:
        if reference_audio is not None or reference_text is not None:
            raise InvalidConfigError("Orpheus does not support reference_audio/reference_text")

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
        if self._model is None:
            raise RuntimeError("Orpheus model is not loaded — call load() first")
        if reference_audio is not None or reference_text is not None:
            raise ValueError("Orpheus does not support reference_audio/reference_text.")
        if not text or not text.strip():
            return

        selected_voice = voice or DEFAULT_VOICE
        kwargs: dict[str, Any] = {
            "prompt": text,
            "voice": selected_voice,
        }
        if speed and speed != 1.0:
            logger.debug("Orpheus runtime does not expose a speed control; ignoring speed=%s", speed)

        for raw_chunk in self._model.generate_speech(**kwargs):
            audio = _audio_array(raw_chunk)
            if audio.size:
                yield SynthesizeChunk(
                    audio=audio.tobytes(),
                    sample_rate=ORPHEUS_SAMPLE_RATE,
                    is_final=False,
                )

        yield SynthesizeChunk(audio=b"", sample_rate=ORPHEUS_SAMPLE_RATE, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id=voice,
                name=voice.title(),
                language="en",
                description="Orpheus preset voice",
            )
            for voice in ORPHEUS_VOICES
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 10_000_000_000
