from __future__ import annotations

import importlib
import logging
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.adapter_runtime import (
    activate_runtime_path,
    install_target_runtime_requirements,
    purge_runtime_modules,
    write_app_fallback_path,
)
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo

logger = logging.getLogger(__name__)

NEUTTS_SAMPLE_RATE = 24_000
NEUTTS_RUNTIME_DEPS = ("neutts==1.2.1",)
NEUTTS_BACKBONE = "neuphonic/neutts-air"
NEUTTS_CODEC = "neuphonic/neucodec"


def _runtime_root() -> Path:
    return vox_runtime_root() / "neutts"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_app_fallback_path(runtime_dir)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _install_neutts_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    if not install_target_runtime_requirements(
        runtime_path,
        NEUTTS_RUNTIME_DEPS,
        timeout=1800,
        install_runner=_run_install_command,
        context="NeuTTS runtime install",
    ):
        raise RuntimeError("Failed to install NeuTTS runtime package.")


def _clear_neutts_modules() -> None:
    purge_runtime_modules(("neutts", "neuttsair", "neucodec"))


def _load_neutts_class() -> type[Any]:
    _ensure_runtime_path()
    try:
        module = importlib.import_module("neuttsair")
    except ImportError:
        _install_neutts_runtime()
        _clear_neutts_modules()
        module = importlib.import_module("neuttsair")

    cls = getattr(module, "NeuTTSAir", None)
    if cls is None:
        raise RuntimeError("NeuTTS runtime is installed, but neuttsair.NeuTTSAir was not found.")
    return cls


def _device_name(device: str) -> str:
    if device == "cuda":
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


def _voice_path(voice: str | None) -> Path | None:
    if not voice:
        return None
    path = Path(voice).expanduser()
    return path if path.is_file() else None


def _write_reference_audio(path: Path, reference_audio: NDArray[np.float32], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(reference_audio, dtype=np.float32), sample_rate)


def _audio_array(audio: Any) -> NDArray[np.float32]:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise RuntimeError("NeuTTS produced no audio.")
    return array


class NeuTTSAirAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._device = "cpu"
        self._model_id = ""

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="neutts-air-tts-torch",
            type=ModelType.TTS,
            architectures=("neutts-air-tts-torch", "neutts-air", "neutts"),
            default_sample_rate=NEUTTS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=True,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._model is not None:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else NEUTTS_BACKBONE
        self._device = _device_name(device)
        cls = _load_neutts_class()
        logger.info("Loading NeuTTS Air model: %s (device=%s)", self._model_id, self._device)
        self._model = cls(
            backbone_repo=self._model_id,
            backbone_device=self._device,
            codec_repo=NEUTTS_CODEC,
            codec_device=self._device,
        )

    def unload(self) -> None:
        self._model = None
        self._device = "cpu"
        self._model_id = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

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
            raise RuntimeError("NeuTTS Air model is not loaded — call load() first")
        if not text or not text.strip():
            return

        voice_file = _voice_path(voice)
        if reference_audio is None and voice_file is None:
            raise ValueError("NeuTTS Air requires reference_audio or a voice path.")

        with tempfile.TemporaryDirectory(prefix="vox-neutts-") as tmpdir:
            if reference_audio is not None:
                ref_path = Path(tmpdir) / "reference.wav"
                _write_reference_audio(ref_path, reference_audio, NEUTTS_SAMPLE_RATE)
            else:
                assert voice_file is not None
                ref_path = voice_file

            ref_codes = self._model.encode_reference(ref_path)
            ref_text = reference_text or ""

            emitted_any = False
            stream_supported = True
            infer_stream = getattr(self._model, "infer_stream", None)
            if callable(infer_stream):
                try:
                    for audio_chunk in infer_stream(text, ref_codes, ref_text):
                        audio = _audio_array(audio_chunk)
                        emitted_any = True
                        yield SynthesizeChunk(audio=audio.tobytes(), sample_rate=NEUTTS_SAMPLE_RATE, is_final=False)
                except NotImplementedError:
                    stream_supported = False
            else:
                stream_supported = False

            if not stream_supported and not emitted_any:
                audio = _audio_array(self._model.infer(text, ref_codes, ref_text))
                yield SynthesizeChunk(audio=audio.tobytes(), sample_rate=NEUTTS_SAMPLE_RATE, is_final=False)

        yield SynthesizeChunk(audio=b"", sample_rate=NEUTTS_SAMPLE_RATE, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="reference",
                name="Reference audio",
                language="en",
                description="Pass reference_audio/reference_text or a voice path.",
                is_cloned=True,
            )
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 4_000_000_000
