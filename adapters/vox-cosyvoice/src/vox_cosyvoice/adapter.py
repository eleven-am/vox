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
)
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.operations.errors import InvalidConfigError

logger = logging.getLogger(__name__)

COSYVOICE_SAMPLE_RATE = 24_000
COSYVOICE_REPO = "git+https://github.com/FunAudioLLM/CosyVoice.git"


def _runtime_root() -> Path:
    return vox_runtime_root() / "cosyvoice"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _install_cosyvoice_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    if not install_target_runtime_requirements(
        runtime_path,
        (COSYVOICE_REPO,),
        timeout=1800,
        install_runner=_run_install_command,
        context="CosyVoice runtime install",
    ):
        raise RuntimeError("Failed to install CosyVoice runtime from GitHub.")


def _clear_cosyvoice_modules() -> None:
    purge_runtime_modules(("cosyvoice", "matcha", "wetext"))


def _load_cosyvoice_class() -> type[Any]:
    _ensure_runtime_path()
    try:
        module = importlib.import_module("cosyvoice.cli.cosyvoice")
    except ImportError:
        _install_cosyvoice_runtime()
        _clear_cosyvoice_modules()
        module = importlib.import_module("cosyvoice.cli.cosyvoice")

    cls = getattr(module, "CosyVoice2", None)
    if cls is None:
        raise RuntimeError("CosyVoice runtime is installed, but cosyvoice.cli.cosyvoice.CosyVoice2 was not found.")
    return cls


def _voice_path(voice: str | None) -> Path | None:
    if not voice:
        return None
    path = Path(voice).expanduser()
    return path if path.is_file() else None


def _write_reference_audio(path: Path, reference_audio: NDArray[np.float32], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(reference_audio, dtype=np.float32), sample_rate)


def _extract_audio(output: Any) -> NDArray[np.float32]:
    if isinstance(output, dict):
        for key in ("tts_speech", "audio", "wav", "waveform"):
            if key in output:
                return _extract_audio(output[key])
        raise RuntimeError("CosyVoice returned a dict without audio data.")
    if isinstance(output, tuple | list) and output and not isinstance(output[0], (int, float, np.number)):
        return _extract_audio(output[0])
    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "cpu"):
        output = output.cpu()
    if hasattr(output, "numpy"):
        output = output.numpy()
    audio = np.asarray(output, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise RuntimeError("CosyVoice produced no audio.")
    return audio


class CosyVoice2Adapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_id = ""
        self._device = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="cosyvoice2-tts-torch",
            type=ModelType.TTS,
            architectures=("cosyvoice2-tts-torch", "cosyvoice2", "cosyvoice"),
            default_sample_rate=COSYVOICE_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=True,
            supported_languages=(),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._model is not None:
            return

        kwargs.pop("_source", None)
        self._model_id = model_path
        self._device = device
        cls = _load_cosyvoice_class()

        logger.info("Loading CosyVoice2 model from %s (device=%s)", model_path, self._device)
        try:
            self._model = cls(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=device == "cuda")
        except TypeError:
            self._model = cls(model_path)

    def unload(self) -> None:
        self._model = None
        self._model_id = ""
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> None:
        if reference_audio is not None:
            if np.asarray(reference_audio, dtype=np.float32).size == 0:
                raise InvalidConfigError("CosyVoice2 reference_audio is empty")
            return
        if _voice_path(voice) is not None:
            return
        if voice and voice.strip():
            return
        raise InvalidConfigError("CosyVoice2 requires reference_audio, a voice path, or a zero_shot_spk_id voice value")

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
            raise RuntimeError("CosyVoice2 model is not loaded — call load() first")
        if not text or not text.strip():
            return

        voice_file = _voice_path(voice)
        zero_shot_spk_id = "" if voice_file is not None else (voice or "")
        if reference_audio is None and voice_file is None and not zero_shot_spk_id:
            raise ValueError("CosyVoice2 requires reference_audio, a voice path, or a zero_shot_spk_id voice value.")

        with tempfile.TemporaryDirectory(prefix="vox-cosyvoice-") as tmpdir:
            prompt_wav = ""
            if reference_audio is not None:
                ref_path = Path(tmpdir) / "reference.wav"
                _write_reference_audio(ref_path, reference_audio, COSYVOICE_SAMPLE_RATE)
                prompt_wav = str(ref_path)
            elif voice_file is not None:
                prompt_wav = str(voice_file)

            outputs = self._model.inference_zero_shot(
                text,
                reference_text or "",
                prompt_wav,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=True,
                speed=speed,
            )

            yielded = False
            for output in outputs:
                audio = _extract_audio(output)
                if audio.size:
                    yielded = True
                    yield SynthesizeChunk(
                        audio=audio.tobytes(),
                        sample_rate=COSYVOICE_SAMPLE_RATE,
                        is_final=False,
                    )

        if not yielded:
            raise RuntimeError("CosyVoice2 produced no audio.")

        yield SynthesizeChunk(audio=b"", sample_rate=COSYVOICE_SAMPLE_RATE, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="reference",
                name="Reference audio",
                language=None,
                description="Pass reference_audio/reference_text, a voice path, or a saved zero_shot_spk_id.",
                is_cloned=True,
            )
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 5_000_000_000
