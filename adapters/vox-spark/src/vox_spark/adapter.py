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
    staged_target_runtime,
    write_app_fallback_path,
)
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo

logger = logging.getLogger(__name__)

SPARK_SAMPLE_RATE = 16_000
SPARK_REPO = "https://github.com/SparkAudio/Spark-TTS.git"
SPARK_RUNTIME_DEPS = (
    "einops==0.8.1",
    "einx==0.3.0",
    "omegaconf==2.3.0",
    "safetensors==0.5.2",
    "soxr==0.5.0.post1",
    "transformers==4.46.2",
)


def _runtime_root() -> Path:
    return vox_runtime_root() / "spark"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_app_fallback_path(runtime_dir)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _run_git_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _install_spark_runtime() -> None:
    runtime_path = Path(_ensure_runtime_path())
    spark_file = runtime_path / "cli" / "SparkTTS.py"
    if not spark_file.is_file():
        with staged_target_runtime(runtime_path) as stage:
            result = _run_git_command(
                ["git", "clone", "--depth", "1", SPARK_REPO, str(stage)],
                timeout=900,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone Spark-TTS runtime: {result.stderr.strip()}")
            if not (stage / "cli" / "SparkTTS.py").is_file():
                raise RuntimeError("Spark-TTS runtime source checkout is incomplete.")
            write_app_fallback_path(stage)

    if not install_target_runtime_requirements(
        runtime_path,
        SPARK_RUNTIME_DEPS,
        timeout=1200,
        install_runner=_run_install_command,
        context="Spark-TTS runtime install",
    ):
        raise RuntimeError("Failed to install Spark-TTS runtime dependencies.")


def _clear_spark_modules() -> None:
    purge_runtime_modules(("cli", "sparktts"))


def _load_spark_class() -> type[Any]:
    _ensure_runtime_path()
    try:
        module = importlib.import_module("cli.SparkTTS")
    except ImportError:
        _install_spark_runtime()
        _clear_spark_modules()
        _ensure_runtime_path()
        module = importlib.import_module("cli.SparkTTS")

    cls = getattr(module, "SparkTTS", None)
    if cls is None:
        raise RuntimeError("Spark-TTS runtime is installed, but cli.SparkTTS.SparkTTS was not found.")
    return cls


def _torch_device(device: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Spark-TTS requires PyTorch in the Vox compute environment.") from exc
    if device == "cuda":
        return torch.device("cuda:0")
    if device == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def _write_reference_audio(path: Path, reference_audio: NDArray[np.float32], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(reference_audio, dtype=np.float32), sample_rate)


def _voice_path(voice: str | None) -> Path | None:
    if not voice:
        return None
    path = Path(voice).expanduser()
    return path if path.is_file() else None


def _audio_array(audio: Any) -> NDArray[np.float32]:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise RuntimeError("Spark-TTS produced no audio.")
    return array


def _speed_label(speed: float) -> str:
    if speed <= 0.75:
        return "low"
    if speed >= 1.25:
        return "high"
    return "moderate"


class SparkTTSAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._device = "cpu"
        self._model_id = ""

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="spark-tts-torch",
            type=ModelType.TTS,
            architectures=("spark-tts-torch", "spark-tts"),
            default_sample_rate=SPARK_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_voice_cloning=True,
            supported_languages=("en", "zh"),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._model is not None:
            return

        kwargs.pop("_source", None)
        self._model_id = model_path
        self._device = device
        cls = _load_spark_class()
        logger.info("Loading Spark-TTS model from %s (device=%s)", model_path, self._device)
        self._model = cls(Path(model_path), _torch_device(device))

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
            raise RuntimeError("Spark-TTS model is not loaded — call load() first")
        if not text or not text.strip():
            return

        voice_file = _voice_path(voice)
        with tempfile.TemporaryDirectory(prefix="vox-spark-") as tmpdir:
            prompt_speech_path: Path | None = None
            if reference_audio is not None:
                prompt_speech_path = Path(tmpdir) / "reference.wav"
                _write_reference_audio(prompt_speech_path, reference_audio, SPARK_SAMPLE_RATE)
            elif voice_file is not None:
                prompt_speech_path = voice_file

            if prompt_speech_path is not None:
                wav = self._model.inference(
                    text,
                    prompt_speech_path=prompt_speech_path,
                    prompt_text=reference_text,
                )
            else:
                wav = self._model.inference(
                    text,
                    gender="female",
                    pitch="moderate",
                    speed=_speed_label(speed),
                )

        audio = _audio_array(wav)
        chunk_size = SPARK_SAMPLE_RATE * 2
        for i in range(0, len(audio), chunk_size):
            yield SynthesizeChunk(
                audio=audio[i : i + chunk_size].tobytes(),
                sample_rate=SPARK_SAMPLE_RATE,
                is_final=False,
            )

        yield SynthesizeChunk(audio=b"", sample_rate=SPARK_SAMPLE_RATE, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="reference",
                name="Reference audio",
                language=None,
                description="Pass reference_audio/reference_text or a voice path.",
                is_cloned=True,
            ),
            VoiceInfo(
                id="female",
                name="Female generated voice",
                language=None,
                description="Spark-TTS controllable generated voice",
            ),
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 6_000_000_000
