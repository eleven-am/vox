from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from vox.core.adapter import STTAdapter
from vox.core.adapter_runtime import install_target_runtime_requirements
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.errors import ModelLoadError
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    TranscribeResult,
    TranscriptSegment,
    WordTimestamp,
)
from vox.core.worker_host import WorkerHost
from vox_parakeet.nemo_worker import RUNTIME_IMPORT_ERROR_MARKER

logger = logging.getLogger(__name__)

PARAKEET_SAMPLE_RATE = 16_000
DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_VRAM_BYTES = 2_500_000_000
DEFAULT_STARTUP_TIMEOUT_SECONDS = 1800.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
STARTUP_TIMEOUT_ENV = "VOX_PARAKEET_STARTUP_TIMEOUT_S"
REQUEST_TIMEOUT_ENV = "VOX_PARAKEET_REQUEST_TIMEOUT_S"
WORKER_DEVICE_ENV = "VOX_PARAKEET_DEVICE"
_RUNTIME_SENTINEL = ".vox-parakeet-nemo-runtime-ready"
_RUNTIME_EXPECTED_RELATIVE_PATHS = (
    Path("nemo") / "__init__.py",
    Path("nemo") / "collections" / "asr" / "__init__.py",
)
_RUNTIME_DEPENDENCIES = (
    "nemo-toolkit[asr]==2.7.3",
    "numpy>=1.26,<2",
    "numba>=0.61,<0.67",
    "llvmlite>=0.44,<0.49",
    "matplotlib>=3.9,<3.10",
)
_WORKER_ENV_NAMES = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
_WORKER_ENV_PREFIXES = ("CUDA_", "NVIDIA_", "HF_", "HUGGINGFACE_", "PYTORCH_")


def _runtime_target_dir() -> Path:
    return vox_runtime_root() / "parakeet-nemo"


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _install_nemo_runtime() -> Path:
    runtime_dir = _runtime_target_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sentinel = runtime_dir / _RUNTIME_SENTINEL
    if sentinel.is_file():
        return runtime_dir

    if not install_target_runtime_requirements(
        runtime_dir,
        _RUNTIME_DEPENDENCIES,
        upgrade=False,
        timeout=1800,
        expected_paths=tuple(runtime_dir / relative for relative in _RUNTIME_EXPECTED_RELATIVE_PATHS),
        installer_order=("uv",),
        install_runner=_run_install_command,
        context="Parakeet NeMo runtime install",
    ):
        raise RuntimeError(
            "Failed to install Parakeet NeMo runtime dependencies."
        )
    sentinel.touch()
    return runtime_dir


def _resolve_model_ref(model_path: str, source: str | None) -> tuple[str, Path | None]:
    if source:
        return source, None

    path = Path(model_path)
    if path.exists():
        if path.is_file():
            return str(path), path if path.suffix == ".nemo" else None

        nemo_files = sorted(path.rglob("*.nemo"))
        if nemo_files:
            return str(path), nemo_files[0]

    return model_path, None


def _to_numpy_audio(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _worker_python_paths(runtime_dir: Path) -> list[str]:
    import vox

    candidates = [
        str(runtime_dir),
        str(Path(vox.__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[1]),
    ]
    paths: list[str] = []
    for candidate in candidates:
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _worker_env(runtime_dir: Path, device: str) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if name in _WORKER_ENV_NAMES or name.startswith(_WORKER_ENV_PREFIXES)
    }
    env["PYTHONPATH"] = os.pathsep.join(_worker_python_paths(runtime_dir))
    env[WORKER_DEVICE_ENV] = device
    return env


def _worker_argv(model_id: str, checkpoint_path: Path | None) -> list[str]:
    argv = [sys.executable, "-m", "vox_parakeet.nemo_worker", "--model-id", model_id]
    if checkpoint_path is not None:
        argv.extend(["--checkpoint", str(checkpoint_path)])
    return argv


def _env_timeout_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {raw!r}")
    return value


class ParakeetNemoAdapter(STTAdapter):
    def __init__(self) -> None:
        self._host: WorkerHost | None = None
        self._model_id: str = DEFAULT_MODEL_ID
        self._device: str = "cuda"
        self._lock = threading.RLock()

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="parakeet-stt-nemo",
            type=ModelType.STT,
            architectures=("parakeet-stt-nemo", "parakeet", "parakeet-nemo", "parakeet-tdt"),
            default_sample_rate=PARAKEET_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_word_timestamps=True,
            supports_language_detection=True,
            supported_languages=(),
        )

    def prepare_runtime(self) -> None:
        _install_nemo_runtime()

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        with self._lock:
            if self._host is not None:
                if self._host.alive:
                    return
                self._host.close()
                self._host = None

            if device not in ("cuda", "auto"):
                raise RuntimeError(
                    "Parakeet NeMo is a CUDA-backed adapter and requires device='cuda' or 'auto'"
                )
            self._device = "cuda"
            source = kwargs.pop("_source", None)
            self._model_id, checkpoint_path = _resolve_model_ref(model_path, source)
            runtime_dir = _install_nemo_runtime()

            logger.info("Loading Parakeet NeMo model: %s (device=%s)", self._model_id, self._device)
            start = time.perf_counter()
            try:
                self._host = WorkerHost(
                    _worker_argv(self._model_id, checkpoint_path),
                    env=_worker_env(runtime_dir, self._device),
                    name="parakeet-nemo",
                    startup_timeout=_env_timeout_seconds(STARTUP_TIMEOUT_ENV, DEFAULT_STARTUP_TIMEOUT_SECONDS),
                )
            except ModelLoadError as error:
                if RUNTIME_IMPORT_ERROR_MARKER in str(error):
                    logger.warning(
                        "Parakeet NeMo runtime failed to import inside the worker; clearing runtime sentinel "
                        "so the next load() reinstalls it"
                    )
                    (runtime_dir / _RUNTIME_SENTINEL).unlink(missing_ok=True)
                raise
            elapsed = time.perf_counter() - start
            logger.info("Parakeet NeMo model loaded in %.2fs", elapsed)

    def unload(self) -> None:
        with self._lock:
            host = self._host
            self._host = None
            self._model_id = DEFAULT_MODEL_ID
            self._device = "cuda"
            if host is not None:
                host.close()
        logger.info("Parakeet NeMo adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        host = self._host
        return host is not None and host.alive

    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> TranscribeResult:
        host = self._host
        if host is None or not host.alive:
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

        temp_write_start = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, PARAKEET_SAMPLE_RATE)
            temp_path = Path(tmp.name)
        temp_write_ms = int((time.perf_counter() - temp_write_start) * 1000)

        try:
            request_start = time.perf_counter()
            response = host.request(
                {"op": "transcribe", "path": str(temp_path), "word_timestamps": word_timestamps},
                timeout=_env_timeout_seconds(REQUEST_TIMEOUT_ENV, DEFAULT_REQUEST_TIMEOUT_SECONDS),
            )
            request_ms = int((time.perf_counter() - request_start) * 1000)
        finally:
            temp_path.unlink(missing_ok=True)

        text = str(response["text"])
        detected_language = language or response.get("language")

        if word_timestamps:
            words = tuple(
                WordTimestamp(
                    word=str(entry["word"]),
                    start_ms=int(entry["start_ms"]),
                    end_ms=int(entry["end_ms"]),
                )
                for entry in response.get("words") or ()
            )
            segments = (
                (
                    TranscriptSegment(
                        text=text,
                        start_ms=0,
                        end_ms=duration_ms,
                        words=words,
                        language=detected_language,
                    ),
                )
                if text
                else ()
            )
        else:
            segments = (
                (
                    TranscriptSegment(
                        text=text,
                        start_ms=0,
                        end_ms=duration_ms,
                        language=detected_language,
                    ),
                )
                if text
                else ()
            )

        logger.info(
            "Parakeet NeMo transcribed %dms audio temp_write_ms=%d request_ms=%d chars=%d",
            duration_ms,
            temp_write_ms,
            request_ms,
            len(text),
        )
        return TranscribeResult(
            text=text,
            segments=segments,
            language=detected_language,
            duration_ms=duration_ms,
            model=self._model_id,
        )

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return DEFAULT_VRAM_BYTES
