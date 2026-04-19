from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo

logger = logging.getLogger(__name__)

XTTS_SAMPLE_RATE = 24_000
XTTS_MODEL_NAME = "coqui/XTTS-v2"


def _runtime_root() -> Path:
    vox_home = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))
    return vox_home / "runtime" / "xtts"


def _clear_modules(prefixes: tuple[str, ...]) -> None:
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _ensure_runtime_path() -> str:
    runtime_root = _runtime_root()
    runtime_path = str(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    fallback_file = runtime_root / "_vox_runtime_fallback_paths.pth"
    fallback_file.write_text(f"{sysconfig.get_paths()['purelib']}\n", encoding="utf-8")
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    return runtime_path


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
            "Failed to bootstrap pip for the XTTS runtime install. "
            f"stderr: {result.stderr.strip()}"
        )


def _install_xtts_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    _ensure_pip_available()
    installers = [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            runtime_path,
            "coqui-tts>=0.27.5",
            "accelerate>=1.10.0,<2.0.0",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            runtime_path,
            "coqui-tts>=0.27.5",
            "accelerate>=1.10.0,<2.0.0",
        ],
    ]
    last_error = ""
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
            logger.info("Bootstrapped XTTS runtime into %s", runtime_path)
            _clear_modules(("TTS", "coqpit", "trainer", "transformers", "tokenizers", "accelerate"))
            return
        last_error = result.stderr.strip() or result.stdout.strip()
        logger.warning("%s failed: %s", " ".join(installer), last_error)

    raise RuntimeError(
        "XTTS-v2 requires the Coqui TTS runtime package. "
        "Install `coqui-tts>=0.27.5` plus PyTorch in the image. "
        f"Last runtime install error: {last_error}"
    )


def _load_torch_runtime() -> Any | None:
    try:
        import torch as torch_module

        return torch_module
    except ImportError:
        return None


def _patch_transformers_runtime() -> None:
    try:
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:
        return

    if hasattr(pytorch_utils, "isin_mps_friendly"):
        return

    torch_runtime = _load_torch_runtime()
    if torch_runtime is not None and hasattr(torch_runtime, "isin"):
        pytorch_utils.isin_mps_friendly = torch_runtime.isin
        return

    def _numpy_fallback(elements, test_elements):
        return np.isin(np.asarray(elements), np.asarray(test_elements))

    pytorch_utils.isin_mps_friendly = _numpy_fallback


def _select_device(device: str) -> str:
    torch = _load_torch_runtime()
    if torch is None:
        return "cpu"
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_tts_runtime() -> type[Any]:
    _ensure_runtime_path()
    _patch_transformers_runtime()
    try:
        from TTS.api import TTS

        return TTS
    except ImportError:
        _install_xtts_runtime()
        _patch_transformers_runtime()
        try:
            from TTS.api import TTS

            return TTS
        except ImportError as exc:
            raise RuntimeError(
                "XTTS-v2 requires the Coqui TTS runtime package. "
                "Install `coqui-tts>=0.27.5` plus PyTorch in the image."
            ) from exc


def _extract_audio(output: Any) -> np.ndarray:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("wav", output.get("audio", output))
    if hasattr(output, "detach") and hasattr(output, "cpu") and hasattr(output, "numpy"):
        output = output.detach().cpu().numpy()
    return np.asarray(output, dtype=np.float32).reshape(-1)


def _write_reference_audio(reference_audio: NDArray[np.float32], base_dir: Path) -> Path:
    ref_path = base_dir / "reference.wav"
    sf.write(str(ref_path), np.asarray(reference_audio, dtype=np.float32), XTTS_SAMPLE_RATE)
    return ref_path


def _inference_context(torch_runtime: Any | None):
    if torch_runtime is None:
        return nullcontext()
    inference_mode = getattr(torch_runtime, "inference_mode", None)
    if not callable(inference_mode):
        return nullcontext()
    context = inference_mode()
    if not callable(getattr(context, "__enter__", None)) or not callable(getattr(context, "__exit__", None)):
        return nullcontext()
    return context


class XTTSAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._tts: Any = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="xtts-tts-torch",
            type=ModelType.TTS,
            architectures=("xtts-tts-torch", "xtts", "xtts-v2"),
            default_sample_rate=XTTS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_voice_cloning=True,
            supported_languages=(),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = _select_device(device)

        TTS = _load_tts_runtime()
        logger.info("Loading XTTS model: %s (device=%s)", self._model_id, self._device)
        start = time.perf_counter()

        self._tts = TTS(model_name=self._model_id, progress_bar=False)
        if hasattr(self._tts, "to"):
            self._tts = self._tts.to(self._device)
        if hasattr(self._tts, "eval"):
            self._tts.eval()

        elapsed = time.perf_counter() - start
        logger.info("XTTS model loaded in %.2fs", elapsed)
        self._loaded = True

    def unload(self) -> None:
        self._tts = None
        self._loaded = False
        torch = _load_torch_runtime()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("XTTS adapter unloaded")

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
        if not self._loaded or self._tts is None:
            raise RuntimeError("XTTS model is not loaded — call load() first")

        if not text or not text.strip():
            return

        if reference_text is not None:
            logger.warning("XTTS ignores reference_text; XTTS-v2 only needs a reference wav")

        speaker_wav: str | None = None
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            if reference_audio is not None:
                temp_dir = tempfile.TemporaryDirectory(prefix="vox-xtts-")
                speaker_wav = str(_write_reference_audio(reference_audio, Path(temp_dir.name)))
            elif voice:
                candidate = Path(voice).expanduser()
                if candidate.is_file():
                    speaker_wav = str(candidate)
                else:
                    raise ValueError(
                        "XTTS requires voice cloning input as a speaker wav path or reference_audio; "
                        f"got voice={voice!r}"
                    )
            else:
                raise ValueError(
                    "XTTS requires reference_audio or a voice wav path; the XTTS-v2 checkpoint is voice-cloning only"
                )

            language_code = (language or "en").strip() or "en"

            torch = _load_torch_runtime()
            context = _inference_context(torch)
            try:
                with context:
                    audio = self._tts.tts(
                        text=text,
                        speaker_wav=speaker_wav,
                        language=language_code,
                        split_sentences=True,
                    )
            except TypeError:
                audio = self._tts.tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language_code,
                    split_sentences=True,
                )

            wav = _extract_audio(audio)
            chunk_size = XTTS_SAMPLE_RATE * 2
            for i in range(0, len(wav), chunk_size):
                chunk = wav[i:i + chunk_size]
                yield SynthesizeChunk(
                    audio=chunk.tobytes(),
                    sample_rate=XTTS_SAMPLE_RATE,
                    is_final=False,
                )

            yield SynthesizeChunk(audio=b"", sample_rate=XTTS_SAMPLE_RATE, is_final=True)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def list_voices(self) -> list[VoiceInfo]:
        return []

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        model_id = kwargs.get("_source") or kwargs.get("model_id") or self._model_id
        if "xtts" in str(model_id).lower():
            return 4_000_000_000
        return 4_000_000_000
