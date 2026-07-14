from __future__ import annotations

import importlib
import logging
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from vox.audio.resampler import resample
from vox.core.adapter import TTSAdapter
from vox.core.adapter_runtime import (
    activate_runtime_path,
    install_target_runtime_requirements,
    purge_runtime_modules,
)
from vox.core.adapter_runtime import (
    runtime_root as vox_runtime_root,
)
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesisParameterInfo, SynthesizeChunk, VoiceInfo
from vox.operations.errors import InvalidConfigError

logger = logging.getLogger(__name__)

COSYVOICE_SAMPLE_RATE = 24_000
COSYVOICE_REPO = "https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_SOURCE_REF = "v2.0"
COSYVOICE_SOURCE_DIR = "CosyVoice"
_DEFAULT_REFERENCE_PROMPT_SECONDS = 5.0
_DEFAULT_TEXT_FRONTEND = False
_DEFAULT_STREAM = False
_DEFAULT_FP16 = False
COSYVOICE_REQUIRED_PATHS = (
    Path("cosyvoice") / "cli" / "cosyvoice.py",
    Path("third_party") / "Matcha-TTS" / "matcha",
)
_RUNTIME_PROBE_ERRORS = (ImportError, ModuleNotFoundError, AttributeError, ValueError)

# Keep this list intentionally narrower than upstream requirements.txt. The Vox
# image owns the shared GPU stack (torch, torchaudio, CUDA libraries,
# transformers, onnxruntime, FastAPI, etc.); installing those here can downgrade
# or duplicate the server runtime. These packages are the CosyVoice-specific
# Python pieces needed around the model/runtime source checkout.
COSYVOICE_RUNTIME_REQUIREMENTS = (
    "conformer==0.3.2",
    "diffusers==0.29.0",
    "gdown==5.1.0",
    "hydra-core==1.3.2",
    "HyperPyYAML==1.2.3",
    "huggingface-hub>=0.34,<1.0",
    "inflect==7.3.1",
    "librosa==0.10.2",
    "lightning==2.2.4",
    "modelscope==1.20.0",
    "numpy==1.26.4",
    "omegaconf==2.3.0",
    "protobuf>=4.25,<5",
    "pyworld==0.3.4",
    "rich==13.7.1",
    "tiktoken>=0.7,<1",
    "wetext==0.0.4",
    "wget==3.2",
    "x-transformers==2.11.24",
)

WHISPER_COMPAT_FILES = {
    "__init__.py": '''
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def _load_audio(audio, device):
    if isinstance(audio, (str, Path)):
        waveform, sample_rate = torchaudio.load(str(audio))
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        audio = waveform
    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    if audio.ndim == 2:
        audio = audio if audio.shape[0] == 1 else audio.mean(dim=0, keepdim=True)
    return audio.to(device=device, dtype=torch.float32)


def log_mel_spectrogram(audio, n_mels=128, padding=0, device=None):
    device = device or (audio.device if torch.is_tensor(audio) else "cpu")
    waveform = _load_audio(audio, device)
    if padding > 0:
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
        center=True,
        power=2.0,
    ).to(device)
    mel = transform(waveform)
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
'''.lstrip(),
    "tokenizer.py": '''
from __future__ import annotations


class Tokenizer:
    def __init__(self, encoding, num_languages=99, language=None, task=None):
        self.encoding = encoding
        self.num_languages = num_languages
        self.language = language
        self.task = task

    def encode(self, text, **kwargs):
        allowed_special = kwargs.get("allowed_special", "all")
        return self.encoding.encode(text, allowed_special=allowed_special)

    def decode(self, tokens):
        return self.encoding.decode(tokens)
'''.lstrip(),
}

MATPLOTLIB_COMPAT_FILES = {
    "__init__.py": '''
from __future__ import annotations

from . import axes, colors

__version__ = "0.0.vox-compat"


def use(*args, **kwargs):
    return None
'''.lstrip(),
    "axes.py": '''
from __future__ import annotations


class Axes:
    pass
'''.lstrip(),
    "colors.py": '''
from __future__ import annotations


class Colormap:
    pass


def is_color_like(value):
    return True
'''.lstrip(),
    "pyplot.py": '''
from __future__ import annotations

from contextlib import contextmanager


class Figure:
    pass


class _Style:
    @contextmanager
    def context(self, *args, **kwargs):
        yield


style = _Style()


def subplots(*args, **kwargs):
    raise ModuleNotFoundError("CosyVoice inference runtime does not include matplotlib plotting support.")
'''.lstrip(),
    "pylab.py": '''
from __future__ import annotations

from .pyplot import *  # noqa: F403
'''.lstrip(),
}


def _runtime_root() -> Path:
    return vox_runtime_root() / "cosyvoice"


def _source_root() -> Path:
    return _runtime_root() / COSYVOICE_SOURCE_DIR


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _write_whisper_compat_package(runtime_dir)
    _write_matplotlib_compat_package(runtime_dir)
    runtime_path = activate_runtime_path(runtime_dir, root=runtime_dir.parent)
    for path in reversed((_source_root(), _source_root() / "third_party" / "Matcha-TTS")):
        if path.exists():
            path_value = str(path)
            if path_value in sys.path:
                sys.path.remove(path_value)
            sys.path.insert(0, path_value)
    importlib.invalidate_caches()
    return runtime_path


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _write_whisper_compat_package(runtime_dir: Path) -> None:
    legacy_module = runtime_dir / "whisper.py"
    if legacy_module.exists():
        legacy_module.unlink()
    package_dir = runtime_dir / "whisper"
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in WHISPER_COMPAT_FILES.items():
        file_path = package_dir / relative_path
        if file_path.exists() and file_path.read_text(encoding="utf-8") == content:
            continue
        file_path.write_text(content, encoding="utf-8")


def _write_matplotlib_compat_package(runtime_dir: Path) -> None:
    package_dir = runtime_dir / "matplotlib"
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in MATPLOTLIB_COMPAT_FILES.items():
        file_path = package_dir / relative_path
        if file_path.exists() and file_path.read_text(encoding="utf-8") == content:
            continue
        file_path.write_text(content, encoding="utf-8")


def _source_checkout_complete(source_dir: Path | None = None) -> bool:
    root = source_dir or _source_root()
    return all((root / path).exists() for path in COSYVOICE_REQUIRED_PATHS)


def _source_checkout_matches_ref(source_dir: Path | None = None) -> bool:
    root = source_dir or _source_root()
    if not _source_checkout_complete(root):
        return False
    if not (root / ".git").exists():
        return False
    result = _run_install_command(
        ["git", "-C", str(root), "describe", "--tags", "--exact-match", "HEAD"],
        30,
    )
    return result.returncode == 0 and result.stdout.strip() == COSYVOICE_SOURCE_REF


def _clone_cosyvoice_source() -> None:
    source_dir = _source_root()
    if _source_checkout_matches_ref(source_dir):
        return

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if source_dir.exists():
        shutil.rmtree(source_dir)

    result = _run_install_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            COSYVOICE_SOURCE_REF,
            "--recurse-submodules",
            "--shallow-submodules",
            COSYVOICE_REPO,
            str(source_dir),
        ],
        1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone CosyVoice runtime source: {result.stderr.strip()}")

    if not _source_checkout_complete(source_dir):
        result = _run_install_command(
            ["git", "-C", str(source_dir), "submodule", "update", "--init", "--recursive"],
            900,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to initialize CosyVoice submodules: {result.stderr.strip()}")

    if not _source_checkout_complete(source_dir):
        raise RuntimeError("CosyVoice runtime source checkout is incomplete.")


def _install_cosyvoice_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    _clone_cosyvoice_source()
    if not install_target_runtime_requirements(
        runtime_path,
        COSYVOICE_RUNTIME_REQUIREMENTS,
        timeout=1800,
        install_runner=_run_install_command,
        context="CosyVoice runtime install",
    ):
        raise RuntimeError("Failed to install CosyVoice runtime requirements.")
    _ensure_runtime_path()


def _clear_cosyvoice_modules() -> None:
    purge_runtime_modules(("cosyvoice", "matcha", "wetext"))


def _cosyvoice_class_from_runtime() -> type[Any] | None:
    runtime_path = _runtime_root().resolve()
    module = importlib.import_module("cosyvoice.cli.cosyvoice")
    if not _module_loaded_from_runtime(module, runtime_path):
        return None
    cls = getattr(module, "CosyVoice2", None)
    return cls if isinstance(cls, type) else None


def _module_loaded_from_runtime(module: Any, runtime_path: Path) -> bool:
    candidate_paths: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        candidate_paths.append(Path(module_file))

    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    if spec_origin and spec_origin not in {"built-in", "frozen"}:
        candidate_paths.append(Path(spec_origin))
    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations:
        candidate_paths.extend(Path(path) for path in search_locations)

    return any(_is_relative_to(path, runtime_path) for path in candidate_paths)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
    except (OSError, ValueError):
        return False
    return True


def _load_cosyvoice_class() -> type[Any]:
    _ensure_runtime_path()
    try:
        cls = _cosyvoice_class_from_runtime()
    except _RUNTIME_PROBE_ERRORS:
        cls = None
    if cls is not None:
        return cls

    _install_cosyvoice_runtime()
    _clear_cosyvoice_modules()
    try:
        cls = _cosyvoice_class_from_runtime()
    except _RUNTIME_PROBE_ERRORS as exc:
        raise RuntimeError(
            "CosyVoice runtime is installed, but cosyvoice.cli.cosyvoice.CosyVoice2 could not be imported."
        ) from exc
    if cls is not None:
        return cls

    raise RuntimeError("CosyVoice runtime is installed, but cosyvoice.cli.cosyvoice.CosyVoice2 was not found.")


def _patch_torchaudio_soundfile_loader() -> None:
    try:
        import torch
        import torchaudio
    except ImportError:
        logger.debug("Torch/torchaudio not available; skipping CosyVoice torchaudio soundfile patch")
        return

    if getattr(torchaudio, "_vox_cosyvoice_soundfile_loader", False):
        return

    original_load = torchaudio.load

    def load_with_soundfile(
        uri: Any,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: str | None = None,
        buffer_size: int = 4096,
        backend: str | None = None,
    ) -> tuple[Any, int]:
        if backend == "soundfile" and isinstance(uri, str | Path):
            frames = -1 if num_frames is None or num_frames < 0 else num_frames
            audio, sample_rate = sf.read(
                str(uri),
                start=frame_offset,
                frames=frames,
                dtype="float32" if normalize else "int16",
                always_2d=True,
            )
            if channels_first:
                audio = audio.T
            return torch.from_numpy(np.ascontiguousarray(audio)), int(sample_rate)
        return original_load(
            uri,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=normalize,
            channels_first=channels_first,
            format=format,
            buffer_size=buffer_size,
            backend=backend,
        )

    torchaudio.load = load_with_soundfile
    torchaudio._vox_cosyvoice_soundfile_loader = True


def _voice_path(voice: str | None) -> Path | None:
    if not voice:
        return None
    path = Path(voice).expanduser()
    return path if path.is_file() else None


def _write_reference_audio(path: Path, reference_audio: NDArray[np.float32], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(reference_audio, dtype=np.float32), sample_rate)


def _reference_speech_16k(reference_audio: NDArray[np.float32]) -> Any:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError("CosyVoice reference audio conversion requires the Torch runtime") from exc
    audio = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return torch.zeros((1, 0), dtype=torch.float32)
    audio_16k = resample(audio, COSYVOICE_SAMPLE_RATE, 16_000)
    return torch.from_numpy(np.asarray(audio_16k, dtype=np.float32)).reshape(1, -1)


def _load_voice_file_16k(path: Path) -> Any:
    audio, sample_rate = sf.read(str(path), dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != COSYVOICE_SAMPLE_RATE:
        audio = resample(audio, int(sample_rate), COSYVOICE_SAMPLE_RATE)
    return _reference_speech_16k(audio)


def _reference_prompt_limit_seconds(params: dict[str, Any] | None) -> float:
    if not params:
        return _DEFAULT_REFERENCE_PROMPT_SECONDS
    raw = params.get("reference_prompt_seconds", _DEFAULT_REFERENCE_PROMPT_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_REFERENCE_PROMPT_SECONDS
    return max(1.0, min(value, 12.0))


def _bool_param(params: dict[str, Any] | None, name: str, default: bool) -> bool:
    if not params or name not in params:
        return default
    return bool(params[name])


def _trim_reference_text(reference_text: str | None, ratio: float) -> str | None:
    if reference_text is None or not reference_text.strip():
        return reference_text

    words = reference_text.strip().split()
    if not words:
        return reference_text
    keep = max(1, min(len(words), int(math.ceil(len(words) * ratio))))
    return " ".join(words[:keep]).strip()


def _trim_reference_prompt(
    *,
    reference_audio: NDArray[np.float32],
    reference_text: str | None,
    max_seconds: float,
    sample_rate: int = COSYVOICE_SAMPLE_RATE,
) -> tuple[NDArray[np.float32], str | None]:
    audio = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio, reference_text

    max_samples = max(1, int(sample_rate * max_seconds))
    if audio.size <= max_samples:
        return audio, reference_text

    ratio = max_samples / float(audio.size)
    return audio[:max_samples], _trim_reference_text(reference_text, ratio)


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


def _require_cuda_device(device: str) -> None:
    if device == "cuda":
        return
    raise RuntimeError(
        "CosyVoice2 requires a Linux x86_64 CUDA runtime. "
        "CPU, ONNX, and Spark/ARM NVIDIA execution are not production-supported "
        "by this adapter."
    )


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

        _require_cuda_device(device)

        kwargs.pop("_source", None)
        self._model_id = model_path
        self._device = device
        cls = _load_cosyvoice_class()
        _patch_torchaudio_soundfile_loader()

        logger.info("Loading CosyVoice2 model from %s (device=%s)", model_path, self._device)
        try:
            self._model = cls(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=_DEFAULT_FP16)
        except TypeError:
            self._model = cls(model_path)

    def unload(self) -> None:
        self._model = None
        self._model_id = ""
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def prepare_runtime(self) -> None:
        _load_cosyvoice_class()

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
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

    def synthesis_parameters(self) -> tuple[SynthesisParameterInfo, ...]:
        return (
            SynthesisParameterInfo(
                name="reference_prompt_seconds",
                type="number",
                default=_DEFAULT_REFERENCE_PROMPT_SECONDS,
                min_value=1.0,
                max_value=12.0,
                description="Maximum cloned reference audio seconds used as CosyVoice2 zero-shot prompt.",
            ),
            SynthesisParameterInfo(
                name="text_frontend",
                type="boolean",
                default=_DEFAULT_TEXT_FRONTEND,
                description=(
                    "Enable CosyVoice text normalization. Defaults off to match the upstream reproducibility guidance."
                ),
            ),
            SynthesisParameterInfo(
                name="stream",
                type="boolean",
                default=_DEFAULT_STREAM,
                description=(
                    "Use CosyVoice internal streaming generation. Defaults off for more stable one-shot synthesis."
                ),
            ),
        )

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
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
            prompt_speech: Any = ""
            if reference_audio is not None:
                ref_path = Path(tmpdir) / "reference.wav"
                reference_audio, reference_text = _trim_reference_prompt(
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                    max_seconds=_reference_prompt_limit_seconds(params),
                )
                _write_reference_audio(ref_path, reference_audio, COSYVOICE_SAMPLE_RATE)
                prompt_speech = _reference_speech_16k(reference_audio)
            elif voice_file is not None:
                prompt_speech = _load_voice_file_16k(voice_file)

            outputs = self._model.inference_zero_shot(
                text,
                reference_text or "",
                prompt_speech,
                zero_shot_spk_id=zero_shot_spk_id,
                stream=_bool_param(params, "stream", _DEFAULT_STREAM),
                speed=speed,
                text_frontend=_bool_param(params, "text_frontend", _DEFAULT_TEXT_FRONTEND),
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
