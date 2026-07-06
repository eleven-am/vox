from __future__ import annotations

import importlib
import inspect
import logging
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator, Callable
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
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesisParameterInfo, SynthesizeChunk, VoiceInfo
from vox.operations.errors import InvalidConfigError

logger = logging.getLogger(__name__)

INDEXTTS_SAMPLE_RATE = 24_000
_EMOTION_PARAM_NAMES = (
    "emotion_happy",
    "emotion_angry",
    "emotion_sad",
    "emotion_afraid",
    "emotion_disgusted",
    "emotion_melancholic",
    "emotion_surprised",
    "emotion_calm",
)
_EMOTION_VECTOR_MAX_SUM = 1.5
INDEXTTS_RUNTIME_PACKAGE = "git+https://github.com/index-tts/index-tts.git"
INDEXTTS_RUNTIME_DEPS = (
    "accelerate==1.8.1",
    "cn2an==0.5.22",
    "cython==3.0.7",
    "descript-audiotools==0.7.2",
    "einops>=0.8.1,<1",
    "ffmpeg-python==0.2.0",
    "g2p-en==2.1.0",
    "jieba==0.42.1",
    "json5==0.10.0",
    "keras==2.9.0",
    "librosa==0.10.2.post1",
    "matplotlib>=3.10,<3.11",
    "modelscope==1.27.0",
    "munch==4.0.0",
    "numba>=0.61,<0.63",
    "numpy>=2.0,<2.4",
    "omegaconf>=2.3.0,<3",
    "opencv-python==4.9.0.80",
    "pandas==2.3.2",
    "protobuf==3.19.6",
    "safetensors==0.5.2",
    "sentencepiece>=0.2.1,<0.3",
    "tensorboard==2.9.1",
    "textstat>=0.7.10,<1",
    "tokenizers==0.21.0",
    "transformers==4.52.1",
    "wetext>=0.0.9; sys_platform != 'linux'",
    "WeTextProcessing; sys_platform == 'linux'",
)
_RUNTIME_PROBE_ERRORS = (ImportError, ModuleNotFoundError, AttributeError, ValueError)
_FORBIDDEN_RUNTIME_PACKAGE_GLOBS = (
    "torch",
    "torch-*.dist-info",
    "torchaudio",
    "torchaudio-*.dist-info",
    "torchvision",
    "torchvision-*.dist-info",
    "triton",
    "triton-*.dist-info",
    "nvidia",
    "nvidia_*.dist-info",
    "cuda",
    "cuda_*.dist-info",
)
_STALE_RUNTIME_REPAIR_GLOBS = (
    "matplotlib",
    "matplotlib-*.dist-info",
    "matplotlib.libs",
    "numpy",
    "numpy-*.dist-info",
    "numpy.libs",
)


def _runtime_root() -> Path:
    return vox_runtime_root() / "indextts"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_app_fallback_path(runtime_dir)
    runtime_path = activate_runtime_path(runtime_dir, root=runtime_dir.parent)
    _apply_numpy_compatibility()
    return runtime_path


def _apply_numpy_compatibility() -> None:
    # TensorBoard 2.9 still references np.bool8. NumPy 2 removed that alias,
    # and reloading a different target-runtime NumPy inside the server process
    # can leave mixed submodules behind. Keep the active NumPy stable instead.
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_


def _remove_forbidden_runtime_packages() -> None:
    runtime_dir = _runtime_root()
    for pattern in _FORBIDDEN_RUNTIME_PACKAGE_GLOBS:
        for path in runtime_dir.glob(pattern):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Failed to remove stale IndexTTS runtime path %s: %s", path, exc)


def _remove_stale_runtime_repair_targets() -> None:
    runtime_dir = _runtime_root()
    for pattern in _STALE_RUNTIME_REPAIR_GLOBS:
        for path in runtime_dir.glob(pattern):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                logger.info("Removed stale IndexTTS runtime path before repair: %s", path.name)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Failed to remove stale IndexTTS runtime path %s: %s", path, exc)


def _run_install_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _install_indextts_runtime() -> None:
    runtime_path = _ensure_runtime_path()
    _remove_forbidden_runtime_packages()
    _remove_stale_runtime_repair_targets()
    _clear_indextts_modules()
    if not install_target_runtime_requirements(
        runtime_path,
        (INDEXTTS_RUNTIME_PACKAGE,),
        no_deps=True,
        upgrade=False,
        timeout=1200,
        install_runner=_run_install_command,
        context="IndexTTS runtime package install",
    ):
        raise RuntimeError("Failed to install IndexTTS runtime package from GitHub.")
    if not install_target_runtime_requirements(
        runtime_path,
        INDEXTTS_RUNTIME_DEPS,
        timeout=1200,
        install_runner=_run_install_command,
        context="IndexTTS runtime dependency install",
    ):
        raise RuntimeError("Failed to install IndexTTS runtime dependencies.")


def _clear_indextts_modules() -> None:
    purge_runtime_modules((
        "indextts",
        "audiotools",
        "transformers",
        "tokenizers",
        "accelerate",
        "modelscope",
        "tensorboard",
        "torch.utils.tensorboard",
        "google",
    ))


def _indextts_class_from_runtime() -> type[Any] | None:
    runtime_path = Path(_ensure_runtime_path()).resolve()
    module = importlib.import_module("indextts.infer_v2")
    if not _module_loaded_from_runtime(module, runtime_path):
        return None
    cls = getattr(module, "IndexTTS2", None)
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


def _load_indextts_class() -> type[Any]:
    _ensure_runtime_path()
    _remove_forbidden_runtime_packages()
    _clear_indextts_modules()
    try:
        cls = _indextts_class_from_runtime()
    except _RUNTIME_PROBE_ERRORS:
        cls = None
    if cls is not None:
        return cls

    _install_indextts_runtime()
    _clear_indextts_modules()
    try:
        cls = _indextts_class_from_runtime()
    except _RUNTIME_PROBE_ERRORS as exc:
        raise RuntimeError(
            "IndexTTS runtime is installed, but indextts.infer_v2.IndexTTS2 could not be imported."
        ) from exc
    if cls is not None:
        return cls

    raise RuntimeError("IndexTTS runtime is installed, but indextts.infer_v2.IndexTTS2 was not found.")


def _voice_path(voice: str | None) -> str | None:
    if not voice:
        return None
    path = Path(voice).expanduser()
    return str(path) if path.is_file() else None


def _write_reference_audio(path: Path, reference_audio: NDArray[np.float32], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(reference_audio, dtype=np.float32), sample_rate)


def _read_audio(path: Path) -> tuple[NDArray[np.float32], int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sample_rate)


def _audio_from_result(result: Any, fallback_path: Path) -> tuple[NDArray[np.float32], int]:
    if fallback_path.is_file():
        return _read_audio(fallback_path)
    if isinstance(result, str | Path):
        return _read_audio(Path(result))
    if isinstance(result, dict):
        for key in ("audio_path", "wav_path", "output_path"):
            value = result.get(key)
            if isinstance(value, str | Path):
                return _read_audio(Path(value))
        for key in ("audio", "wav", "waveform"):
            if key in result:
                return _audio_array(result[key]), INDEXTTS_SAMPLE_RATE
    if result is not None:
        return _audio_array(result), INDEXTTS_SAMPLE_RATE
    raise RuntimeError("IndexTTS produced no audio.")


def _audio_array(audio: Any) -> NDArray[np.float32]:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise RuntimeError("IndexTTS produced no audio.")
    return array


def _audio_array_for_save(audio: Any) -> NDArray[Any]:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    array = np.asarray(audio)
    if array.ndim == 2 and array.shape[0] <= 8:
        array = array.T
    return array


def _patch_torchaudio_save() -> None:
    try:
        torchaudio = importlib.import_module("torchaudio")
    except ModuleNotFoundError:
        return
    current_save = torchaudio.save
    if getattr(current_save, "_vox_indextts_soundfile_patch", False):
        return

    original_save = current_save

    def save_with_soundfile(uri: Any, src: Any, sample_rate: int, *args: Any, **kwargs: Any) -> Any:
        if isinstance(uri, str | Path):
            array = _audio_array_for_save(src)
            subtype = "PCM_16" if array.dtype == np.int16 else None
            sf.write(str(uri), array, int(sample_rate), subtype=subtype)
            return None
        return original_save(uri, src, sample_rate, *args, **kwargs)

    save_with_soundfile._vox_indextts_soundfile_patch = True  # type: ignore[attr-defined]
    torchaudio.save = save_with_soundfile


def _require_cuda_device(device: str) -> None:
    if device == "cuda":
        return
    raise RuntimeError(
        "IndexTTS requires a Linux x86_64 CUDA runtime. "
        "CPU, ONNX, and Spark/ARM NVIDIA execution are not production-supported "
        "by this adapter."
    )


def _candidate_model_configs(model_root: Path) -> list[Path]:
    candidates = [
        model_root / "config.yaml",
        model_root / "config.yml",
        model_root / "indextts2.yaml",
        model_root / "checkpoints" / "config.yaml",
    ]
    return [candidate for candidate in candidates if candidate.is_file()]


def _constructor_accepts(cls: type[Any], *args: Any, **kwargs: Any) -> bool:
    try:
        inspect.signature(cls).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return False
    return True


def _construct_model(cls: type[Any], model_path: Path, device: str) -> Any:
    cfg_candidates = _candidate_model_configs(model_path)
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for cfg_path in cfg_candidates:
        attempts.append((
            (),
            {
                "cfg_path": str(cfg_path),
                "model_dir": str(model_path),
                "device": device,
                "use_fp16": device == "cuda",
                "use_cuda_kernel": False,
                "use_deepspeed": False,
            },
        ))
        attempts.append((
            (),
            {
                "cfg_path": str(cfg_path),
                "model_dir": str(model_path),
                "device": device,
            },
        ))
        attempts.append((
            (str(cfg_path), str(model_path)),
            {"device": device},
        ))
    attempts.extend(
        (
            ((), {"model_dir": str(model_path), "device": device}),
            ((str(model_path),), {"device": device}),
            ((str(model_path),), {}),
        )
    )

    rejected: list[str] = []
    for args, kwargs in attempts:
        if not _constructor_accepts(cls, *args, **kwargs):
            rejected.append(f"args={args!r} kwargs={kwargs!r}")
            continue
        return cls(*args, **kwargs)

    raise RuntimeError("Could not initialize IndexTTS2 with the available constructor signatures.") from (
        TypeError("; ".join(rejected)) if rejected else None
    )


def _emotion_vector_from_params(params: dict[str, Any] | None) -> list[float] | None:
    if not params or not any(name in params for name in _EMOTION_PARAM_NAMES):
        return None
    vector = [float(params.get(name, 0.0)) for name in _EMOTION_PARAM_NAMES]
    total = sum(vector)
    if total > _EMOTION_VECTOR_MAX_SUM:
        raise InvalidConfigError(
            "IndexTTS emotion_* parameters must sum to "
            f"{_EMOTION_VECTOR_MAX_SUM} or less; got {total:.3f}"
        )
    return vector


def _emo_audio_prompt_from_params(params: dict[str, Any] | None) -> str | None:
    if not params or "emo_audio_prompt" not in params:
        return None
    path_value = str(params["emo_audio_prompt"]).strip()
    if not path_value:
        raise InvalidConfigError("IndexTTS emo_audio_prompt must be a non-empty audio file path")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise InvalidConfigError(f"IndexTTS emo_audio_prompt does not exist or is not a file: {path_value}")
    return str(path)


def _inference_kwargs_from_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}

    kwargs: dict[str, Any] = {}
    emotion_vector = _emotion_vector_from_params(params)
    if emotion_vector is not None:
        kwargs["emo_vector"] = emotion_vector
    emo_audio_prompt = _emo_audio_prompt_from_params(params)
    if emo_audio_prompt is not None:
        kwargs["emo_audio_prompt"] = emo_audio_prompt
    if "emo_alpha" in params:
        kwargs["emo_alpha"] = float(params["emo_alpha"])
    if "use_emo_text" in params:
        kwargs["use_emo_text"] = bool(params["use_emo_text"])
    if "emo_text" in params:
        kwargs["emo_text"] = str(params["emo_text"])
        kwargs.setdefault("use_emo_text", True)
    if "use_random" in params:
        kwargs["use_random"] = bool(params["use_random"])
    return kwargs


def _infer_to_file(
    model: Any,
    text: str,
    reference_path: str,
    output_path: Path,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    _patch_torchaudio_save()
    inference_kwargs = _inference_kwargs_from_params(params)
    attempts: list[Callable[[], Any]] = [
        lambda: model.infer(
            spk_audio_prompt=reference_path,
            text=text,
            output_path=str(output_path),
            **inference_kwargs,
        ),
        lambda: model.infer(
            audio_prompt=reference_path,
            text=text,
            output_path=str(output_path),
            **inference_kwargs,
        ),
        lambda: model.infer(
            text=text,
            audio_prompt=reference_path,
            output_path=str(output_path),
            **inference_kwargs,
        ),
    ]
    if not inference_kwargs:
        attempts.append(lambda: model.infer(reference_path, text, str(output_path)))
    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))
    raise RuntimeError("Could not call IndexTTS2.infer with the supported adapter signatures.") from TypeError(
        "; ".join(errors)
    )


class IndexTTSAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._device = "cpu"
        self._sample_rate = INDEXTTS_SAMPLE_RATE

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="indextts-tts-torch",
            type=ModelType.TTS,
            architectures=("indextts-tts-torch", "indextts2", "indextts"),
            default_sample_rate=INDEXTTS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_voice_cloning=True,
            supported_languages=("en", "zh"),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._model is not None:
            return

        _require_cuda_device(device)

        kwargs.pop("_source", None)
        self._device = device
        cls = _load_indextts_class()
        logger.info("Loading IndexTTS2 runtime from %s (device=%s)", model_path, self._device)
        self._model = _construct_model(cls, Path(model_path), self._device)

    def unload(self) -> None:
        self._model = None
        self._device = "cpu"
        self._sample_rate = INDEXTTS_SAMPLE_RATE

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def prepare_runtime(self) -> None:
        _load_indextts_class()

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        _emotion_vector_from_params(params)
        _emo_audio_prompt_from_params(params)
        if reference_audio is not None:
            if np.asarray(reference_audio, dtype=np.float32).size == 0:
                raise InvalidConfigError("IndexTTS reference_audio is empty")
            return
        if _voice_path(voice) is not None:
            return
        raise InvalidConfigError("IndexTTS requires reference_audio or a voice path for speaker cloning")

    def synthesis_parameters(self) -> tuple[SynthesisParameterInfo, ...]:
        return (
            SynthesisParameterInfo(
                name="emo_alpha",
                type="number",
                default=None,
                min_value=0.0,
                max_value=1.0,
                description="IndexTTS2 emotion-conditioning strength. None uses the backend default.",
            ),
            SynthesisParameterInfo(
                name="use_emo_text",
                type="boolean",
                default=False,
                description="Infer emotion from the synthesis text when true.",
            ),
            SynthesisParameterInfo(
                name="emo_text",
                type="string",
                default=None,
                description="Separate emotion description text; implies use_emo_text=true.",
            ),
            SynthesisParameterInfo(
                name="emo_audio_prompt",
                type="string",
                default=None,
                description="Server-side audio file path used as the IndexTTS2 emotional reference prompt.",
            ),
            SynthesisParameterInfo(
                name="use_random",
                type="boolean",
                default=False,
                description="Enable IndexTTS2 stochastic emotion sampling; may reduce cloning fidelity.",
            ),
            *(
                SynthesisParameterInfo(
                    name=name,
                    type="number",
                    default=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    description="IndexTTS2 emotion-vector dial; all emotion_* values must sum to 1.5 or less.",
                )
                for name in _EMOTION_PARAM_NAMES
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
            raise RuntimeError("IndexTTS model is not loaded — call load() first")
        if not text or not text.strip():
            return

        voice_file = _voice_path(voice)
        if reference_audio is None and voice_file is None:
            raise ValueError("IndexTTS requires reference_audio or a voice path for speaker cloning.")

        with tempfile.TemporaryDirectory(prefix="vox-indextts-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            if reference_audio is not None:
                ref_path = tmpdir_path / "reference.wav"
                _write_reference_audio(ref_path, reference_audio, self._sample_rate)
                reference_path = str(ref_path)
            else:
                assert voice_file is not None
                reference_path = voice_file

            output_path = tmpdir_path / "output.wav"
            result = _infer_to_file(self._model, text, reference_path, output_path, params=params)
            audio, sample_rate = _audio_from_result(result, output_path)

        chunk_size = sample_rate * 2
        for i in range(0, len(audio), chunk_size):
            yield SynthesizeChunk(
                audio=audio[i:i + chunk_size].tobytes(),
                sample_rate=sample_rate,
                is_final=False,
            )

        yield SynthesizeChunk(audio=b"", sample_rate=sample_rate, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="reference",
                name="Reference audio",
                language=None,
                description="Pass reference_audio or a voice path to clone a speaker.",
                is_cloned=True,
            )
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 6_000_000_000
