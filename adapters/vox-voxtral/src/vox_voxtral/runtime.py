from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VoxtralRuntimeInfo:
    python_executable: str
    env: dict[str, str]
    stage_configs_path: str
    site_packages: str


VOXTRAL_TIER_SPARK_16GB = "voxtral-tts-16gb"
VOXTRAL_TIER_SMALL_24GB = "voxtral-tts-24gb"
VOXTRAL_TIER_DEFAULT = "voxtral-tts-default"


def voxtral_tts_tier_extras(tier: str) -> dict[str, Any]:
    if tier == VOXTRAL_TIER_SPARK_16GB:
        return {
            "generation_gpu_memory_utilization": 0.62,
            "tokenizer_gpu_memory_utilization": 0.01,
            "generation_max_model_len": 512,
            "tokenizer_max_num_batched_tokens": 4096,
            "tokenizer_max_model_len": 4096,
            "kv_cache_dtype": "fp8",
            "attention_backend": "triton_attn",
            "strip_dtype": True,
            "vram_bytes": 12_000_000_000,
        }
    if tier == VOXTRAL_TIER_SMALL_24GB:
        return {
            "generation_gpu_memory_utilization": 0.4,
            "tokenizer_gpu_memory_utilization": 0.05,
            "generation_max_model_len": 1536,
            "tokenizer_max_num_batched_tokens": 4096,
            "tokenizer_max_model_len": 4096,
            "vram_bytes": 12_000_000_000,
        }
    return {
        "generation_gpu_memory_utilization": 0.4,
        "tokenizer_gpu_memory_utilization": 0.1,
        "generation_max_model_len": 2048,
        "tokenizer_max_num_batched_tokens": 8192,
        "tokenizer_max_model_len": 8192,
        "vram_bytes": 16_000_000_000,
    }


def _vox_home() -> Path:
    return Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))


def _runtime_root() -> Path:
    return _vox_home() / "runtime"


def _runtime_venv() -> Path:
    return _runtime_root() / "voxtral-tts"


def _stt_runtime_dir() -> Path:
    return _runtime_root() / "voxtral-stt"


def _python_bin(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _ensure_venv(venv_dir: Path) -> Path:
    python_bin = _python_bin(venv_dir)
    if python_bin.exists():
        return python_bin

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    result = _run(["uv", "venv", str(venv_dir), "--python", "3.12"], timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create Voxtral runtime venv: {result.stderr.strip()}")
    return python_bin


def _purelib(python_bin: Path) -> Path:
    result = _run(
        [
            str(python_bin),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to inspect Voxtral runtime site-packages: {result.stderr.strip()}")
    return Path(result.stdout.strip())


def _app_purelib() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _module_available(import_name: str) -> bool:
    if import_name in sys.modules:
        return True
    try:
        return find_spec(import_name) is not None
    except (ImportError, ValueError):
        return True


def _clear_stt_runtime_modules() -> None:
    for module_name in list(sys.modules):
        if module_name in {
            "transformers",
            "huggingface_hub",
            "mistral_common",
            "tokenizers",
        } or module_name.startswith(
            (
                "transformers.",
                "huggingface_hub.",
                "mistral_common.",
                "tokenizers.",
            )
        ):
            sys.modules.pop(module_name, None)


def _build_env(
    python_bin: Path,
    *,
    extra_pythonpaths: list[str] | None = None,
    include_app_torch_lib: bool = True,
) -> dict[str, str]:
    env = os.environ.copy()

    pythonpath_parts: list[str] = []
    if extra_pythonpaths:
        pythonpath_parts.extend(extra_pythonpaths)
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    if pythonpath_parts:
        env["PYTHONPATH"] = ":".join(part for part in pythonpath_parts if part)

    ld_library_parts: list[str] = []
    runtime_torch_lib = _purelib(python_bin) / "torch" / "lib"
    if runtime_torch_lib.is_dir():
        ld_library_parts.append(str(runtime_torch_lib))

    app_torch_lib = _app_purelib() / "torch" / "lib"
    if include_app_torch_lib and app_torch_lib.is_dir():
        ld_library_parts.append(str(app_torch_lib))

    for system_cuda_path in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/aarch64-linux/lib",
    ):
        if os.path.isdir(system_cuda_path):
            ld_library_parts.append(system_cuda_path)

    existing_ld_library_path = env.get("LD_LIBRARY_PATH")
    if existing_ld_library_path:
        ld_library_parts.append(existing_ld_library_path)
    if ld_library_parts:
        merged_parts: list[str] = []
        for part in ld_library_parts:
            if part and part not in merged_parts:
                merged_parts.append(part)
        env["LD_LIBRARY_PATH"] = ":".join(merged_parts)
    return env


def _ensure_fallback_site_packages(python_bin: Path, *, fallback_paths: list[str] | None = None) -> None:
    purelib = _purelib(python_bin)
    fallback_file = purelib / "_vox_runtime_fallback_paths.pth"
    requested = [path for path in (fallback_paths or []) if path]
    requested.append(str(_app_purelib()))

    existing: list[str] = []
    if fallback_file.exists():
        existing = [line.strip() for line in fallback_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    merged: list[str] = []
    for path in requested + existing:
        if path and path not in merged:
            merged.append(path)

    fallback_file.write_text("".join(f"{path}\n" for path in merged), encoding="utf-8")


def _has_gpu_torch(python_bin: Path) -> bool:
    runtime_torch_lib = _purelib(python_bin) / "torch" / "lib"
    if not runtime_torch_lib.is_dir():
        return False

    result = _run(
        [
            str(python_bin),
            "-c",
            (
                "import json, torch; "
                "available = bool(torch.cuda.is_available()); "
                "print(json.dumps({"
                "    'version': torch.__version__, "
                "    'cuda': torch.version.cuda or 'none', "
                "    'available': available, "
                "    'file': torch.__file__, "
                "}))"
            ),
        ],
        env=_build_env(python_bin),
        timeout=120,
    )
    if result.returncode != 0:
        return False
    try:
        payload = yaml.safe_load(result.stdout)
    except yaml.YAMLError:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("cuda") in (None, "none"):
        return False
    if not payload.get("available"):
        return False
    runtime_torch_root = (_purelib(python_bin) / "torch").resolve()
    torch_file = payload.get("file")
    if not isinstance(torch_file, str):
        return False
    try:
        resolved_file = Path(torch_file).resolve()
    except OSError:
        return False
    return runtime_torch_root in resolved_file.parents


def _has_vllm_omni_runtime(python_bin: Path, *, extra_pythonpaths: list[str] | None = None) -> bool:
    del extra_pythonpaths  # runtime-local validation must ignore app fallback imports

    purelib = _purelib(python_bin)
    stage_config = purelib / "vllm_omni" / "model_executor" / "stage_configs" / "voxtral_tts.yaml"
    return (
        (purelib / "vllm").exists()
        and (purelib / "vllm_omni").exists()
        and stage_config.is_file()
    )


def _normalized_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name).split("+", 1)[0]
    except metadata.PackageNotFoundError:
        return None


def _default_torchvision_version(torch_version: str) -> str:
    major_minor = ".".join(torch_version.split(".")[:2])
    return {
        "2.6": "0.21.0",
        "2.7": "0.22.0",
        "2.8": "0.23.0",
        "2.9": "0.24.0",
        "2.10": "0.25.0",
    }.get(major_minor, "0.25.0")


def _detected_cuda_version() -> str:
    try:
        import torch

        return str(getattr(torch.version, "cuda", "") or "")
    except Exception:
        return ""


def _detected_total_gpu_memory_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        properties = torch.cuda.get_device_properties(0)
        total_memory = getattr(properties, "total_memory", None)
        if total_memory is None:
            return None
        return int(total_memory)
    except Exception:
        return None


def _is_small_gpu(total_memory_bytes: int | None = None) -> bool:
    if total_memory_bytes is None:
        total_memory_bytes = _detected_total_gpu_memory_bytes()
    return total_memory_bytes is not None and total_memory_bytes <= 24 * 1024**3


def _is_16gb_gpu(total_memory_bytes: int | None = None) -> bool:
    if total_memory_bytes is None:
        total_memory_bytes = _detected_total_gpu_memory_bytes()
    return total_memory_bytes is not None and total_memory_bytes <= 16 * 1024**3


def _default_generation_gpu_memory_utilization() -> float:
    if _is_16gb_gpu():
        return 0.62
    if _is_small_gpu():
        return 0.4
    return 0.4


def _default_tokenizer_gpu_memory_utilization() -> float:
    if _is_16gb_gpu():
        return 0.01
    if _is_small_gpu():
        return 0.05
    return 0.1


def _default_generation_max_model_len() -> int:
    if _is_16gb_gpu():
        return 512
    if _is_small_gpu():
        return 1536
    return 2048


def _default_tokenizer_max_num_batched_tokens() -> int:
    if _is_small_gpu():
        return 4096
    return 8192


def _default_tokenizer_max_model_len() -> int:
    if _is_small_gpu():
        return 4096
    return 8192


def _default_attention_backend() -> str | None:
    if _is_16gb_gpu():
        return "triton_attn"
    return None


def recommended_voxtral_tts_vram_bytes() -> int:
    if _is_small_gpu():
        return 12_000_000_000
    return 16_000_000_000


def _arm64_runtime_torch_local_suffix() -> str:
    if platform.machine() not in {"aarch64", "arm64"}:
        return ""
    if _detected_cuda_version().startswith("13."):
        return ""
    return "+cu129"


def _resolved_torch_runtime_version() -> str:
    configured = os.environ.get("VOX_VOXTRAL_TORCH_VERSION")
    if configured:
        return configured
    return f"2.10.0{_arm64_runtime_torch_local_suffix()}"


def _resolved_torchaudio_runtime_version(torch_version: str) -> str:
    return os.environ.get("VOX_VOXTRAL_TORCHAUDIO_VERSION", torch_version)


def _resolved_torchvision_runtime_version(torch_version: str) -> str:
    configured = os.environ.get("VOX_VOXTRAL_TORCHVISION_VERSION")
    if configured:
        return configured
    suffix = _arm64_runtime_torch_local_suffix()
    base = _default_torchvision_version(torch_version)
    if suffix and "+" not in base:
        return f"{base}{suffix}"
    return base


def _gpu_torch_index_url() -> str:
    configured = os.environ.get("VOX_VOXTRAL_TORCH_INDEX_URL")
    if configured:
        return configured

    cuda_version = _detected_cuda_version()

    if platform.machine() in {"aarch64", "arm64"}:
        if cuda_version.startswith("13."):
            return "https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"
        return "https://download.pytorch.org/whl/cu129"
    return "https://download.pytorch.org/whl/cu128"


def _install_gpu_torch(python_bin: Path) -> None:
    torch_version = _resolved_torch_runtime_version()
    torchaudio_version = _resolved_torchaudio_runtime_version(torch_version)
    torchvision_version = _resolved_torchvision_runtime_version(torch_version)
    index_url = _gpu_torch_index_url()
    result = _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "--reinstall",
            "--index-url",
            index_url,
            f"torch=={torch_version}",
            f"torchvision=={torchvision_version}",
            f"torchaudio=={torchaudio_version}",
        ],
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install CUDA Torch runtime for Voxtral TTS: {result.stderr.strip()}")
    if not _has_gpu_torch(python_bin):
        raise RuntimeError(
            "Installed Torch runtime for Voxtral TTS, but it is still not a runtime-local CUDA build. "
            f"index_url={index_url}"
        )


def _install_vllm_runtime(python_bin: Path) -> None:
    result = _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_bin),
            "vllm==0.18.0",
            "vllm-omni==0.18.0",
            "mistral-common[audio]>=1.10.0",
        ],
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install vLLM-Omni runtime for Voxtral TTS: {result.stderr.strip()}")
    if not _has_gpu_torch(python_bin):
        raise RuntimeError(
            "vLLM-Omni installation replaced the Voxtral TTS runtime Torch build with a non-CUDA variant"
        )
    if not _has_vllm_omni_runtime(python_bin):
        raise RuntimeError("Installed vLLM-Omni runtime but runtime-local stage configs are still missing")


def _write_stage_config(python_bin: Path) -> Path:
    purelib = _purelib(python_bin)
    source = purelib / "vllm_omni" / "model_executor" / "stage_configs" / "voxtral_tts.yaml"
    if not source.is_file():
        raise RuntimeError("Voxtral TTS runtime installed but stage config file is missing")

    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for stage in config.get("stage_args", []):
        engine_args = stage.get("engine_args", {})
        engine_args["max_num_seqs"] = 1
        default_attention_backend = _default_attention_backend()

        model_stage = engine_args.get("model_stage")
        if model_stage == "audio_generation":
            current = engine_args.get("gpu_memory_utilization")
            target_value, explicit_target = _target_gpu_memory_utilization(current, model_stage=model_stage)
            if explicit_target or current is None or _is_16gb_gpu() or float(current) > target_value:
                engine_args["gpu_memory_utilization"] = target_value
            engine_args["enforce_eager"] = True
            generation_max_model_len = _default_generation_max_model_len()
            if int(engine_args.get("max_model_len", 4096)) > generation_max_model_len:
                engine_args["max_model_len"] = generation_max_model_len
            if _is_16gb_gpu():
                engine_args.pop("dtype", None)
                engine_args["kv_cache_dtype"] = "fp8"
            if default_attention_backend is not None:
                engine_args["attention_backend"] = default_attention_backend
        elif model_stage == "audio_tokenizer":
            current = engine_args.get("gpu_memory_utilization")
            target_value, explicit_target = _target_gpu_memory_utilization(current, model_stage=model_stage)
            if explicit_target or current is None or float(current) > target_value:
                engine_args["gpu_memory_utilization"] = target_value
            tokenizer_max_batched_tokens = _default_tokenizer_max_num_batched_tokens()
            if int(engine_args.get("max_num_batched_tokens", 65536)) > tokenizer_max_batched_tokens:
                engine_args["max_num_batched_tokens"] = tokenizer_max_batched_tokens
            tokenizer_max_model_len = _default_tokenizer_max_model_len()
            if int(engine_args.get("max_model_len", 65536)) > tokenizer_max_model_len:
                engine_args["max_model_len"] = tokenizer_max_model_len
            if _is_16gb_gpu():
                engine_args.pop("dtype", None)
            if default_attention_backend is not None:
                engine_args["attention_backend"] = default_attention_backend

    target = _runtime_root() / "voxtral-tts-stage-config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def _parse_gpu_memory_utilization(raw_value: str | float | int | None, *, default: float) -> float:
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, 0.01), 0.95)


def _target_gpu_memory_utilization(
    current_value: str | float | int | None,
    *,
    model_stage: str | None,
) -> tuple[float, bool]:
    global_raw = os.environ.get("VOX_VOXTRAL_GPU_MEMORY_UTILIZATION")
    if model_stage == "audio_generation":
        explicit_raw = os.environ.get("VOX_VOXTRAL_GENERATION_GPU_MEMORY_UTILIZATION")
        raw_value = explicit_raw or os.environ.get(
            "VOX_VOXTRAL_GENERATION_GPU_MEMORY_UTILIZATION",
            global_raw if global_raw is not None else str(_default_generation_gpu_memory_utilization()),
        )
        return _parse_gpu_memory_utilization(
            raw_value,
            default=_default_generation_gpu_memory_utilization(),
        ), bool(explicit_raw or global_raw)
    if model_stage == "audio_tokenizer":
        explicit_raw = os.environ.get("VOX_VOXTRAL_TOKENIZER_GPU_MEMORY_UTILIZATION")
        raw_value = explicit_raw or os.environ.get(
            "VOX_VOXTRAL_TOKENIZER_GPU_MEMORY_UTILIZATION",
            global_raw if global_raw is not None else str(_default_tokenizer_gpu_memory_utilization()),
        )
        return _parse_gpu_memory_utilization(
            raw_value,
            default=_default_tokenizer_gpu_memory_utilization(),
        ), bool(explicit_raw or global_raw)
    return _parse_gpu_memory_utilization(global_raw or current_value, default=0.4), bool(global_raw)


def _pinned_stt_dependency_versions() -> dict[str, str]:
    return {
        "transformers": "4.57.6",
        "tokenizers": "0.22.2",
        "huggingface_hub": "0.36.2",
    }


def _runtime_has_dist_info(runtime_dir: Path, dist_name: str, version: str) -> bool:
    normalized = dist_name.replace("-", "_")
    return (runtime_dir / f"{normalized}-{version}.dist-info").exists()


def ensure_voxtral_stt_runtime() -> str:
    runtime_dir = _stt_runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = str(runtime_dir)

    fallback_file = runtime_dir / "_vox_runtime_fallback_paths.pth"
    fallback_file.write_text(f"{_app_purelib()}\n", encoding="utf-8")

    if runtime_path in sys.path:
        sys.path.remove(runtime_path)
    sys.path.insert(0, runtime_path)

    required_specs: list[str] = []
    pinned_versions = _pinned_stt_dependency_versions()

    if not _runtime_has_dist_info(runtime_dir, "transformers", pinned_versions["transformers"]):
        required_specs.extend(
            [
                f"transformers=={pinned_versions['transformers']}",
            ]
        )
    if not _runtime_has_dist_info(runtime_dir, "tokenizers", pinned_versions["tokenizers"]):
        required_specs.append(f"tokenizers=={pinned_versions['tokenizers']}")
    if not _runtime_has_dist_info(runtime_dir, "huggingface_hub", pinned_versions["huggingface_hub"]):
        required_specs.append(f"huggingface-hub=={pinned_versions['huggingface_hub']}")

    if not (runtime_dir / "mistral_common").exists():
        required_specs.append("mistral-common[audio]>=1.10.0")

    _clear_stt_runtime_modules()
    importlib.invalidate_caches()

    if required_specs:
        installers = [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                runtime_path,
                "--upgrade",
                *required_specs,
            ],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                runtime_path,
                "--upgrade",
                *required_specs,
            ],
        ]

        install_error = ""
        for installer in installers:
            result = None
            try:
                result = _run(installer, timeout=1800)
            except FileNotFoundError:
                continue
            if result.returncode == 0:
                break
            install_error = result.stderr.strip() or result.stdout.strip()
        else:
            raise RuntimeError(
                "Failed to install Voxtral STT runtime dependencies. "
                f"stderr: {install_error}"
            )

    return runtime_path


def ensure_voxtral_tts_runtime(*, extra_pythonpaths: list[str] | None = None) -> VoxtralRuntimeInfo:
    venv_dir = _runtime_venv()
    python_bin = _ensure_venv(venv_dir)

    if not _has_gpu_torch(python_bin):
        shutil.rmtree(venv_dir, ignore_errors=True)
        python_bin = _ensure_venv(venv_dir)
        _install_gpu_torch(python_bin)

    if not _has_vllm_omni_runtime(python_bin, extra_pythonpaths=extra_pythonpaths):
        _install_vllm_runtime(python_bin)

    env = _build_env(
        python_bin,
        extra_pythonpaths=extra_pythonpaths,
        include_app_torch_lib=False,
    )
    purelib = _purelib(python_bin)
    stage_configs_path = _write_stage_config(python_bin)

    return VoxtralRuntimeInfo(
        python_executable=str(python_bin),
        env=env,
        stage_configs_path=str(stage_configs_path),
        site_packages=str(purelib),
    )
