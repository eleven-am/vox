from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class VoxtralRuntimeInfo:
    python_executable: str
    env: dict[str, str]
    stage_configs_path: str
    site_packages: str


def _vox_home() -> Path:
    return Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))


def _runtime_root() -> Path:
    return _vox_home() / "runtime"


def _runtime_venv() -> Path:
    return _runtime_root() / "voxtral-tts"


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


def _build_env(python_bin: Path, *, extra_pythonpaths: list[str] | None = None) -> dict[str, str]:
    env = os.environ.copy()

    pythonpath_parts: list[str] = []
    if extra_pythonpaths:
        pythonpath_parts.extend(extra_pythonpaths)
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    if pythonpath_parts:
        env["PYTHONPATH"] = ":".join(part for part in pythonpath_parts if part)
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
    result = _run(
        [
            str(python_bin),
            "-c",
            (
                "import torch; "
                "print(torch.__version__); "
                "print(torch.version.cuda or 'none'); "
                "print(int(torch.cuda.is_available()))"
            ),
        ],
        timeout=120,
    )
    if result.returncode != 0:
        return False
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return len(lines) >= 3 and lines[1] != "none"


def _has_vllm_omni_runtime(python_bin: Path, *, extra_pythonpaths: list[str] | None = None) -> bool:
    env = _build_env(python_bin, extra_pythonpaths=extra_pythonpaths)
    result = _run(
        [
            str(python_bin),
            "-c",
            (
                "from importlib.resources import files; "
                "import vllm_omni; "
                "p = files('vllm_omni').joinpath('model_executor/stage_configs/voxtral_tts.yaml'); "
                "print(int(p.is_file()))"
            ),
        ],
        env=env,
        timeout=120,
    )
    return result.returncode == 0 and result.stdout.strip().endswith("1")


def _install_gpu_torch(python_bin: Path) -> None:
    index_url = os.environ.get("VOX_VOXTRAL_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130")
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
            "torch==2.10.0",
            "torchvision==0.25.0",
            "torchaudio==2.10.0",
        ],
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install CUDA Torch runtime for Voxtral TTS: {result.stderr.strip()}")


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

        model_stage = engine_args.get("model_stage")
        if model_stage == "audio_generation":
            current = engine_args.get("gpu_memory_utilization")
            target_value, explicit_target = _target_gpu_memory_utilization(current, model_stage=model_stage)
            if explicit_target or current is None or float(current) > target_value:
                engine_args["gpu_memory_utilization"] = target_value
            engine_args["enforce_eager"] = True
            if int(engine_args.get("max_model_len", 4096)) > 2048:
                engine_args["max_model_len"] = 2048
        elif model_stage == "audio_tokenizer":
            current = engine_args.get("gpu_memory_utilization")
            target_value, explicit_target = _target_gpu_memory_utilization(current, model_stage=model_stage)
            if explicit_target or current is None or float(current) > target_value:
                engine_args["gpu_memory_utilization"] = target_value
            if int(engine_args.get("max_num_batched_tokens", 65536)) > 8192:
                engine_args["max_num_batched_tokens"] = 8192
            if int(engine_args.get("max_model_len", 65536)) > 8192:
                engine_args["max_model_len"] = 8192

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
    return min(max(value, 0.05), 0.95)


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
            global_raw if global_raw is not None else "0.4",
        )
        return _parse_gpu_memory_utilization(raw_value, default=0.4), bool(explicit_raw or global_raw)
    if model_stage == "audio_tokenizer":
        explicit_raw = os.environ.get("VOX_VOXTRAL_TOKENIZER_GPU_MEMORY_UTILIZATION")
        raw_value = explicit_raw or os.environ.get(
            "VOX_VOXTRAL_TOKENIZER_GPU_MEMORY_UTILIZATION",
            global_raw if global_raw is not None else "0.1",
        )
        return _parse_gpu_memory_utilization(raw_value, default=0.1), bool(explicit_raw or global_raw)
    return _parse_gpu_memory_utilization(global_raw or current_value, default=0.4), bool(global_raw)


def ensure_voxtral_tts_runtime(*, extra_pythonpaths: list[str] | None = None) -> VoxtralRuntimeInfo:
    venv_dir = _runtime_venv()
    python_bin = _ensure_venv(venv_dir)
    _ensure_fallback_site_packages(python_bin, fallback_paths=extra_pythonpaths)

    if not _has_gpu_torch(python_bin):
        shutil.rmtree(venv_dir, ignore_errors=True)
        python_bin = _ensure_venv(venv_dir)
        _ensure_fallback_site_packages(python_bin, fallback_paths=extra_pythonpaths)
        _install_gpu_torch(python_bin)

    if not _has_vllm_omni_runtime(python_bin, extra_pythonpaths=extra_pythonpaths):
        _install_vllm_runtime(python_bin)

    env = _build_env(python_bin, extra_pythonpaths=extra_pythonpaths)
    purelib = _purelib(python_bin)
    stage_configs_path = _write_stage_config(python_bin)

    return VoxtralRuntimeInfo(
        python_executable=str(python_bin),
        env=env,
        stage_configs_path=str(stage_configs_path),
        site_packages=str(purelib),
    )
