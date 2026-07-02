"""Environment capability detection and per-model requirement inference.

Lets `vox pull` fail fast with a clear reason when the current environment can't
run a model (e.g. a torch-based model on a torch-free lean image), instead of
downloading it and crashing at load time. Requirements are inferred from the
catalog entry's existing fields — no registry changes needed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any

from vox.core.runtime import detect_runtime_capabilities

ALLOW_INCOMPATIBLE_ENV = "VOX_ALLOW_INCOMPATIBLE"
_TORCH_OVERRIDE_ENV = "VOX_HAS_TORCH"
_ONNX_OVERRIDE_ENV = "VOX_HAS_ONNXRUNTIME"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_override(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return None


def _module_available(module: str) -> bool:
    if sys.modules.get(module) is not None:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def torch_available() -> bool:
    override = _env_override(_TORCH_OVERRIDE_ENV)
    if override is not None:
        return override
    return _module_available("torch")


def onnxruntime_available() -> bool:
    override = _env_override(_ONNX_OVERRIDE_ENV)
    if override is not None:
        return override
    return _module_available("onnxruntime")


def incompatible_pull_allowed() -> bool:
    return _env_override(ALLOW_INCOMPATIBLE_ENV) is True


@dataclass(frozen=True)
class ModelRequirements:
    runtime: str | None = None       # "torch" | "onnxruntime" | None
    accelerator: str | None = None   # "cuda" | None (None = CPU is fine)


def infer_model_requirements(entry: dict[str, Any]) -> ModelRequirements:
    fmt = str(entry.get("format") or "").strip().lower()
    adapter = str(entry.get("adapter") or "").strip().lower()

    runtime: str | None = None
    if fmt == "pytorch":
        runtime = "torch"
    elif fmt == "onnx":
        runtime = "onnxruntime"
    # ct2/gguf install their inference runtime into an isolated adapter runtime
    # on pull, so they carry no base-environment runtime requirement.

    accelerator = "cuda" if adapter.endswith("-vllm") else None
    return ModelRequirements(runtime=runtime, accelerator=accelerator)


def missing_capabilities_for(entry: dict[str, Any]) -> list[str]:
    """Return human-readable reasons this environment can't run the model."""
    requirements = infer_model_requirements(entry)
    missing: list[str] = []

    if requirements.runtime == "torch" and not torch_available():
        missing.append("requires PyTorch (torch), which is not installed in this environment")
    elif requirements.runtime == "onnxruntime" and not onnxruntime_available():
        missing.append("requires onnxruntime, which is not installed in this environment")

    if requirements.accelerator == "cuda":
        capabilities = detect_runtime_capabilities()
        if not capabilities.has_gpu_accelerator:
            missing.append("requires a CUDA GPU, which is not available in this environment")

    return missing
