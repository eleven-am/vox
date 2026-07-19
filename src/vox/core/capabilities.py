"""Environment capability detection and per-model requirement inference.

The pull gate is catalog-driven because adapter packages are not installed or
imported until after compatibility is checked. Direct runtime probing is the
authority.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

from vox.core.runtime import RuntimeCapabilities, detect_runtime_capabilities

ALLOW_INCOMPATIBLE_ENV = "VOX_ALLOW_INCOMPATIBLE"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
_VERSION_PART_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class RuntimeRequirement:
    python_modules: tuple[str, ...] = ()
    min_versions: dict[str, str] = field(default_factory=dict)
    accelerators: tuple[str, ...] = ()
    systems: tuple[str, ...] = ()
    machines: tuple[str, ...] = ()
    min_vram_gb: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityCheck:
    runnable: bool
    preferred_backend: str | None = None
    missing: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelRequirements:
    runtime: str | None = None       # "torch" | "onnxruntime" | None
    accelerator: str | None = None   # "cuda" | None
    requirement: RuntimeRequirement = field(default_factory=RuntimeRequirement)


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


def _package_version(module: str) -> str | None:
    package_name = {
        "onnxruntime": "onnxruntime",
        "torch": "torch",
    }.get(module, module.replace("_", "-"))
    loaded = sys.modules.get(module)
    version = getattr(loaded, "__version__", None)
    if version:
        return str(version)
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in _VERSION_PART_RE.findall(value))


def _version_at_least(actual: str | None, minimum: str) -> bool:
    actual_parts = _version_tuple(actual)
    minimum_parts = _version_tuple(minimum)
    if not actual_parts or not minimum_parts:
        return False
    length = max(len(actual_parts), len(minimum_parts))
    actual_normalized = actual_parts + (0,) * (length - len(actual_parts))
    minimum_normalized = minimum_parts + (0,) * (length - len(minimum_parts))
    return actual_normalized >= minimum_normalized


def torch_available(*, snapshot: RuntimeCapabilities | None = None) -> bool:
    snapshot = snapshot if snapshot is not None else detect_runtime_capabilities()
    return snapshot.torch_installed


def onnxruntime_available(*, snapshot: RuntimeCapabilities | None = None) -> bool:
    snapshot = snapshot if snapshot is not None else detect_runtime_capabilities()
    return snapshot.onnxruntime_installed


def incompatible_pull_allowed() -> bool:
    return _env_override(ALLOW_INCOMPATIBLE_ENV) is True


def runtime_requirement_from_mapping(data: Any) -> RuntimeRequirement:
    if not isinstance(data, dict):
        return RuntimeRequirement()
    return RuntimeRequirement(
        python_modules=tuple(str(item) for item in data.get("python_modules", ()) or ()),
        min_versions={
            str(key): str(value)
            for key, value in (data.get("min_versions") or {}).items()
        },
        accelerators=tuple(str(item) for item in data.get("accelerators", ()) or ()),
        systems=tuple(str(item).lower() for item in data.get("systems", ()) or ()),
        machines=tuple(str(item).lower() for item in data.get("machines", ()) or ()),
        min_vram_gb=_optional_float(data.get("min_vram_gb")),
        notes=tuple(str(item) for item in data.get("notes", ()) or ()),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_model_requirements(entry: dict[str, Any]) -> ModelRequirements:
    explicit = runtime_requirement_from_mapping(
        (entry.get("runtime") or {}).get("required")
    )
    fmt = str(entry.get("format") or "").strip().lower()
    adapter = str(entry.get("adapter") or "").strip().lower()
    adapter_package = str(entry.get("adapter_package") or "").strip()

    runtime: str | None = None
    inferred_modules: tuple[str, ...] = ()
    if fmt == "pytorch":
        runtime = "torch"
        inferred_modules = ("torch",)
    elif fmt == "onnx":
        runtime = "onnxruntime"
        if not adapter_package:
            inferred_modules = ("onnxruntime",)
    # ct2/gguf install their inference runtime into an isolated adapter runtime
    # on pull, so they carry no base-environment runtime requirement.
    # Adapter-backed ONNX entries follow the same rule: the adapter package owns
    # installing and verifying onnxruntime in its isolated runtime directory.
    explicit_modules = explicit.python_modules
    if fmt == "onnx" and adapter_package:
        explicit_modules = tuple(
            module for module in explicit_modules
            if module != "onnxruntime"
        )

    accelerator = "cuda" if adapter.endswith("-vllm") else None
    inferred_accelerators = ("cuda",) if accelerator == "cuda" else ()

    requirement = RuntimeRequirement(
        python_modules=tuple(dict.fromkeys((*inferred_modules, *explicit_modules))),
        min_versions=explicit.min_versions,
        accelerators=tuple(dict.fromkeys((*inferred_accelerators, *explicit.accelerators))),
        systems=explicit.systems,
        machines=explicit.machines,
        min_vram_gb=explicit.min_vram_gb,
        notes=explicit.notes,
    )
    return ModelRequirements(
        runtime=runtime,
        accelerator=accelerator,
        requirement=requirement,
    )


def check_runtime_requirement(
    requirement: RuntimeRequirement,
    *,
    snapshot: RuntimeCapabilities | None = None,
    preferred_backend: str | None = None,
) -> CapabilityCheck:
    snapshot = snapshot if snapshot is not None else detect_runtime_capabilities()
    missing: list[str] = []

    for module in requirement.python_modules:
        if not _module_available_from_snapshot(module, snapshot):
            missing.append(
                f"requires {_module_label(module)}, which is not installed in this environment"
            )

    for module, minimum in requirement.min_versions.items():
        actual = _version_for_module(module, snapshot)
        if not _version_at_least(actual, minimum):
            detail = f"found {actual}" if actual else "version is unknown or module is missing"
            missing.append(f"requires {module}>={minimum} ({detail})")

    if requirement.accelerators and not any(
        _accelerator_available(accelerator, snapshot)
        for accelerator in requirement.accelerators
    ):
        missing.append(
            "requires one of accelerators "
            f"{', '.join(requirement.accelerators)}, which is not available in this environment"
        )

    if requirement.systems and snapshot.system.lower() not in requirement.systems:
        missing.append(
            f"requires system {', '.join(requirement.systems)}; detected {snapshot.system or 'UNKNOWN'}"
        )

    if requirement.machines and snapshot.machine.lower() not in requirement.machines:
        missing.append(
            f"requires machine {', '.join(requirement.machines)}; detected {snapshot.machine or 'UNKNOWN'}"
        )

    if requirement.min_vram_gb is not None:
        actual = snapshot.vram_gb
        if actual is None:
            missing.append(
                f"requires VRAM >= {requirement.min_vram_gb:g} GiB; detected UNKNOWN"
            )
        elif actual < requirement.min_vram_gb:
            missing.append(
                f"requires VRAM >= {requirement.min_vram_gb:g} GiB; "
                f"detected {actual:.1f} GiB"
            )

    return CapabilityCheck(
        runnable=not missing,
        preferred_backend=preferred_backend if not missing else None,
        missing=tuple(missing),
        warnings=(),
    )


def _module_available_from_snapshot(module: str, snapshot: RuntimeCapabilities) -> bool:
    if module == "torch":
        return snapshot.torch_installed
    if module == "onnxruntime":
        return snapshot.onnxruntime_installed
    return _module_available(module)


def _module_label(module: str) -> str:
    if module == "torch":
        return "PyTorch (torch)"
    if module == "onnxruntime":
        return "onnxruntime"
    return f"Python module {module!r}"


def _version_for_module(module: str, snapshot: RuntimeCapabilities) -> str | None:
    if module == "torch":
        return snapshot.torch_version
    if module == "onnxruntime":
        return snapshot.onnxruntime_version
    return _package_version(module)


def _accelerator_available(accelerator: str, snapshot: RuntimeCapabilities) -> bool:
    normalized = accelerator.strip().lower()
    if normalized == "cuda":
        return snapshot.torch_cuda
    if normalized == "mps":
        return snapshot.mps or snapshot.torch_mps_available
    if normalized == "onnx_cuda":
        return snapshot.onnx_cuda
    return normalized == "cpu"


def check_model_capabilities(
    entry: dict[str, Any],
    *,
    snapshot: RuntimeCapabilities | None = None,
) -> CapabilityCheck:
    requirements = infer_model_requirements(entry)
    return check_runtime_requirement(requirements.requirement, snapshot=snapshot)


def missing_capabilities_for(
    entry: dict[str, Any],
    *,
    snapshot: RuntimeCapabilities | None = None,
) -> list[str]:
    """Return human-readable reasons this environment can't run the model."""
    return list(check_model_capabilities(entry, snapshot=snapshot).missing)
