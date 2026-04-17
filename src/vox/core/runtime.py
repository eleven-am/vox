from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeCapabilities:
    system: str
    machine: str
    torch_cuda: bool
    onnx_cuda: bool
    onnx_coreml: bool
    mps: bool
    nvidia_device: bool

    @property
    def has_gpu_accelerator(self) -> bool:
        return self.torch_cuda or self.onnx_cuda or self.nvidia_device


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_mps_available() -> bool:
    try:
        import torch

        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        is_available = getattr(mps, "is_available", None)
        return bool(is_available()) if callable(is_available) else False
    except Exception:
        return False


def _onnx_available_providers() -> set[str]:
    try:
        import onnxruntime as ort

        return set(ort.get_available_providers())
    except Exception:
        return set()


def _has_nvidia_device_files() -> bool:
    candidate_paths = (
        "/dev/nvidiactl",
        "/dev/nvidia0",
        "/proc/driver/nvidia/version",
    )
    return any(Path(path).exists() for path in candidate_paths)


def _nvidia_visible_devices_configured() -> bool:
    value = os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip().lower()
    if not value:
        return False
    return value not in {"void", "none", "no", "off"}


def detect_runtime_capabilities() -> RuntimeCapabilities:
    providers = _onnx_available_providers()
    return RuntimeCapabilities(
        system=platform.system().lower(),
        machine=platform.machine().lower(),
        torch_cuda=_torch_cuda_available(),
        onnx_cuda="CUDAExecutionProvider" in providers,
        onnx_coreml="CoreMLExecutionProvider" in providers,
        mps=_torch_mps_available(),
        nvidia_device=_has_nvidia_device_files() or _nvidia_visible_devices_configured(),
    )


def infer_runtime_profile(*, device_hint: str | None = None) -> str:
    hint = (device_hint or os.environ.get("VOX_DEVICE", "auto")).strip().lower()
    capabilities = detect_runtime_capabilities()

    if capabilities.system != "linux":
        return "default"

    if capabilities.machine not in {"arm64", "aarch64"}:
        return "default"

    if hint == "cuda":
        return "spark"

    return "spark" if capabilities.has_gpu_accelerator else "default"
