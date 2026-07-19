from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class RuntimeCapabilities:
    system: str = ""
    machine: str = ""
    python_version: str = ""
    torch_installed: bool = False
    torch_version: str | None = None
    torch_cuda: bool = False
    torch_cuda_version: str | None = None
    torch_device_count: int | None = None
    torch_device_names: tuple[str, ...] = ()
    torch_compute_capability: int | None = None
    torch_mps_available: bool = False
    onnxruntime_installed: bool = False
    onnxruntime_version: str | None = None
    onnxruntime_providers: tuple[str, ...] = ()
    onnx_cuda: bool = False
    onnx_coreml: bool = False
    mps: bool = False
    ram_gb: float | None = None
    vram_gb: float | None = None
    nvidia_device: bool = False
    nvidia_smi_available: bool = False
    nvidia_driver_version: str | None = None
    driver_cuda_version: str | None = None


@dataclass(frozen=True)
class _TorchProbe:
    installed: bool = False
    version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    device_count: int | None = None
    device_names: tuple[str, ...] = ()
    compute_capability: int | None = None
    mps_available: bool = False
    vram_gb: float | None = None


@dataclass(frozen=True)
class _OnnxProbe:
    installed: bool = False
    version: str | None = None
    providers: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NvidiaSmiProbe:
    available: bool = False
    driver_version: str | None = None
    cuda_version: str | None = None
    vram_gb: float | None = None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _torch_probe() -> _TorchProbe:
    try:
        import torch  # type: ignore[import-not-found]

        cuda = getattr(torch, "cuda", None)
        cuda_available = bool(cuda is not None and cuda.is_available())
        device_count = None
        device_names: list[str] = []
        compute_caps: list[int] = []
        vram_values: list[float] = []

        if cuda_available and cuda is not None:
            try:
                device_count = int(cuda.device_count())
            except Exception:
                device_count = None
            for index in range(device_count or 0):
                with suppress(Exception):
                    device_names.append(str(cuda.get_device_name(index)))
                try:
                    major, minor = cuda.get_device_capability(index)
                    compute_caps.append(int(major) * 10 + int(minor))
                except Exception:
                    pass
                try:
                    props = cuda.get_device_properties(index)
                    total_memory = getattr(props, "total_memory", None)
                    if total_memory is not None:
                        vram_values.append(int(total_memory) / (1024**3))
                except Exception:
                    pass

        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        is_mps_available = getattr(mps, "is_available", None)

        version = getattr(torch, "__version__", None) or _package_version("torch")
        torch_version = str(version) if version is not None else None
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        return _TorchProbe(
            installed=True,
            version=torch_version,
            cuda_available=cuda_available,
            cuda_version=str(torch_cuda_version) if torch_cuda_version else None,
            device_count=device_count,
            device_names=tuple(device_names),
            compute_capability=max(compute_caps) if compute_caps else None,
            mps_available=bool(is_mps_available()) if callable(is_mps_available) else False,
            vram_gb=max(vram_values) if vram_values else None,
        )
    except Exception:
        # torch metadata may exist while `import torch` fails (broken CUDA wheel,
        # missing shared library). It is not usable, so report it unavailable
        # rather than let the gate pass and fail later at load.
        return _TorchProbe(installed=False)


def _onnx_probe() -> _OnnxProbe:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        providers = tuple(str(provider) for provider in ort.get_available_providers())
        version = getattr(ort, "__version__", None) or _package_version("onnxruntime")
        return _OnnxProbe(
            installed=True,
            version=str(version) if version else None,
            providers=providers,
        )
    except Exception:
        # As with torch: metadata may exist while `import onnxruntime` fails.
        # Report it unavailable rather than pass the gate and fail at load.
        return _OnnxProbe(installed=False)


def _nvidia_smi_cuda_version() -> str | None:
    # cuda_version is not a valid --query-gpu field; it only appears in the
    # header of plain `nvidia-smi` output ("CUDA Version: 12.4").
    with suppress(Exception):
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", result.stdout)
            if match:
                return match.group(1)
    return None


def _nvidia_smi_probe() -> _NvidiaSmiProbe:
    if shutil.which("nvidia-smi") is None:
        return _NvidiaSmiProbe()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return _NvidiaSmiProbe(available=True)

    if result.returncode != 0:
        return _NvidiaSmiProbe(available=True, cuda_version=_nvidia_smi_cuda_version())

    driver_versions: list[str] = []
    memory_values: list[float] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0]:
            driver_versions.append(parts[0])
        if len(parts) >= 2:
            with suppress(ValueError):
                memory_values.append(float(parts[1]) / 1024)

    return _NvidiaSmiProbe(
        available=True,
        driver_version=driver_versions[0] if driver_versions else None,
        cuda_version=_nvidia_smi_cuda_version(),
        vram_gb=max(memory_values) if memory_values else None,
    )


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


def _ram_gb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.virtual_memory().total) / (1024**3)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            return float(page_size * pages) / (1024**3)
        except (OSError, ValueError):
            return None
    return None


def detect_runtime_capabilities() -> RuntimeCapabilities:
    torch = _torch_probe()
    onnx = _onnx_probe()
    nvidia_smi = _nvidia_smi_probe()
    nvidia_device = (
        _has_nvidia_device_files()
        or _nvidia_visible_devices_configured()
        or nvidia_smi.available
    )
    vram_candidates = [
        value for value in (torch.vram_gb, nvidia_smi.vram_gb) if value is not None
    ]
    return RuntimeCapabilities(
        system=platform.system().lower(),
        machine=platform.machine().lower(),
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        torch_installed=torch.installed,
        torch_version=torch.version,
        torch_cuda=torch.cuda_available,
        torch_cuda_version=torch.cuda_version,
        torch_device_count=torch.device_count,
        torch_device_names=torch.device_names,
        torch_compute_capability=torch.compute_capability,
        torch_mps_available=torch.mps_available,
        onnxruntime_installed=onnx.installed,
        onnxruntime_version=onnx.version,
        onnxruntime_providers=onnx.providers,
        onnx_cuda="CUDAExecutionProvider" in onnx.providers,
        onnx_coreml="CoreMLExecutionProvider" in onnx.providers,
        mps=torch.mps_available,
        ram_gb=_ram_gb(),
        vram_gb=max(vram_candidates) if vram_candidates else None,
        nvidia_device=nvidia_device,
        nvidia_smi_available=nvidia_smi.available,
        nvidia_driver_version=nvidia_smi.driver_version,
        driver_cuda_version=nvidia_smi.cuda_version or torch.cuda_version,
    )
