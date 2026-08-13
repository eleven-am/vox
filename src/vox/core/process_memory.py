from __future__ import annotations

import ctypes
import gc
import os
import sys
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:
    resource = None


def _read_proc_status_bytes(key: str, *, pid: int | None = None) -> int | None:
    target = "self" if pid is None else str(pid)
    try:
        lines = Path(f"/proc/{target}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    prefix = f"{key}:"
    for line in lines:
        if not line.startswith(prefix):
            continue
        fields = line[len(prefix) :].split()
        if not fields:
            return None
        multiplier = 1024 if len(fields) == 1 or fields[1].lower() == "kb" else 1
        try:
            return int(fields[0]) * multiplier
        except ValueError:
            return None
    return None


def _resource_peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def process_memory_status(*, pid: int | None = None) -> dict[str, Any]:
    rss_bytes = _read_proc_status_bytes("VmRSS", pid=pid)
    peak_rss_bytes = _read_proc_status_bytes("VmHWM", pid=pid)
    if pid is None and peak_rss_bytes is None:
        peak_rss_bytes = _resource_peak_rss_bytes()
    return {
        "pid": os.getpid() if pid is None else pid,
        "rss_bytes": rss_bytes,
        "peak_rss_bytes": peak_rss_bytes,
    }


def _read_cgroup_value(name: str) -> int | None:
    try:
        raw = (Path("/sys/fs/cgroup") / name).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def cgroup_memory_status() -> dict[str, int | None]:
    return {
        "current_bytes": _read_cgroup_value("memory.current"),
        "peak_bytes": _read_cgroup_value("memory.peak"),
        "limit_bytes": _read_cgroup_value("memory.max"),
    }


def accelerator_memory_status(device: str) -> dict[str, int | None]:
    allocated_bytes = None
    reserved_bytes = None
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                allocated_bytes = int(torch.cuda.memory_allocated())
                reserved_bytes = int(torch.cuda.memory_reserved())
        except (ImportError, RuntimeError):
            pass
    return {
        "torch_allocated_bytes": allocated_bytes,
        "torch_reserved_bytes": reserved_bytes,
    }


def runtime_memory_status(*, device: str, pid: int | None = None) -> dict[str, Any]:
    return {
        **process_memory_status(pid=pid),
        **accelerator_memory_status(device),
    }


def _malloc_trim() -> bool:
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError):
        return False


def trim_process_memory(*, device: str) -> dict[str, Any]:
    before = runtime_memory_status(device=device)
    collected = gc.collect()
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        except (ImportError, RuntimeError):
            pass
    malloc_trimmed = _malloc_trim()
    after = runtime_memory_status(device=device)
    return {
        "before": before,
        "after": after,
        "gc_collected": collected,
        "malloc_trimmed": malloc_trimmed,
    }
