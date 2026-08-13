from __future__ import annotations

import sys
from types import SimpleNamespace

import vox.core.process_memory as memory


def test_process_memory_status_reads_current_and_peak_bytes(monkeypatch):
    values = {"VmRSS": 1_000, "VmHWM": 2_000}
    monkeypatch.setattr(memory, "_read_proc_status_bytes", lambda key, pid=None: values[key])
    monkeypatch.setattr(memory.os, "getpid", lambda: 41)

    assert memory.process_memory_status() == {
        "pid": 41,
        "rss_bytes": 1_000,
        "peak_rss_bytes": 2_000,
    }


def test_cgroup_memory_status_reports_current_peak_and_limit(monkeypatch):
    values = {
        "memory.current": 3_000,
        "memory.peak": 4_000,
        "memory.max": 5_000,
    }
    monkeypatch.setattr(memory, "_read_cgroup_value", values.__getitem__)

    assert memory.cgroup_memory_status() == {
        "current_bytes": 3_000,
        "peak_bytes": 4_000,
        "limit_bytes": 5_000,
    }


def test_trim_process_memory_releases_python_cuda_and_native_caches(monkeypatch):
    calls: list[str] = []
    snapshots = iter(
        [
            {"rss_bytes": 2_000, "torch_reserved_bytes": 4_000},
            {"rss_bytes": 1_000, "torch_reserved_bytes": 2_000},
        ]
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(memory, "runtime_memory_status", lambda **_kwargs: next(snapshots))
    monkeypatch.setattr(memory.gc, "collect", lambda: calls.append("gc") or 7)
    monkeypatch.setattr(memory, "_malloc_trim", lambda: calls.append("malloc_trim") or True)

    result = memory.trim_process_memory(device="cuda")

    assert calls == ["gc", "synchronize", "empty_cache", "ipc_collect", "malloc_trim"]
    assert result == {
        "before": {"rss_bytes": 2_000, "torch_reserved_bytes": 4_000},
        "after": {"rss_bytes": 1_000, "torch_reserved_bytes": 2_000},
        "gc_collected": 7,
        "malloc_trimmed": True,
    }
