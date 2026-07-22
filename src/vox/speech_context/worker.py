from __future__ import annotations

import os
import resource
import sys
import time
from collections.abc import Callable
from typing import Any

from vox.core.worker_host import worker_main


def _peak_rss_bytes(value: int) -> int:
    # Linux reports KiB; macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


def run_analysis_worker(handler: Callable[[str], dict[str, Any]]) -> int:
    """Serve one isolated analyzer and attach process-scoped resource evidence."""

    def handle(request: dict[str, Any]) -> dict[str, Any]:
        if request.get("op") != "analyze":
            raise ValueError(f"unsupported speech-context operation: {request.get('op')!r}")
        audio_path = request.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError("audio_path must be a non-empty string")

        started = time.perf_counter()
        raw = handler(audio_path)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "raw": raw,
            "analysis_ms": round((time.perf_counter() - started) * 1000, 3),
            "resources": {
                "scope": "analyzer_process",
                "cpu_user_seconds": round(usage.ru_utime, 6),
                "cpu_system_seconds": round(usage.ru_stime, 6),
                "peak_rss_bytes": _peak_rss_bytes(usage.ru_maxrss),
                "gpu_peak_memory_bytes": 0,
                "gpu_status": "not_used",
                "pid": os.getpid(),
            },
        }

    return worker_main(handle)
