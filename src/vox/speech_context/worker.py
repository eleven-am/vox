from __future__ import annotations

import os
import resource
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

from vox.core.worker_host import worker_main


def _peak_rss_bytes(value: int) -> int:
    # Linux reports KiB; macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


def run_analysis_worker(handlers: Mapping[str, Callable[[str], dict[str, Any]]]) -> int:
    def handle(request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        handler = handlers.get(str(operation))
        if handler is None:
            raise ValueError(f"unsupported speech-context operation: {operation!r}")
        audio_path = request.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError("audio_path must be a non-empty string")

        started = time.perf_counter()
        result = handler(audio_path)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "raw" if operation == "analyze" else "result": result,
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
