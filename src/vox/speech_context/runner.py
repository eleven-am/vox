from __future__ import annotations

import asyncio
import contextlib
import hashlib
import tempfile
import time
import wave
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from vox.audio.pipeline import prepare_for_stt
from vox.core.worker_host import WorkerHost
from vox.speech_context.reducer import SpeechContextReductionError, reduce_speech_context
from vox.speech_context.runtime import (
    RUNTIME_SPECS,
    RuntimeSpec,
    SpeechContextError,
    create_worker_host,
)
from vox.speech_context.runtime import (
    runtime_details as _runtime_inventory,
)
from vox.speech_context.runtime import (
    runtime_path as _runtime_path,
)
from vox.streaming.codecs import float32_to_pcm16

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "parakeet-stt:tdt-0.6b-v3"
SCHEMA_VERSION = 2


def _canonicalize_audio(source: Path, destination: Path) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    waveform = prepare_for_stt(source_bytes, target_rate=SAMPLE_RATE, format_hint=source.suffix.lstrip(".") or None)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(float32_to_pcm16(waveform))
    canonical_bytes = destination.read_bytes()
    return {
        "path": str(source.resolve()),
        "bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "canonical_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "duration_ms": round(len(waveform) / SAMPLE_RATE * 1000, 3),
    }


async def _run_worker(spec: RuntimeSpec, audio_path: Path, *, home: Path | None, timeout: float) -> dict[str, Any]:
    runtime = _runtime_path(spec, home=home)

    def run() -> dict[str, Any]:
        host: WorkerHost | None = None
        startup_started = time.perf_counter()
        try:
            host = create_worker_host(
                spec,
                home=home,
                startup_timeout=timeout,
            )
            startup_ms = round((time.perf_counter() - startup_started) * 1000, 3)
            response = host.request({"op": "analyze", "audio_path": str(audio_path)}, timeout=timeout)
            response["startup_ms"] = startup_ms
            response["runtime"] = _runtime_inventory(spec, runtime)
            return response
        finally:
            if host is not None:
                with contextlib.suppress(Exception):
                    host.close()

    return await asyncio.to_thread(run)


async def _run_transcription(
    audio_path: Path,
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")},
            data={
                "model": model,
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"],
            },
        )
    response.raise_for_status()
    return {
        "raw": response.json(),
        "analysis_ms": round((time.perf_counter() - started) * 1000, 3),
        "resources": {
            "scope": "remote_service",
            "cpu_user_seconds": None,
            "cpu_system_seconds": None,
            "peak_rss_bytes": None,
            "gpu_peak_memory_bytes": None,
            "gpu_status": "unavailable_across_http_boundary",
        },
        "runtime": {
            "runtime_bytes": None,
            "model_bytes": None,
            "status": "unavailable_across_http_boundary",
        },
    }


async def _capture(
    name: str,
    operation: Awaitable[dict[str, Any]],
    *,
    experiment_started: float,
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    try:
        payload = await operation
    except Exception as error:
        return name, {
            "status": "failed",
            "started_offset_ms": round((started - experiment_started) * 1000, 3),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    return name, {
        "status": "ok",
        "started_offset_ms": round((started - experiment_started) * 1000, 3),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        **payload,
    }


async def collect_speech_context_evidence(
    audio_file: Path,
    *,
    base_url: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = 300.0,
    home: Path | None = None,
) -> dict[str, Any]:
    if not audio_file.is_file():
        raise SpeechContextError(f"audio file does not exist: {audio_file}")

    with tempfile.TemporaryDirectory(prefix="vox-speech-context-") as directory:
        canonical_path = Path(directory) / "input.wav"
        input_evidence = _canonicalize_audio(audio_file, canonical_path)
        started = time.perf_counter()
        captures = await asyncio.gather(
            _capture(
                "transcription",
                _run_transcription(
                    canonical_path,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                ),
                experiment_started=started,
            ),
            *(
                _capture(
                    key,
                    _run_worker(spec, canonical_path, home=home, timeout=timeout),
                    experiment_started=started,
                )
                for key, spec in RUNTIME_SPECS.items()
            ),
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

    results = dict(captures)
    reduction_started = time.perf_counter()
    try:
        speech_context = reduce_speech_context(
            results,
            duration_ms=input_evidence["duration_ms"],
        )
    except SpeechContextReductionError as error:
        speech_context = {
            "schema_version": 1,
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    speech_context["reduction_ms"] = round(
        (time.perf_counter() - reduction_started) * 1000,
        3,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "input": input_evidence,
        "execution": {
            "mode": "concurrent",
            "elapsed_ms": elapsed_ms,
            "result_elapsed_sum_ms": round(
                sum(float(result.get("elapsed_ms", 0)) for result in results.values()),
                3,
            ),
        },
        "timeline": {
            "origin_ms": 0,
            "duration_ms": input_evidence["duration_ms"],
            "unit": "milliseconds",
            "tracks": {
                "transcription": "results.transcription.raw.words and .segments",
                "prosody": "results.prosody.raw.low_level_descriptors.frames and .functionals.frames",
                "audio_events": ("results.audio_events.raw.scores, .embeddings, and .log_mel_spectrogram"),
            },
        },
        "results": results,
        "speech_context": speech_context,
    }
