from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import wave
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from vox.audio.pipeline import prepare_for_stt
from vox.core.adapter_runtime import runtime_root
from vox.core.worker_host import WorkerHost
from vox.streaming.codecs import float32_to_pcm16

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "parakeet-stt:tdt-0.6b-v3"
SCHEMA_VERSION = 1
RUNTIME_SCHEMA_VERSION = 1
YAMNET_MODEL_URL = "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite"
YAMNET_MODEL_SHA256 = "141fba1cdaae842c816f28edc4937e8b4f0af4c8df21862ccc6b52dc567993c3"
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "950c21457b3b3de045cb2b907e973c744e743af9/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)
YAMNET_CLASS_MAP_SHA256 = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"


class SpeechContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeSpec:
    key: str
    directory: str
    requirements_file: str
    module: str
    license: str
    model_file: str | None = None
    no_deps: bool = False
    required_files: tuple[str, ...] = ()


RUNTIME_SPECS = {
    "prosody": RuntimeSpec(
        key="prosody",
        directory="speech-context-prosody",
        requirements_file="requirements-opensmile.txt",
        module="vox.speech_context.opensmile_worker",
        license="audEERING Research License",
        no_deps=True,
        required_files=(
            "lib/python3.12/site-packages/numpy/__init__.py",
            "lib/python3.12/site-packages/opensmile/core/lib.py",
        ),
    ),
    "audio_events": RuntimeSpec(
        key="audio_events",
        directory="speech-context-audio-events",
        requirements_file="requirements-yamnet.txt",
        module="vox.speech_context.yamnet_worker",
        license="Apache-2.0",
        model_file="assets/yamnet.tflite",
        required_files=(
            "lib/python3.12/site-packages/numpy/__init__.py",
            "lib/python3.12/site-packages/ai_edge_litert/interpreter.py",
            "assets/yamnet.tflite",
            "assets/yamnet_class_map.csv",
        ),
    ),
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _requirements_path(spec: RuntimeSpec) -> Path:
    return _repository_root() / "scripts" / "speech-context" / spec.requirements_file


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            with contextlib.suppress(OSError):
                total += item.stat().st_size
    return total


def _marker_payload(spec: RuntimeSpec) -> dict[str, Any]:
    requirements = _requirements_path(spec)
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "requirements_sha256": _sha256_file(requirements),
        "python": "3.12",
        "license": spec.license,
    }


def _runtime_path(spec: RuntimeSpec, *, home: Path | None = None) -> Path:
    return runtime_root(home=home) / spec.directory


def _runtime_is_ready(spec: RuntimeSpec, path: Path) -> bool:
    marker_path = path / ".vox-speech-context-runtime.json"
    python = path / "bin" / "python"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        python.is_file()
        and marker == _marker_payload(spec)
        and all((path / relative).is_file() for relative in spec.required_files)
    )


def _run_install(command: list[str], *, env: dict[str, str], timeout: int = 900) -> None:
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise SpeechContextError(f"runtime installation failed: {' '.join(command)}: {detail}")


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    actual = _sha256_file(destination)
    if actual != expected_sha256:
        raise SpeechContextError(
            f"download checksum mismatch for {destination.name}: expected {expected_sha256}, got {actual}"
        )


def _install_runtime(spec: RuntimeSpec, *, home: Path | None = None) -> dict[str, Any]:
    target = _runtime_path(spec, home=home)
    if _runtime_is_ready(spec, target):
        return _runtime_inventory(spec, target)
    if target.exists():
        raise SpeechContextError(
            f"{target} exists but does not match the experiment lock; remove that experimental runtime explicitly"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{target.name}.installing-{os.getpid()}")
    if stage.exists():
        raise SpeechContextError(f"stale installation directory exists: {stage}")
    cache = target.with_name(f".{target.name}.uv-cache-{os.getpid()}")
    if cache.exists():
        raise SpeechContextError(f"stale installation cache exists: {cache}")
    env = {**os.environ, "UV_CACHE_DIR": str(cache)}
    try:
        _run_install(["uv", "venv", str(stage), "--python", "3.12"], env=env, timeout=300)
        _run_install(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(stage / "bin" / "python"),
                *(["--no-deps"] if spec.no_deps else []),
                "--requirement",
                str(_requirements_path(spec)),
            ],
            env=env,
        )
        if spec.key == "audio_events":
            assets = stage / "assets"
            assets.mkdir()
            _download_verified(YAMNET_MODEL_URL, assets / "yamnet.tflite", YAMNET_MODEL_SHA256)
            _download_verified(
                YAMNET_CLASS_MAP_URL,
                assets / "yamnet_class_map.csv",
                YAMNET_CLASS_MAP_SHA256,
            )
        (stage / ".vox-speech-context-runtime.json").write_text(
            json.dumps(_marker_payload(spec), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return _runtime_inventory(spec, target)


def install_experimental_runtimes(
    *,
    accept_opensmile_research_license: bool,
    home: Path | None = None,
) -> dict[str, Any]:
    if not accept_opensmile_research_license:
        raise SpeechContextError(
            "openSMILE is research-only and cannot be installed without explicit license acknowledgement"
        )
    return {
        key: _install_runtime(spec, home=home)
        for key, spec in RUNTIME_SPECS.items()
    }


def _runtime_inventory(spec: RuntimeSpec, path: Path) -> dict[str, Any]:
    model_path = path / spec.model_file if spec.model_file else None
    return {
        "status": "ready" if _runtime_is_ready(spec, path) else "missing",
        "path": str(path),
        "runtime_bytes": _tree_bytes(path),
        "model_bytes": model_path.stat().st_size if model_path and model_path.is_file() else 0,
        "license": spec.license,
    }


def runtime_inventory(*, home: Path | None = None) -> dict[str, Any]:
    return {
        key: _runtime_inventory(spec, _runtime_path(spec, home=home))
        for key, spec in RUNTIME_SPECS.items()
    }


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


def _worker_environment(spec: RuntimeSpec, runtime: Path) -> dict[str, str]:
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key in {
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "PATH",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
        }
    }
    allowed["PYTHONPATH"] = str(_repository_root() / "src")
    allowed["PYTHONDONTWRITEBYTECODE"] = "1"
    if spec.key == "audio_events":
        allowed["VOX_SPEECH_CONTEXT_ASSETS"] = str(runtime / "assets")
    return allowed


async def _run_worker(spec: RuntimeSpec, audio_path: Path, *, home: Path | None, timeout: float) -> dict[str, Any]:
    runtime = _runtime_path(spec, home=home)
    if not _runtime_is_ready(spec, runtime):
        raise SpeechContextError(
            f"{spec.key} runtime is not installed; run the speech-context install command first"
        )

    def run() -> dict[str, Any]:
        host: WorkerHost | None = None
        startup_started = time.perf_counter()
        try:
            host = WorkerHost(
                [str(runtime / "bin" / "python"), "-m", spec.module],
                env=_worker_environment(spec, runtime),
                name=f"speech-context-{spec.key}",
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
                "audio_events": (
                    "results.audio_events.raw.scores, .embeddings, and .log_mel_spectrogram"
                ),
            },
        },
        "results": results,
    }
