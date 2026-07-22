from __future__ import annotations

import asyncio
import json
import math
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from vox.speech_context import runner
from vox.speech_context.opensmile_worker import _json_number
from vox.speech_context.runner import RuntimeSpec, SpeechContextError


def _write_wav(path: Path, *, duration_ms: int = 1_200, frequency: float = 220.0) -> None:
    samples = int(16_000 * duration_ms / 1000)
    timeline = np.arange(samples, dtype=np.float32) / 16_000
    waveform = np.sin(2 * math.pi * frequency * timeline) * 0.25
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes((waveform * 32767).astype("<i2").tobytes())


def _worker_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw": raw,
        "analysis_ms": 12.5,
        "startup_ms": 4.0,
        "resources": {
            "scope": "analyzer_process",
            "cpu_user_seconds": 0.01,
            "cpu_system_seconds": 0.002,
            "peak_rss_bytes": 1024,
            "gpu_peak_memory_bytes": 0,
            "gpu_status": "not_used",
        },
        "runtime": {"runtime_bytes": 2048, "model_bytes": 512, "status": "ready"},
    }


def _transcription_payload(text: str = "thinking out loud") -> dict[str, Any]:
    return {
        "raw": {
            "text": text,
            "duration": 1.2,
            "words": [
                {"word": "thinking", "start": 0.0, "end": 0.45},
                {"word": "out", "start": 0.7, "end": 0.9},
                {"word": "loud", "start": 0.95, "end": 1.2},
            ],
            "segments": [{"text": text, "start": 0.0, "end": 1.2}],
        },
        "analysis_ms": 20.0,
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


@pytest.mark.asyncio
async def test_all_analyzers_start_concurrently_and_preserve_complete_raw_outputs(tmp_path, monkeypatch):
    audio = tmp_path / "input.wav"
    _write_wav(audio)
    started: list[str] = []
    all_started = asyncio.Event()
    canonical_paths: list[Path] = []

    async def wait_for_peers(name: str, path: Path) -> None:
        started.append(name)
        canonical_paths.append(path)
        if len(started) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)
        await asyncio.sleep(0.02)

    async def fake_transcription(path: Path, **_kwargs: Any) -> dict[str, Any]:
        await wait_for_peers("transcription", path)
        return _transcription_payload()

    async def fake_worker(spec: RuntimeSpec, path: Path, **_kwargs: Any) -> dict[str, Any]:
        await wait_for_peers(spec.key, path)
        if spec.key == "prosody":
            return _worker_payload({
                "low_level_descriptors": {
                    "columns": ["pitch", "loudness"],
                    "frames": [{"start_ms": 0.0, "end_ms": 20.0, "values": [3.0, 4.0]}],
                },
                "functionals": {
                    "columns": ["pitch_mean"],
                    "frames": [{"start_ms": 0.0, "end_ms": 1200.0, "values": [3.0]}],
                },
            })
        return _worker_payload({
            "classes": [{"index": 0, "id": "/m/speech", "label": "Speech"}],
            "scores": [{"start_ms": 0.0, "end_ms": 960.0, "values": [0.75]}],
            "embeddings": [{"start_ms": 0.0, "end_ms": 960.0, "values": [0.1, 0.2]}],
            "log_mel_spectrogram": [
                {"start_ms": 0.0, "end_ms": 25.0, "values": [-1.0, -2.0]}
            ],
        })

    monkeypatch.setattr(runner, "_run_transcription", fake_transcription)
    monkeypatch.setattr(runner, "_run_worker", fake_worker)

    evidence = await runner.collect_speech_context_evidence(audio, base_url="http://vox.test")

    assert set(started) == {"transcription", "prosody", "audio_events"}
    assert evidence["execution"]["mode"] == "concurrent"
    assert evidence["execution"]["result_elapsed_sum_ms"] > evidence["execution"]["elapsed_ms"]
    assert evidence["results"]["transcription"]["raw"]["words"][1]["word"] == "out"
    assert evidence["results"]["prosody"]["raw"]["low_level_descriptors"]["frames"][0][
        "values"
    ] == [3.0, 4.0]
    assert evidence["results"]["audio_events"]["raw"]["embeddings"][0]["values"] == [0.1, 0.2]
    assert evidence["timeline"] == {
        "origin_ms": 0,
        "duration_ms": 1200.0,
        "unit": "milliseconds",
        "tracks": {
            "transcription": "results.transcription.raw.words and .segments",
            "prosody": "results.prosody.raw.low_level_descriptors.frames and .functionals.frames",
            "audio_events": "results.audio_events.raw.scores, .embeddings, and .log_mel_spectrogram",
        },
    }
    assert "vad" not in evidence
    assert len({path for path in canonical_paths}) == 1
    assert canonical_paths[0].name == "input.wav"
    assert not canonical_paths[0].exists()


@pytest.mark.asyncio
async def test_backend_failure_is_explicit_without_erasing_other_evidence(tmp_path, monkeypatch):
    audio = tmp_path / "input.wav"
    _write_wav(audio)

    async def failed_transcription(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Parakeet is unavailable")

    async def fake_worker(spec: RuntimeSpec, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _worker_payload({"backend": spec.key, "complete": [1, 2, 3]})

    monkeypatch.setattr(runner, "_run_transcription", failed_transcription)
    monkeypatch.setattr(runner, "_run_worker", fake_worker)

    evidence = await runner.collect_speech_context_evidence(audio, base_url="http://vox.test")

    assert evidence["results"]["transcription"]["status"] == "failed"
    assert evidence["results"]["transcription"]["error"] == {
        "type": "RuntimeError",
        "message": "Parakeet is unavailable",
    }
    assert evidence["results"]["prosody"]["status"] == "ok"
    assert evidence["results"]["prosody"]["raw"]["complete"] == [1, 2, 3]
    assert evidence["results"]["audio_events"]["status"] == "ok"


@pytest.mark.parametrize(
    ("case", "duration_ms", "raw_event"),
    [
        ("silence", 1_000, "Silence"),
        ("thinking_pause", 2_200, "Speech"),
        ("laughter", 1_400, "Laughter"),
        ("sighing", 1_100, "Breathing"),
        ("background_noise", 1_600, "Vehicle"),
        ("overlapping_events", 1_900, ["Speech", "Music"]),
        ("short_audio", 80, "Speech"),
        ("long_audio", 8_000, "Narration"),
    ],
)
@pytest.mark.asyncio
async def test_adversarial_audio_conditions_preserve_unfiltered_backend_evidence(
    case: str,
    duration_ms: int,
    raw_event: str | list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """This proves transport/preservation, not that a mocked model classified the signal correctly."""

    audio = tmp_path / f"{case}.wav"
    _write_wav(audio, duration_ms=duration_ms)

    async def fake_transcription(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _transcription_payload(case)

    async def fake_worker(spec: RuntimeSpec, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _worker_payload({
            "case": case,
            "raw_event": raw_event,
            "all_scores": [0.01, 0.33, 0.89, 0.02],
            "backend": spec.key,
        })

    monkeypatch.setattr(runner, "_run_transcription", fake_transcription)
    monkeypatch.setattr(runner, "_run_worker", fake_worker)

    evidence = await runner.collect_speech_context_evidence(audio, base_url="http://vox.test")

    assert evidence["input"]["duration_ms"] == duration_ms
    assert evidence["results"]["transcription"]["raw"]["text"] == case
    for key in ("prosody", "audio_events"):
        assert evidence["results"][key]["raw"]["raw_event"] == raw_event
        assert evidence["results"][key]["raw"]["all_scores"] == [0.01, 0.33, 0.89, 0.02]


def test_install_requires_explicit_opensmile_license_acceptance(tmp_path):
    with pytest.raises(SpeechContextError, match="explicit license acknowledgement"):
        runner.install_experimental_runtimes(
            accept_opensmile_research_license=False,
            home=tmp_path,
        )
    assert not (tmp_path / "runtime").exists()


def test_existing_invalid_runtime_is_rejected_instead_of_repaired(tmp_path):
    spec = runner.RUNTIME_SPECS["prosody"]
    runtime = tmp_path / "runtime" / spec.directory
    runtime.mkdir(parents=True)
    sentinel = runtime / "user-state"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(SpeechContextError, match="does not match the experiment lock"):
        runner._install_runtime(spec, home=tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_runtime_installer_keeps_uv_cache_outside_the_venv_stage(tmp_path, monkeypatch):
    spec = runner.RUNTIME_SPECS["prosody"]
    observed_cache: Path | None = None

    def fake_install(command: list[str], *, env: dict[str, str], timeout: int = 900) -> None:
        nonlocal observed_cache
        observed_cache = Path(env["UV_CACHE_DIR"])
        observed_cache.mkdir(parents=True, exist_ok=True)
        if command[1] == "venv":
            stage = Path(command[2])
            assert not stage.exists()
            (stage / "bin").mkdir(parents=True)
            (stage / "bin" / "python").write_text("", encoding="utf-8")
            for relative in spec.required_files:
                required = stage / relative
                required.parent.mkdir(parents=True, exist_ok=True)
                required.write_text("", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_install", fake_install)

    inventory = runner._install_runtime(spec, home=tmp_path)

    assert inventory["status"] == "ready"
    assert observed_cache is not None
    assert not observed_cache.exists()


def test_runtime_marker_does_not_hide_missing_load_bearing_files(tmp_path):
    spec = runner.RUNTIME_SPECS["audio_events"]
    runtime = tmp_path / "runtime" / spec.directory
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "python").write_text("", encoding="utf-8")
    for relative in spec.required_files:
        required = runtime / relative
        required.parent.mkdir(parents=True, exist_ok=True)
        required.write_text("", encoding="utf-8")
    (runtime / ".vox-speech-context-runtime.json").write_text(
        json.dumps(runner._marker_payload(spec)),
        encoding="utf-8",
    )

    assert runner._runtime_is_ready(spec, runtime)
    (runtime / "assets" / "yamnet_class_map.csv").unlink()
    assert not runner._runtime_is_ready(spec, runtime)


def test_worker_environment_prevents_runtime_growth_from_bytecode(tmp_path):
    spec = runner.RUNTIME_SPECS["prosody"]
    environment = runner._worker_environment(spec, tmp_path)

    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.asyncio
async def test_transcription_upload_uses_async_safe_bytes_and_complete_timestamps(tmp_path, monkeypatch):
    audio = tmp_path / "input.wav"
    _write_wav(audio)
    observed: dict[str, Any] = {}
    real_client = httpx.AsyncClient

    async def handle(request: httpx.Request) -> httpx.Response:
        observed["request"] = request
        observed["body"] = await request.aread()
        return httpx.Response(
            200,
            json={"text": "complete", "words": [], "segments": []},
        )

    transport = httpx.MockTransport(handle)

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        observed["client"] = kwargs
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(runner.httpx, "AsyncClient", client_factory)

    result = await runner._run_transcription(
        audio,
        base_url="http://vox.test/",
        api_key="secret",
        model="parakeet-stt:tdt-0.6b-v3",
        timeout=30,
    )

    assert observed["client"] == {
        "base_url": "http://vox.test",
        "headers": {"Authorization": "Bearer secret"},
        "timeout": 30,
    }
    request = observed["request"]
    assert request.url == httpx.URL("http://vox.test/v1/audio/transcriptions")
    assert request.headers["Authorization"] == "Bearer secret"
    body = observed["body"]
    assert audio.read_bytes() in body
    assert body.count(b'name="timestamp_granularities[]"') == 2
    assert b"\r\nword\r\n" in body
    assert b"\r\nsegment\r\n" in body
    assert result["raw"] == {"text": "complete", "words": [], "segments": []}


def test_canonicalization_uses_one_mono_16khz_pcm16_timeline(tmp_path):
    source = tmp_path / "source.wav"
    destination = tmp_path / "canonical.wav"
    _write_wav(source, duration_ms=375)

    evidence = runner._canonicalize_audio(source, destination)

    with wave.open(str(destination), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == 6_000
    assert evidence["duration_ms"] == 375.0
    assert evidence["sha256"] != evidence["canonical_sha256"]


def test_nonfinite_prosody_values_remain_explicit_in_strict_json():
    values = [_json_number(value) for value in (math.nan, math.inf, -math.inf, 1.25)]

    assert values == ["NaN", "Infinity", "-Infinity", 1.25]
    assert json.loads(json.dumps(values, allow_nan=False)) == values


def test_runtime_inventory_reports_explicit_missing_state(tmp_path):
    inventory = runner.runtime_inventory(home=tmp_path)
    assert inventory["prosody"] == {
        "status": "missing",
        "path": str(tmp_path / "runtime" / "speech-context-prosody"),
        "runtime_bytes": 0,
        "model_bytes": 0,
        "license": "audEERING Research License",
    }
    assert inventory["audio_events"]["status"] == "missing"


def test_runtime_marker_is_lock_derived_from_exact_requirements(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("package==1.2.3\n", encoding="utf-8")
    spec = RuntimeSpec(
        key="test",
        directory="test-runtime",
        requirements_file=requirements.name,
        module="test.worker",
        license="Apache-2.0",
    )
    monkeypatch.setattr(runner, "_requirements_path", lambda _spec: requirements)

    marker = runner._marker_payload(spec)

    assert marker == {
        "schema_version": 1,
        "requirements_sha256": runner._sha256_file(requirements),
        "python": "3.12",
        "license": "Apache-2.0",
    }
    assert json.dumps(marker, sort_keys=True)
