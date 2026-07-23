from __future__ import annotations

import asyncio
import io
import json
import tarfile
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vox.speech_context import runner
from vox.speech_context import runtime as speech_context_runtime
from vox.speech_context.reducer import SpeechContextReductionError
from vox.speech_context.runtime import RuntimeSpec, SpeechContextError
from vox.speech_context.types import SpeechContext, SpeechContextSpan


def _write_wav(path: Path, duration_ms: int = 1000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.zeros(duration_ms * 16, dtype="<i2").tobytes())


def _speaker_raw(**extra: Any) -> dict[str, Any]:
    return {
        "windows": [
            {
                "start_ms": 0,
                "end_ms": 1000,
                "language": "<|en|>",
                "emotion": "<|HAPPY|>",
                "event": "<|Laughter|>",
                "text": "diagnostic",
            }
        ],
        **extra,
    }


def _sounds_raw(**extra: Any) -> dict[str, Any]:
    return {
        "classes": [
            {
                "index": 0,
                "id": "/m/dog",
                "label": "Dog",
                "ancestor_ids": ["/m/animal"],
            }
        ],
        "scores": [
            {
                "start_ms": 0,
                "end_ms": 960,
                "values": [0.8],
            }
        ],
        "embeddings": [{"values": [0.1, 0.2]}],
        **extra,
    }


def _worker_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw": raw,
        "analysis_ms": 5.0,
        "resources": {"scope": "analyzer_process"},
        "runtime": {"status": "ready"},
    }


@pytest.mark.asyncio
async def test_evidence_runner_executes_all_tracks_concurrently_and_keeps_raw_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    audio = tmp_path / "input.wav"
    _write_wav(audio)
    barrier = asyncio.Barrier(3)
    started: set[str] = set()

    async def fake_transcription(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        started.add("transcription")
        await barrier.wait()
        return {"raw": {"text": "hello"}, "analysis_ms": 4.0}

    async def fake_worker(spec: RuntimeSpec, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        started.add(spec.key)
        await barrier.wait()
        return _worker_payload(_speaker_raw() if spec.key == "speaker" else _sounds_raw())

    monkeypatch.setattr(runner, "_run_transcription", fake_transcription)
    monkeypatch.setattr(runner, "_run_worker", fake_worker)

    evidence = await runner.collect_speech_context_evidence(
        audio,
        base_url="http://vox.test",
    )

    assert started == {"transcription", "speaker", "sounds"}
    assert evidence["execution"]["mode"] == "concurrent"
    assert evidence["results"]["speaker"]["raw"]["windows"][0]["text"] == "diagnostic"
    assert evidence["results"]["sounds"]["raw"]["embeddings"] == [{"values": [0.1, 0.2]}]
    assert evidence["timeline"]["tracks"] == {
        "transcription": "results.transcription.raw.words and .segments",
        "speaker": "results.speaker.raw.windows",
        "sounds": "results.sounds.raw.scores, .embeddings, and .log_mel_spectrogram",
    }
    assert evidence["speech_context"] == {
        "schema_version": 2,
        "status": "complete",
        "emotions": [{"label": "happy", "start_ms": 0, "end_ms": 1000}],
        "vocal": [{"label": "laughter", "start_ms": 0, "end_ms": 1000}],
        "sounds": [{"label": "dog", "start_ms": 0, "end_ms": 960, "score": 0.8}],
        "reduction_ms": evidence["speech_context"]["reduction_ms"],
    }


@pytest.mark.asyncio
async def test_backend_failure_is_explicit_without_erasing_other_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    audio = tmp_path / "input.wav"
    _write_wav(audio)

    async def failed_transcription(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Parakeet is unavailable")

    async def fake_worker(spec: RuntimeSpec, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _worker_payload(_speaker_raw() if spec.key == "speaker" else _sounds_raw())

    monkeypatch.setattr(runner, "_run_transcription", failed_transcription)
    monkeypatch.setattr(runner, "_run_worker", fake_worker)

    evidence = await runner.collect_speech_context_evidence(
        audio,
        base_url="http://vox.test",
    )

    assert evidence["results"]["transcription"]["status"] == "failed"
    assert evidence["results"]["speaker"]["status"] == "ok"
    assert evidence["results"]["sounds"]["status"] == "ok"
    assert evidence["speech_context"]["status"] == "complete"


@pytest.mark.asyncio
async def test_service_harness_calls_production_service_without_transcription(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    audio = tmp_path / "input.wav"
    _write_wav(audio)

    class FakeService:
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["home"] == tmp_path / "home"

        async def analyze_wave_path(self, path: Path) -> SpeechContext:
            assert path.name == "input.wav"
            return SpeechContext(
                status="complete",
                emotions=(SpeechContextSpan("happy", 0, 1000),),
                vocal=(),
                sounds=(),
            )

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(runner, "SpeechContextService", FakeService)
    monkeypatch.setattr(
        runner,
        "runtime_inventory",
        lambda **_kwargs: {"speaker": {"status": "ready"}, "sounds": {"status": "ready"}},
    )

    evidence = await runner.collect_speech_context_service_evidence(
        audio,
        home=tmp_path / "home",
    )

    assert evidence["execution"]["mode"] == "speech_context_service"
    assert evidence["speech_context"] == {
        "schema_version": 2,
        "status": "complete",
        "emotions": [{"label": "happy", "start_ms": 0, "end_ms": 1000}],
        "vocal": [],
        "sounds": [],
    }


def test_existing_invalid_runtime_is_rejected_instead_of_repaired(tmp_path: Path):
    spec = runner.RUNTIME_SPECS["speaker"]
    runtime = tmp_path / "runtime" / spec.directory
    runtime.mkdir(parents=True)
    sentinel = runtime / "user-state"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(
        SpeechContextError,
        match="does not match the speech-context runtime lock",
    ):
        speech_context_runtime.install_runtime(spec, home=tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_runtime_installer_stages_atomically_and_keeps_cache_outside_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = runner.RUNTIME_SPECS["speaker"]
    observed_cache: Path | None = None

    def fake_install(
        command: list[str],
        *,
        env: dict[str, str],
        timeout: int = 900,
    ) -> None:
        nonlocal observed_cache
        observed_cache = Path(env["UV_CACHE_DIR"])
        observed_cache.mkdir(parents=True, exist_ok=True)
        if command[1] == "venv":
            stage = Path(command[2])
            (stage / "bin").mkdir(parents=True)
            (stage / "bin" / "python").write_text("", encoding="utf-8")
            for relative in (
                "lib/python3.12/site-packages/numpy/__init__.py",
                "lib/python3.12/site-packages/sherpa_onnx/__init__.py",
                "lib/python3.12/site-packages/sherpa_onnx/lib/_sherpa_onnx.test.so",
            ):
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

    def fake_assets(_spec: RuntimeSpec, stage: Path) -> None:
        for relative in ("assets/model.int8.onnx", "assets/tokens.txt"):
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    monkeypatch.setattr(speech_context_runtime, "run_install", fake_install)
    monkeypatch.setattr(speech_context_runtime, "install_runtime_assets", fake_assets)

    inventory = speech_context_runtime.install_runtime(spec, home=tmp_path)

    assert inventory["status"] == "ready"
    assert observed_cache is not None
    assert not observed_cache.exists()
    assert not list((tmp_path / "runtime").glob(".*.installing-*"))


def test_runtime_marker_does_not_hide_missing_native_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = runner.RUNTIME_SPECS["speaker"]
    runtime = speech_context_runtime.runtime_path(spec, home=tmp_path)
    for relative in ("bin/python", *spec.required_files):
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    marker = runtime / ".vox-speech-context-runtime.json"
    marker.write_text(
        json.dumps(speech_context_runtime.marker_payload(spec)),
        encoding="utf-8",
    )

    assert not speech_context_runtime.runtime_is_ready(spec, runtime)


def test_runtime_marker_invalidates_changed_model_assets(tmp_path: Path):
    spec = runner.RUNTIME_SPECS["speaker"]
    runtime = speech_context_runtime.runtime_path(spec, home=tmp_path)
    for relative in (
        "bin/python",
        *spec.required_files,
        "lib/python3.12/site-packages/sherpa_onnx/lib/_sherpa_onnx.test.so",
    ):
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    marker = runtime / ".vox-speech-context-runtime.json"
    marker.write_text(
        json.dumps(speech_context_runtime.marker_payload(spec)),
        encoding="utf-8",
    )

    changed = replace(spec, asset_revision="new-model-revision")

    assert speech_context_runtime.runtime_is_ready(spec, runtime)
    assert not speech_context_runtime.runtime_is_ready(changed, runtime)


def test_sensevoice_archive_extractor_copies_only_named_files(tmp_path: Path):
    archive = tmp_path / "model.tar.bz2"
    with tarfile.open(archive, "w:bz2") as handle:
        for name, content in (
            ("root/model.onnx", b"model"),
            ("root/tokens.txt", b"tokens"),
            ("../../escape", b"blocked"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            handle.addfile(info, io.BytesIO(content))

    destination = tmp_path / "assets"
    speech_context_runtime.extract_archive_files(
        archive,
        destination,
        {
            "root/model.onnx": "model.onnx",
            "root/tokens.txt": "tokens.txt",
        },
    )

    assert (destination / "model.onnx").read_bytes() == b"model"
    assert (destination / "tokens.txt").read_bytes() == b"tokens"
    assert not (tmp_path.parent / "escape").exists()


def test_worker_environment_excludes_application_pythonpath(tmp_path: Path, monkeypatch):
    spec = runner.RUNTIME_SPECS["speaker"]
    monkeypatch.setenv("PYTHONPATH", "/polluted")
    monkeypatch.setenv("VOX_API_KEY", "secret")

    environment = speech_context_runtime.worker_environment(spec, tmp_path)

    assert environment["PYTHONPATH"] == str(speech_context_runtime.package_import_root())
    assert environment["VOX_SPEECH_CONTEXT_ASSETS"] == str(tmp_path / "assets")
    assert "VOX_API_KEY" not in environment


def test_runtime_inventory_uses_new_track_names(tmp_path: Path):
    inventory = speech_context_runtime.runtime_inventory(home=tmp_path)

    assert set(inventory) == {"speaker", "sounds"}
    assert inventory["speaker"]["status"] == "missing"
    assert inventory["sounds"]["status"] == "missing"


def test_reducer_failure_is_recorded_without_losing_raw_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    audio = tmp_path / "input.wav"
    _write_wav(audio)

    async def fake_transcription(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"raw": {"text": "hello"}}

    async def fake_worker(spec: RuntimeSpec, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _worker_payload(_speaker_raw() if spec.key == "speaker" else _sounds_raw())

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise SpeechContextReductionError("malformed analyzer output")

    monkeypatch.setattr(runner, "_run_transcription", fake_transcription)
    monkeypatch.setattr(runner, "_run_worker", fake_worker)
    monkeypatch.setattr(runner, "reduce_speech_context", fail)

    evidence = asyncio.run(runner.collect_speech_context_evidence(audio, base_url="http://vox.test"))

    assert evidence["results"]["speaker"]["raw"]["windows"]
    assert evidence["results"]["sounds"]["raw"]["classes"]
    assert evidence["speech_context"]["status"] == "failed"
    assert evidence["speech_context"]["error"]["message"] == "malformed analyzer output"
