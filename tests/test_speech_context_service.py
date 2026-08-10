from __future__ import annotations

import asyncio
import logging
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from vox.audio.pipeline import AudioChunk
from vox.speech_context.runtime import RuntimeSpec
from vox.speech_context.service import SpeechContextService, cancel_speech_context_task
from vox.speech_context.types import SpeechContext, speech_context_payload


class _Host:
    def __init__(self, spec: RuntimeSpec, barrier: threading.Barrier | None = None) -> None:
        self.spec = spec
        self.barrier = barrier
        self.alive = True
        self.calls = 0
        self.frames: list[int] = []

    def request(self, payload: dict, *, timeout: float) -> dict:
        assert payload["op"] == "analyze_compact"
        assert timeout > 0
        self.calls += 1
        with wave.open(payload["audio_path"], "rb") as handle:
            self.frames.append(handle.getnframes())
        if self.barrier is not None:
            self.barrier.wait(timeout=1)
        if self.spec.key == "speaker":
            return {
                "result": {
                    "emotions": [
                        {"label": "happy", "start_ms": 0, "end_ms": 200},
                    ],
                    "vocal": [
                        {"label": "laughter", "start_ms": 100, "end_ms": 300},
                    ],
                }
            }
        return {
            "result": {
                "sounds": [
                    {"label": "dog", "start_ms": 400, "end_ms": 600, "score": 0.75},
                ]
            }
        }

    def close(self) -> None:
        self.alive = False


class _BlockingHost(_Host):
    def __init__(self, spec: RuntimeSpec) -> None:
        super().__init__(spec)
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.audio_paths: list[Path] = []

    def request(self, payload: dict, *, timeout: float) -> dict:
        self.audio_paths.append(Path(payload["audio_path"]))
        self.started.set()
        self.release.wait(timeout)
        if self.closed.is_set():
            raise RuntimeError("closed")
        return super().request(payload, timeout=timeout)

    def close(self) -> None:
        self.closed.set()
        self.release.set()
        super().close()


async def _wait_for_hosts(hosts: dict[str, _BlockingHost], count: int) -> None:
    async with asyncio.timeout(1):
        while len(hosts) < count or not all(host.started.is_set() for host in hosts.values()):
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_service_runs_tracks_concurrently_reuses_workers_and_preserves_timeline():
    barrier = threading.Barrier(2)
    hosts: dict[str, _Host] = {}

    def factory(spec: RuntimeSpec) -> _Host:
        host = _Host(spec, barrier)
        hosts[spec.key] = host
        return host

    service = SpeechContextService(host_factory=factory)
    chunks = (
        AudioChunk(
            data=np.full(8_000, 0.1, dtype=np.float32),
            sample_rate=16_000,
            duration_ms=500,
            offset_ms=1_000,
        ),
        AudioChunk(
            data=np.full(8_000, 0.2, dtype=np.float32),
            sample_rate=16_000,
            duration_ms=500,
            offset_ms=2_000,
        ),
    )

    first = await service.analyze_chunks(chunks, timeline_offset_ms=1_000)
    second = await service.analyze_chunks(chunks, timeline_offset_ms=1_000)
    payload = speech_context_payload(first)

    assert payload == {
        "schema_version": 2,
        "status": "complete",
        "emotions": [{"label": "happy", "start_ms": 1000, "end_ms": 1200}],
        "vocal": [{"label": "laughter", "start_ms": 1100, "end_ms": 1300}],
        "sounds": [{"label": "dog", "start_ms": 1400, "end_ms": 1600, "score": 0.75}],
    }
    assert speech_context_payload(second) == payload
    assert set(hosts) == {"speaker", "sounds"}
    assert all(host.calls == 2 for host in hosts.values())
    assert all(host.frames == [24_000, 24_000] for host in hosts.values())

    await service.close()
    assert all(not host.alive for host in hosts.values())


@pytest.mark.asyncio
async def test_service_preload_starts_both_workers_without_running_analysis():
    hosts: dict[str, _Host] = {}

    def factory(spec: RuntimeSpec) -> _Host:
        host = _Host(spec)
        hosts[spec.key] = host
        return host

    service = SpeechContextService(host_factory=factory)

    await service.preload()
    await service.preload()

    assert set(hosts) == {"speaker", "sounds"}
    assert all(host.calls == 0 for host in hosts.values())
    await service.close()


class _FailingHost(_Host):
    def request(self, payload: dict, *, timeout: float) -> dict:
        if self.spec.key == "sounds":
            raise RuntimeError("unavailable")
        return super().request(payload, timeout=timeout)


class _DiagnosticHost(_Host):
    def request(self, payload: dict, *, timeout: float) -> dict:
        response = super().request(payload, timeout=timeout)
        if self.spec.key == "speaker":
            response["result"]["_pre_reduction"] = {
                "windows": [
                    {
                        "start_ms": 0,
                        "end_ms": 2500,
                        "emotion": "<|SAD|>",
                        "event": "<|Speech|>",
                        "text": "internal only",
                    }
                ]
            }
        else:
            response["result"]["_pre_reduction"] = {
                "chunks": [
                    {
                        "offset_ms": 0,
                        "class_maxima": [
                            {"label": "Crying, sobbing", "score": 0.0412},
                        ],
                    }
                ]
            }
        return response


class _MalformedSpeakerHost(_Host):
    def request(self, payload: dict, *, timeout: float) -> dict:
        response = super().request(payload, timeout=timeout)
        if self.spec.key == "speaker":
            response["result"]["vocal"] = [
                {"label": "laughter", "start_ms": 200, "end_ms": 100},
            ]
        return response


class _MalformedSoundScoreHost(_Host):
    def request(self, payload: dict, *, timeout: float) -> dict:
        response = super().request(payload, timeout=timeout)
        if self.spec.key == "sounds":
            response["result"]["sounds"][0]["score"] = 1.1
        return response


@pytest.mark.asyncio
async def test_service_reports_partial_without_leaking_internal_error():
    service = SpeechContextService(host_factory=lambda spec: _FailingHost(spec))
    chunk = AudioChunk(
        data=np.full(1_600, 0.1, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=100,
        offset_ms=0,
    )

    payload = speech_context_payload(await service.analyze_chunks((chunk,)))

    assert payload == {
        "schema_version": 2,
        "status": "partial",
        "emotions": [{"label": "happy", "start_ms": 0, "end_ms": 200}],
        "vocal": [{"label": "laughter", "start_ms": 100, "end_ms": 300}],
        "unavailable": ["sounds"],
    }
    await service.close()


@pytest.mark.asyncio
async def test_service_logs_compact_track_completion_without_payloads(caplog):
    service = SpeechContextService(host_factory=lambda spec: _DiagnosticHost(spec))
    chunk = AudioChunk(
        data=np.full(16_000, 0.1, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=1_000,
        offset_ms=0,
    )

    with caplog.at_level(logging.INFO, logger="vox.speech_context.service"):
        context = await service.analyze_chunks((chunk,))

    assert "speech context SenseVoice complete analysis_ms=" in caplog.text
    assert "speech context YAMNet complete analysis_ms=" in caplog.text
    assert "internal only" not in caplog.text
    assert "Crying, sobbing" not in caplog.text
    assert "pre-reduction" not in caplog.text
    assert "_pre_reduction" not in speech_context_payload(context)
    await service.close()


@pytest.mark.asyncio
async def test_service_rejects_malformed_speaker_track_atomically():
    service = SpeechContextService(host_factory=lambda spec: _MalformedSpeakerHost(spec))
    chunk = AudioChunk(
        data=np.full(1_600, 0.1, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=100,
        offset_ms=0,
    )

    payload = speech_context_payload(await service.analyze_chunks((chunk,)))

    assert payload == {
        "schema_version": 2,
        "status": "partial",
        "sounds": [{"label": "dog", "start_ms": 400, "end_ms": 600, "score": 0.75}],
        "unavailable": ["speaker"],
    }
    await service.close()


@pytest.mark.asyncio
async def test_service_rejects_out_of_range_sound_score():
    service = SpeechContextService(host_factory=lambda spec: _MalformedSoundScoreHost(spec))
    chunk = AudioChunk(
        data=np.full(1_600, 0.1, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=100,
        offset_ms=0,
    )

    payload = speech_context_payload(await service.analyze_chunks((chunk,)))

    assert payload == {
        "schema_version": 2,
        "status": "partial",
        "emotions": [{"label": "happy", "start_ms": 0, "end_ms": 200}],
        "vocal": [{"label": "laughter", "start_ms": 100, "end_ms": 300}],
        "unavailable": ["sounds"],
    }
    await service.close()


@pytest.mark.asyncio
async def test_service_analyzes_existing_wave_without_owning_the_file(tmp_path: Path):
    audio_path = tmp_path / "input.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 3_200)
    service = SpeechContextService(host_factory=lambda spec: _Host(spec))

    context = await service.analyze_wave_path(audio_path, timeline_offset_ms=500)

    assert audio_path.is_file()
    assert context.sounds is not None
    assert context.sounds[0].start_ms == 900
    await service.close()


@pytest.mark.asyncio
async def test_empty_audio_fails_both_tracks_without_starting_workers():
    service = SpeechContextService(host_factory=lambda spec: pytest.fail(spec.key))

    context = await service.analyze_chunks(())

    assert context == SpeechContext(
        status="failed",
        unavailable=("speaker", "sounds"),
    )
    await service.close()


@pytest.mark.asyncio
async def test_context_task_cancellation_is_drained():
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def analyze() -> SpeechContext:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return SpeechContext(status="failed", unavailable=("speaker", "sounds"))

    task = asyncio.create_task(analyze())
    await started.wait()
    await cancel_speech_context_task(task)

    assert task.cancelled()
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_service_rejects_analysis_beyond_count_admission_limit():
    hosts: dict[str, _BlockingHost] = {}

    def factory(spec: RuntimeSpec) -> _BlockingHost:
        host = _BlockingHost(spec)
        hosts[spec.key] = host
        return host

    service = SpeechContextService(host_factory=factory)
    chunk = AudioChunk(
        data=np.full(1_600, 0.1, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=100,
        offset_ms=0,
    )
    first = asyncio.create_task(service.analyze_chunks((chunk,)))
    await _wait_for_hosts(hosts, 2)
    second = asyncio.create_task(service.analyze_chunks((chunk,)))
    await asyncio.sleep(0)
    third = asyncio.create_task(service.analyze_chunks((chunk,)))

    try:
        async with asyncio.timeout(0.1):
            with pytest.raises(RuntimeError, match="capacity"):
                await third
    finally:
        for host in hosts.values():
            host.release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
async def test_service_rejects_analysis_beyond_audio_admission_limit():
    service = SpeechContextService(
        host_factory=lambda spec: _Host(spec),
        max_admitted_audio_bytes=6_399,
    )
    chunk = AudioChunk(
        data=np.full(1_600, 0.1, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=100,
        offset_ms=0,
    )

    with pytest.raises(RuntimeError, match="capacity"):
        await service.analyze_chunks((chunk,))

    await service.close()


@pytest.mark.asyncio
async def test_context_task_cancellation_stops_workers_and_releases_temporary_audio():
    hosts: dict[str, _BlockingHost] = {}

    def factory(spec: RuntimeSpec) -> _BlockingHost:
        host = _BlockingHost(spec)
        hosts[spec.key] = host
        return host

    service = SpeechContextService(host_factory=factory)
    chunk = AudioChunk(
        data=np.full(1_600, 0.1, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=100,
        offset_ms=0,
    )
    task = asyncio.create_task(service.analyze_chunks((chunk,)))
    await _wait_for_hosts(hosts, 2)
    paths = [path for host in hosts.values() for path in host.audio_paths]

    await cancel_speech_context_task(task)

    try:
        assert all(host.closed.is_set() for host in hosts.values())
        assert all(not path.exists() for path in paths)
    finally:
        for host in hosts.values():
            host.release.set()
        await service.close()
