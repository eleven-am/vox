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


def _prosody() -> dict:
    return {
        "pitch": {"mean_st": 1.0, "median_st": 0.5, "range_st": 4.0, "variation": 0.2},
        "energy": {"mean": -20.0, "range": 8.0, "peaks_per_second": 2.0},
        "voice_quality": {"hnr_db": 12.0, "jitter": None, "shimmer_db": 0.4},
        "spectral_variation": 0.3,
        "delivery": {
            "voiced_segments_per_second": 1.5,
            "mean_voiced_ms": 300.0,
            "mean_unvoiced_ms": 120.0,
        },
    }


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
        if self.spec.key == "prosody":
            return {"result": _prosody()}
        return {
            "result": {
                "candidates": [
                    {"label": "Laughter", "spans": [[0, 120, 0.9]]},
                ]
            }
        }

    def close(self) -> None:
        self.alive = False


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

    assert payload["status"] == "complete"
    assert payload["audio_events"]["candidates"] == [
        {"label": "Laughter", "spans": [[1_000, 1_120, 0.9]]},
    ]
    assert payload["prosody"]["voice_quality"]["jitter"] is None
    assert set(payload) == {"schema_version", "status", "prosody", "audio_events"}
    assert speech_context_payload(second) == payload
    assert set(hosts) == {"prosody", "audio_events"}
    assert all(host.calls == 2 for host in hosts.values())
    assert all(host.frames == [24_000, 24_000] for host in hosts.values())

    await service.close()
    assert all(not host.alive for host in hosts.values())


class _FailingHost(_Host):
    def request(self, payload: dict, *, timeout: float) -> dict:
        if self.spec.key == "audio_events":
            raise RuntimeError("unavailable")
        return super().request(payload, timeout=timeout)


class _DiagnosticHost(_Host):
    def request(self, payload: dict, *, timeout: float) -> dict:
        response = super().request(payload, timeout=timeout)
        if self.spec.key == "audio_events":
            response["result"]["_pre_reduction"] = {
                "chunks": [
                    {
                        "offset_ms": 0,
                        "frame_count": 1,
                        "omitted_frame_count": 0,
                        "frames": [
                            {
                                "start_ms": 0.0,
                                "end_ms": 960.0,
                                "candidates": [{"label": "Crying, sobbing", "score": 0.0412}],
                            }
                        ],
                        "class_maxima": [{"label": "Crying, sobbing", "score": 0.0412}],
                    }
                ]
            }
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

    context = await service.analyze_chunks((chunk,))
    payload = speech_context_payload(context)

    assert payload["status"] == "partial"
    assert payload["unavailable"] == ["audio_events"]
    assert "prosody" in payload
    assert "audio_events" not in payload
    assert "error" not in payload

    await service.close()


@pytest.mark.asyncio
async def test_service_logs_pre_reduction_yamnet_diagnostic_without_exposing_it(caplog):
    service = SpeechContextService(host_factory=lambda spec: _DiagnosticHost(spec))
    chunk = AudioChunk(
        data=np.full(16_000, 0.1, dtype=np.float32),
        sample_rate=16_000,
        duration_ms=1_000,
        offset_ms=0,
    )

    with caplog.at_level(logging.INFO, logger="vox.speech_context.service"):
        context = await service.analyze_chunks((chunk,))

    assert "speech context YAMNet pre-reduction payload=" in caplog.text
    assert '"label":"Crying, sobbing","score":0.0412' in caplog.text
    assert context.audio_events is not None
    assert [candidate.label for candidate in context.audio_events.candidates] == ["Laughter"]
    assert "_pre_reduction" not in speech_context_payload(context)

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

    assert context.status == "complete"
    assert audio_path.is_file()
    assert context.audio_events is not None
    assert context.audio_events.candidates[0].spans[0].start_ms == 500

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
        return SpeechContext(status="failed", unavailable=("prosody", "audio_events"))

    task = asyncio.create_task(analyze())
    await started.wait()

    await cancel_speech_context_task(task)

    assert task.cancelled()
    assert stopped.is_set()
