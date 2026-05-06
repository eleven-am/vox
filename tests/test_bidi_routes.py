from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.errors import ModelNotFoundError
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    TranscribeResult,
    TranscriptSegment,
)
from vox.server.routes import bidi


class FakeChunkingSTTAdapter(STTAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt",
            type=ModelType.STT,
            architectures=("fake",),
            default_sample_rate=16000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls += 1
        return TranscribeResult(
            text=f"chunk {self.calls}",
            language="en",
            duration_ms=int(len(audio) / 16000 * 1000),
            segments=(
                TranscriptSegment(
                    text=f"chunk {self.calls}",
                    start_ms=0,
                    end_ms=int(len(audio) / 16000 * 1000),
                ),
            ),
        )


class FakeStreamingTTSAdapter(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts",
            type=ModelType.TTS,
            architectures=("fake",),
            default_sample_rate=24000,
            supported_formats=(ModelFormat.ONNX,),
            supports_streaming=True,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    async def synthesize(self, text, **kwargs):
        audio_a = np.zeros(2400, dtype=np.float32)
        audio_b = np.ones(2400, dtype=np.float32) * 0.25
        yield SynthesizeChunk(audio=audio_a.tobytes(), sample_rate=24000, is_final=False)
        yield SynthesizeChunk(audio=audio_b.tobytes(), sample_rate=24000, is_final=True)


class MockScheduler:
    def __init__(self) -> None:
        self._adapters: dict[str, STTAdapter | TTSAdapter] = {}

    def register(self, model_name: str, adapter: STTAdapter | TTSAdapter) -> None:
        self._adapters[model_name] = adapter

    @asynccontextmanager
    async def acquire(self, model_name: str):
        adapter = self._adapters.get(model_name)
        if adapter is None:
            raise ModelNotFoundError(model_name)
        yield adapter


def _build_app(*, scheduler: MockScheduler, registry: Any, store: Any) -> FastAPI:
    app = FastAPI()
    app.state.scheduler = scheduler
    app.state.registry = registry
    app.state.store = store
    app.include_router(bidi.router)
    return app


class _StoreModel:
    def __init__(self, full_name: str, model_type: str) -> None:
        self.full_name = full_name
        self.type = type("T", (), {"value": model_type})()


def test_longform_stt_wire_round_trip_emits_ready_progress_done():
    scheduler = MockScheduler()
    adapter = FakeChunkingSTTAdapter()
    scheduler.register("test-stt:latest", adapter)
    store = type("Store", (), {"list_models": lambda self: [_StoreModel("test-stt:latest", "stt")]})()
    registry = type("Registry", (), {"available_models": lambda self: {}})()

    with (
        TestClient(_build_app(scheduler=scheduler, registry=registry, store=store)) as client,
        client.websocket_connect("/v1/audio/transcriptions/stream") as websocket,
    ):
        websocket.send_text(json.dumps({
            "type": "config",
            "sample_rate": 16000,
            "chunk_ms": 1000,
            "overlap_ms": 200,
        }))
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["input_format"] == "pcm16"

        audio = np.zeros(16000, dtype=np.int16).tobytes()
        websocket.send_bytes(audio)

        progress = websocket.receive_json()
        assert progress["type"] == "progress"
        assert progress["uploaded_ms"] == 1000

        websocket.send_text(json.dumps({"type": "end"}))
        done = websocket.receive_json()

    assert done["type"] == "done"
    assert done["text"] == "chunk 1"
    assert done["language"] == "en"
    assert len(done["segments"]) == 1


def test_longform_tts_wire_round_trip_emits_ready_audio_done():
    scheduler = MockScheduler()
    scheduler.register("test-tts:latest", FakeStreamingTTSAdapter())
    store = type("Store", (), {"list_models": lambda self: [_StoreModel("test-tts:latest", "tts")]})()
    registry = type("Registry", (), {"available_models": lambda self: {}})()

    with (
        TestClient(_build_app(scheduler=scheduler, registry=registry, store=store)) as client,
        client.websocket_connect("/v1/audio/speech/stream") as websocket,
    ):
        websocket.send_text(json.dumps({
            "type": "config",
            "response_format": "pcm16",
        }))
        ready = websocket.receive_json()
        assert ready["type"] == "ready"

        websocket.send_text(json.dumps({"type": "text", "text": "Hello world. " * 40}))
        websocket.send_text(json.dumps({"type": "end"}))

        audio_start = websocket.receive_json()
        assert audio_start["type"] == "audio_start"
        assert audio_start["response_format"] == "pcm16"

        saw_audio = False
        done = None
        while done is None:
            message = websocket.receive()
            if "bytes" in message:
                assert message["bytes"]
                saw_audio = True
                continue
            payload = json.loads(message["text"])
            if payload["type"] == "progress":
                continue
            if payload["type"] == "done":
                done = payload
                break
            raise AssertionError(f"Unexpected websocket payload: {payload}")

    assert saw_audio
    assert done["type"] == "done"
    assert done["response_format"] == "pcm16"


def test_longform_tts_wire_rejects_unsupported_response_format():
    scheduler = MockScheduler()
    scheduler.register("test-tts:latest", FakeStreamingTTSAdapter())
    store = type("Store", (), {"list_models": lambda self: [_StoreModel("test-tts:latest", "tts")]})()
    registry = type("Registry", (), {"available_models": lambda self: {}})()

    with (
        TestClient(_build_app(scheduler=scheduler, registry=registry, store=store)) as client,
        client.websocket_connect("/v1/audio/speech/stream") as websocket,
    ):
        websocket.send_text(json.dumps({
            "type": "config",
            "response_format": "wav",
        }))
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert "Unsupported response_format" in response["message"]


def test_longform_stt_wire_rejects_unsupported_input_format():
    scheduler = MockScheduler()
    scheduler.register("test-stt:latest", FakeChunkingSTTAdapter())
    store = type("Store", (), {"list_models": lambda self: [_StoreModel("test-stt:latest", "stt")]})()
    registry = type("Registry", (), {"available_models": lambda self: {}})()

    with (
        TestClient(_build_app(scheduler=scheduler, registry=registry, store=store)) as client,
        client.websocket_connect("/v1/audio/transcriptions/stream") as websocket,
    ):
        websocket.send_text(json.dumps({
            "type": "config",
            "input_format": "m4a",
        }))
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert "Unsupported input_format" in response["message"]
