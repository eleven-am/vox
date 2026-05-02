"""Integration tests for the WS /v1/conversation endpoint.

Uses FastAPI TestClient.websocket_connect to drive a scripted agent-side
conversation against a real ConversationSession backed by mocked STT/TTS.
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.server.routes.conversation import router as conversation_router


class ScriptedTTS(TTSAdapter):
    def __init__(self, chunks: int = 3) -> None:
        self._chunks = chunks

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted-tts", type=ModelType.TTS,
            architectures=("scripted",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **_):
        for _ in range(self._chunks):
            yield SynthesizeChunk(
                audio=np.full(512, 0.01, dtype=np.float32).tobytes(),
                sample_rate=24_000, is_final=False,
            )
            await asyncio.sleep(0.01)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class DummyScheduler:
    def __init__(self, adapter): self._a = adapter

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self._a


def _build_app() -> FastAPI:
    app = FastAPI()
    app.state.scheduler = DummyScheduler(ScriptedTTS())
    app.include_router(conversation_router)
    return app


def _drain_until(ws, predicate, max_events: int = 50) -> list[dict]:
    events: list[dict] = []
    for _ in range(max_events):
        try:
            msg = ws.receive()
        except WebSocketDisconnect:
            break
        if msg.get("type") == "websocket.disconnect":
            break
        if "text" in msg and msg["text"]:
            payload = json.loads(msg["text"])
            events.append(payload)
            if predicate(payload):
                break
    return events


class TestSessionUpdate:
    def test_accepts_config_and_emits_session_created(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {
                    "stt_model": "fake-stt:1",
                    "tts_model": "fake-tts:1",
                    "voice": "default",
                    "language": "en",
                },
            })
            msg = ws.receive_json()
            assert msg["type"] == "session.created"
            assert msg["session"]["stt_model"] == "fake-stt:1"
            assert msg["session"]["tts_model"] == "fake-tts:1"
            assert msg["session"]["voice"] == "default"
            assert msg["session"]["output_sample_rate"] == 16_000
            assert msg["session"]["output_audio_format"] == "pcm16"

    def test_missing_stt_model_errors(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {"tts_model": "x:1"},
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "stt_model" in msg["message"]

    def test_double_session_update_errors(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {"stt_model": "x:1", "tts_model": "y:1"},
            })
            ws.receive_json()

            ws.send_json({
                "type": "session.update",
                "session": {"stt_model": "a:1", "tts_model": "b:1"},
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "already configured" in msg["message"]

    def test_must_send_session_update_first(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({"type": "input_audio_buffer.append", "audio": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "session.update" in msg["message"]

    def test_turn_policy_passed_through(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {
                    "stt_model": "x:1",
                    "tts_model": "y:1",
                    "turn_policy": {
                        "min_interrupt_duration_ms": 150,
                        "stable_speaking_min_ms": 100,
                    },
                },
            })
            msg = ws.receive_json()
            assert msg["session"]["turn_policy"]["min_interrupt_duration_ms"] == 150
            assert msg["session"]["turn_policy"]["stable_speaking_min_ms"] == 100


class TestResponseFlow:
    def test_streaming_response_emits_audio_and_done(self):
        """Agent sends start + delta + commit → audio deltas + done."""
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {
                    "stt_model": "x:1",
                    "tts_model": "y:1",
                    "voice": "default",
                    "sample_rate": 48_000,
                },
            })
            ws.receive_json()

            ws.send_json({"type": "response.start"})
            ws.send_json({"type": "response.delta", "delta": "hi there"})
            ws.send_json({"type": "response.commit"})

            events = _drain_until(ws, lambda e: e.get("type") == "response.done")
            types = [e["type"] for e in events]
            assert "response.created" in types
            assert "response.committed" in types
            assert "response.audio.delta" in types
            assert "response.done" in types

    def test_audio_delta_payload_is_base64_pcm(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {
                    "stt_model": "x:1",
                    "tts_model": "y:1",
                    "voice": "default",
                    "sample_rate": 48_000,
                },
            })
            ws.receive_json()

            ws.send_json({"type": "response.delta", "delta": "hi"})
            ws.send_json({"type": "response.commit"})
            events = _drain_until(ws, lambda e: e.get("type") == "response.done")

            deltas = [e for e in events if e["type"] == "response.audio.delta"]
            assert deltas
            for d in deltas:
                decoded = base64.b64decode(d["audio"])
                assert len(decoded) > 0
                assert d["sample_rate"] == 48_000
                assert d["audio_format"] == "pcm16"
                assert np.frombuffer(decoded, dtype=np.int16).size > 512

    def test_response_delta_without_delta_field_errors(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {"stt_model": "x:1", "tts_model": "y:1"},
            })
            ws.receive_json()

            ws.send_json({"type": "response.delta"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "delta" in msg["message"]

    def test_streamed_response_delta_starts_audio_before_commit(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
            })
            ws.receive_json()

            ws.send_json({"type": "response.delta", "delta": "Hello world. Still pending"})
            early_events = _drain_until(ws, lambda e: e.get("type") == "response.audio.delta")
            early_types = [e["type"] for e in early_events]
            assert "response.created" in early_types
            assert "response.audio.delta" in early_types
            assert "response.done" not in early_types

            ws.send_json({"type": "response.commit"})
            final_events = _drain_until(ws, lambda e: e.get("type") == "response.done")
            assert any(e["type"] == "response.done" for e in final_events)


class TestBadInput:
    def test_invalid_json_frame_returns_error(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_text("not json")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "invalid JSON" in msg["message"]

    def test_unknown_type_returns_error(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {"stt_model": "x:1", "tts_model": "y:1"},
            })
            ws.receive_json()

            ws.send_json({"type": "make.coffee"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown message type" in msg["message"]

    def test_missing_type_returns_error(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({"no_type_here": "oops"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "missing 'type'" in msg["message"]

    def test_invalid_base64_audio_returns_error(self):
        client = TestClient(_build_app())
        with client.websocket_connect("/v1/conversation") as ws:
            ws.send_json({
                "type": "session.update",
                "session": {"stt_model": "x:1", "tts_model": "y:1"},
            })
            ws.receive_json()

            ws.send_json({"type": "input_audio_buffer.append", "audio": "not valid base64!!!"})
