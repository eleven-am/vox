from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vox.core.adapter import TTSAdapter
from vox.core.store import BlobStore
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.server.routes.bidi import router as bidi_router


LONG_TEXT = (
    "In the quiet morning light the farmer walked to the fields, thinking of the "
    "planting season ahead, and the challenges of the dry summer that had passed. "
    "He remembered his father teaching him how to read the soil and the weather, "
    "lessons that had served him well through decades of harvests. "
    "Now it was his turn to pass them on to his own children, who would soon be "
    "old enough to take the reins of the farm themselves."
)


class _CapturingTTS(TTSAdapter):
    def __init__(self, max_input_chars: int = 0) -> None:
        self._cap = max_input_chars
        self.calls: list[str] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="cap-tts", type=ModelType.TTS,
            architectures=("cap",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            max_input_chars=self._cap,
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **_):
        self.calls.append(text)
        yield SynthesizeChunk(
            audio=np.zeros(512, dtype=np.float32).tobytes(),
            sample_rate=24_000, is_final=False,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class _DummyScheduler:
    def __init__(self, adapter):
        self._adapter = adapter

    @asynccontextmanager
    async def acquire(self, _model):
        yield self._adapter

    def list_loaded(self):
        return []


def _build_app(adapter, tmp_path: Path):
    app = FastAPI()
    app.state.store = BlobStore(root=tmp_path)
    reg = MagicMock()
    reg.available_models.return_value = {"cap-tts": {"latest": {"type": "tts"}}}
    reg.resolve_model_ref.side_effect = lambda n, t, explicit_tag=False: (n, t or "latest")
    app.state.registry = reg
    app.state.scheduler = _DummyScheduler(adapter)
    app.include_router(bidi_router)
    return app


def _drain(ws, max_events: int = 50) -> list[dict]:
    msgs: list[dict] = []
    for _ in range(max_events):
        try:
            msg = ws.receive()
        except WebSocketDisconnect:
            break
        if msg.get("type") == "websocket.disconnect":
            break
        if "text" in msg and msg["text"]:
            payload = json.loads(msg["text"])
            msgs.append(payload)
            if payload.get("type") == "done":
                break
    return msgs


class TestAdapterDeclaredCap:
    def test_cap_zero_means_no_prechunking(self, tmp_path: Path):
        """max_input_chars=0 → entire text in one synthesize() call, even for long input."""
        adapter = _CapturingTTS(max_input_chars=0)
        app = _build_app(adapter, tmp_path)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "cap-tts:latest",
                "voice": "default",
                "response_format": "pcm16",
            })
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["chunk_chars"] == 0

            ws.send_json({"type": "text", "text": LONG_TEXT})
            ws.send_json({"type": "end"})
            _drain(ws)


        assert len(adapter.calls) == 1
        assert adapter.calls[0] == LONG_TEXT

    def test_cap_splits_at_sentence_boundaries(self, tmp_path: Path):
        """Non-zero cap forces chunking on sentence boundaries when text exceeds the cap."""
        adapter = _CapturingTTS(max_input_chars=200)
        app = _build_app(adapter, tmp_path)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "cap-tts:latest",
                "voice": "default",
                "response_format": "pcm16",
            })
            assert ws.receive_json()["chunk_chars"] == 200
            ws.send_json({"type": "text", "text": LONG_TEXT})
            ws.send_json({"type": "end"})
            _drain(ws)


        assert len(adapter.calls) >= 2
        for chunk in adapter.calls:

            assert len(chunk) <= 300

    def test_cap_short_text_sent_whole(self, tmp_path: Path):
        """If total text fits in adapter cap, it goes in one call."""
        adapter = _CapturingTTS(max_input_chars=2000)
        app = _build_app(adapter, tmp_path)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "cap-tts:latest",
                "voice": "default",
                "response_format": "pcm16",
            })
            ws.receive_json()
            ws.send_json({"type": "text", "text": LONG_TEXT})
            ws.send_json({"type": "end"})
            _drain(ws)

        assert len(adapter.calls) == 1
        assert adapter.calls[0] == LONG_TEXT


class TestClientOverride:
    def test_client_override_wins_over_adapter(self, tmp_path: Path):
        """Explicit chunk_chars in config overrides adapter's declared cap."""
        adapter = _CapturingTTS(max_input_chars=2000)
        app = _build_app(adapter, tmp_path)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "cap-tts:latest",
                "voice": "default",
                "response_format": "pcm16",
                "chunk_chars": 150,
            })
            ready = ws.receive_json()
            assert ready["chunk_chars"] == 150

            ws.send_json({"type": "text", "text": LONG_TEXT})
            ws.send_json({"type": "end"})
            _drain(ws)

        assert len(adapter.calls) >= 3

    def test_client_override_zero_forces_single_call(self, tmp_path: Path):
        """Client can force single-call synthesis by passing chunk_chars=0, even for small caps."""
        adapter = _CapturingTTS(max_input_chars=100)
        app = _build_app(adapter, tmp_path)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "cap-tts:latest",
                "voice": "default",
                "response_format": "pcm16",
                "chunk_chars": 0,
            })
            ready = ws.receive_json()
            assert ready["chunk_chars"] == 0

            ws.send_json({"type": "text", "text": LONG_TEXT})
            ws.send_json({"type": "end"})
            _drain(ws)

        assert len(adapter.calls) == 1

    def test_invalid_chunk_chars_returns_error(self, tmp_path: Path):
        adapter = _CapturingTTS(max_input_chars=0)
        app = _build_app(adapter, tmp_path)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "cap-tts:latest",
                "voice": "default",
                "response_format": "pcm16",
                "chunk_chars": "not-a-number",
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "chunk_chars" in msg["message"]


class TestKokoroDefaults:
    def test_kokoro_adapter_declares_zero_cap(self):
        """Verify the Kokoro adapter opts out of HTTP-layer chunking (library chunks internally)."""
        from unittest.mock import patch
        with patch.dict("sys.modules", {
            "onnxruntime": MagicMock(),
            "kokoro_onnx": MagicMock(),
        }):
            from vox_kokoro.adapter import KokoroAdapter
            info = KokoroAdapter().info()
            assert info.max_input_chars == 0


class TestQwen3Defaults:
    def test_qwen3_tts_adapter_declares_zero_cap(self):
        from unittest.mock import patch
        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "qwen_asr": MagicMock(),
            "qwen_tts": MagicMock(),
        }):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter
            info = Qwen3TTSAdapter().info()
            assert info.max_input_chars == 0
