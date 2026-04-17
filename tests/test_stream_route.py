from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.server.routes import stream
from vox.streaming.types import StreamTranscript


def _build_stream_app(*, store=None, registry=None, scheduler=None) -> FastAPI:
    app = FastAPI()
    app.state.store = store or MagicMock()
    app.state.registry = registry or MagicMock()
    app.state.scheduler = scheduler or MagicMock()
    app.include_router(stream.router)
    return app


class TestAudioStreamConfig:
    def test_config_rejects_missing_model_when_no_default_exists(self):
        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {}

        with (
            TestClient(_build_stream_app(store=store, registry=registry)) as client,
            client.websocket_connect("/v1/audio/stream") as websocket,
        ):
            websocket.send_text(json.dumps({"type": "config"}))
            response = websocket.receive_json()

        assert response == {
            "type": "error",
            "message": "No STT model specified and no default STT model available",
        }

    def test_config_uses_default_model_when_available(self):
        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {
            "whisper": {
                "large-v3": {"type": "stt"},
            }
        }

        with (
            TestClient(_build_stream_app(store=store, registry=registry)) as client,
            client.websocket_connect("/v1/audio/stream") as websocket,
        ):
            websocket.send_text(json.dumps({"type": "config"}))
            response = websocket.receive_json()
            websocket.send_text(json.dumps({"type": "end"}))

        assert response == {"type": "ready"}

    def test_end_flush_preserves_language_and_word_timestamp_options(self, monkeypatch):
        calls: list[dict] = []

        class FakePipeline:
            def __init__(self, scheduler):
                self.scheduler = scheduler

            def configure(self, config):
                self.config = config

            async def transcribe_async(self, **kwargs):
                calls.append(kwargs)
                return StreamTranscript(
                    text="final transcript",
                    words=[{"word": "final", "start_ms": 0, "end_ms": 100}],
                    segments=[{"text": "final transcript", "start_ms": 0, "end_ms": 100, "words": []}],
                )

            def reset(self):
                pass

            def shutdown(self):
                pass

        class FakePartialService:
            def __init__(self, transcribe_async_fn):
                self.transcribe_async_fn = transcribe_async_fn

            def flush_remaining_audio(self, session):
                return np.zeros(1600, dtype=np.float32)

        monkeypatch.setattr(stream, "StreamPipeline", FakePipeline)
        monkeypatch.setattr(stream, "PartialTranscriptService", FakePartialService)
        monkeypatch.setattr(stream, "enrich_transcript", lambda transcript, language: None)

        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {"whisper": {"large-v3": {"type": "stt"}}}

        with (
            TestClient(_build_stream_app(store=store, registry=registry)) as client,
            client.websocket_connect("/v1/audio/stream") as websocket,
        ):
            websocket.send_text(json.dumps({
                "type": "config",
                "language": "fr",
                "include_word_timestamps": True,
            }))
            ready = websocket.receive_json()
            websocket.send_text(json.dumps({"type": "end"}))
            transcript = websocket.receive_json()

        assert ready == {"type": "ready"}
        assert len(calls) == 1
        assert isinstance(calls[0]["audio"], np.ndarray)
        assert calls[0]["language"] == "fr"
        assert calls[0]["word_timestamps"] is True
        assert transcript["text"] == "final transcript"
        assert transcript["words"][0]["word"] == "final"
        assert transcript["segments"][0]["text"] == "final transcript"
