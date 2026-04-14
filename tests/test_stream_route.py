from __future__ import annotations

import json
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.server.routes import stream


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
