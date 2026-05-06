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


class TestAudioStreamWireMapping:
    def test_config_with_no_default_model_emits_wire_error(self):
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

    def test_config_success_emits_ready_wire_event(self):
        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {
            "whisper": {"large-v3": {"type": "stt"}},
        }

        with (
            TestClient(_build_stream_app(store=store, registry=registry)) as client,
            client.websocket_connect("/v1/audio/stream") as websocket,
        ):
            websocket.send_text(json.dumps({"type": "config"}))
            response = websocket.receive_json()
            websocket.send_text(json.dumps({"type": "end"}))

        assert response == {"type": "ready"}

    def test_unknown_message_type_emits_wire_error(self):
        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {
            "whisper": {"large-v3": {"type": "stt"}},
        }

        with (
            TestClient(_build_stream_app(store=store, registry=registry)) as client,
            client.websocket_connect("/v1/audio/stream") as websocket,
        ):
            websocket.send_text(json.dumps({"type": "config"}))
            websocket.receive_json()
            websocket.send_text(json.dumps({"type": "totally-unknown"}))
            error = websocket.receive_json()
            websocket.send_text(json.dumps({"type": "end"}))

        assert error["type"] == "error"
        assert "totally-unknown" in error["message"]
