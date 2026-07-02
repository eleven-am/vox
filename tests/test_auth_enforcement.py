from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.server.auth import (
    is_metadata_authorized,
    require_ws_api_key,
)
from vox.server.middleware import ApiKeyAuthMiddleware
from vox.server.routes.health import router as health_router
from vox.server.routes.models import router as models_router


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ApiKeyAuthMiddleware)
    registry = MagicMock()
    registry.available_models.return_value = {}
    app.state.registry = registry
    app.state.scheduler = MagicMock()
    app.state.scheduler.list_loaded.return_value = []
    app.state.store = MagicMock()
    app.state.store.list_models.return_value = []
    app.include_router(health_router)
    app.include_router(models_router)
    return app


def test_open_when_no_key_configured(monkeypatch):
    monkeypatch.delenv("VOX_API_KEY", raising=False)
    client = TestClient(_build_app())
    assert client.get("/v1/models").status_code == 200


def test_protected_route_rejects_without_key(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    client = TestClient(_build_app())

    unauthorized = client.get("/v1/models")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["detail"] == "missing or invalid API key"

    authorized = client.get("/v1/models", headers={"authorization": "Bearer secret"})
    assert authorized.status_code == 200


def test_protected_route_accepts_x_api_key_header(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    client = TestClient(_build_app())
    assert client.get("/v1/models", headers={"x-api-key": "secret"}).status_code == 200


def test_health_is_exempt_from_auth(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    client = TestClient(_build_app())
    assert client.get("/v1/health").status_code == 200


def test_metadata_authorization(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    assert is_metadata_authorized((("authorization", "Bearer secret"),)) is True
    assert is_metadata_authorized((("x-api-key", "secret"),)) is True
    assert is_metadata_authorized((("x-api-key", "wrong"),)) is False
    assert is_metadata_authorized(()) is False


def test_metadata_authorization_open_without_key(monkeypatch):
    monkeypatch.delenv("VOX_API_KEY", raising=False)
    assert is_metadata_authorized(()) is True


class _FakeWebSocket:
    def __init__(self, headers=None, query=None):
        self.headers = headers or {}
        self.query_params = query or {}
        self.closed_code = None

    async def close(self, code=1000, reason=""):
        self.closed_code = code


@pytest.mark.asyncio
async def test_require_ws_api_key_rejects_without_key(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    ws = _FakeWebSocket()
    assert await require_ws_api_key(ws) is False
    assert ws.closed_code == 1008


@pytest.mark.asyncio
async def test_require_ws_api_key_accepts_valid_key(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    ws = _FakeWebSocket(headers={"x-api-key": "secret"})
    assert await require_ws_api_key(ws) is True
    assert ws.closed_code is None


@pytest.mark.asyncio
async def test_require_ws_api_key_open_without_key(monkeypatch):
    monkeypatch.delenv("VOX_API_KEY", raising=False)
    ws = _FakeWebSocket()
    assert await require_ws_api_key(ws) is True
