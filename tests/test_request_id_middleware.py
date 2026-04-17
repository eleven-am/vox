"""Tests for the FastAPI request-id middleware."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.logging_config import configure_logging, reset_for_tests
from vox.logging_context import request_id_var
from vox.server.middleware import HEADER, RequestIdMiddleware


@pytest.fixture(autouse=True)
def _reset_logging():
    reset_for_tests()
    yield
    reset_for_tests()


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/hello")
    async def hello():
        return {"rid": request_id_var.get()}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    return app


class TestRequestIdHeader:
    def test_generates_header_when_missing(self):
        client = TestClient(_build_app())
        resp = client.get("/hello")
        assert resp.status_code == 200
        assert HEADER in resp.headers
        assert resp.headers[HEADER]
        assert resp.json()["rid"] == resp.headers[HEADER]

    def test_preserves_incoming_header(self):
        client = TestClient(_build_app())
        resp = client.get("/hello", headers={HEADER: "caller-supplied"})
        assert resp.headers[HEADER] == "caller-supplied"
        assert resp.json()["rid"] == "caller-supplied"

    def test_empty_incoming_header_generates_new(self):
        client = TestClient(_build_app())
        resp = client.get("/hello", headers={HEADER: "   "})
        assert resp.headers[HEADER]
        assert resp.headers[HEADER].strip() != ""


class TestAccessLog:
    def test_successful_request_logs_once(self, caplog):
        caplog.set_level(logging.INFO, logger="vox.server.request")
        client = TestClient(_build_app())
        client.get("/hello")

        request_logs = [r for r in caplog.records if r.name == "vox.server.request"]
        assert len(request_logs) == 1
        assert "GET /hello -> 200" in request_logs[0].message

    def test_health_path_is_quiet(self, caplog):
        caplog.set_level(logging.INFO, logger="vox.server.request")
        client = TestClient(_build_app())
        client.get("/healthz")
        assert not [r for r in caplog.records if r.name == "vox.server.request"]

    def test_failing_request_logs_exception(self, caplog):
        caplog.set_level(logging.INFO, logger="vox.server.request")
        client = TestClient(_build_app(), raise_server_exceptions=False)
        client.get("/boom")

        records = [r for r in caplog.records if r.name == "vox.server.request"]
        assert any(r.levelno >= logging.ERROR for r in records)


class TestContextVarLifecycle:
    def test_rid_cleared_after_request(self):
        client = TestClient(_build_app())
        assert request_id_var.get() == "-"
        client.get("/hello")
        assert request_id_var.get() == "-"
