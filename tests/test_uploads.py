from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from vox.server.uploads import (
    UploadSizeLimitMiddleware,
    configured_max_upload_bytes,
    read_upload_limited,
)


def test_configured_max_upload_bytes_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("VOX_MAX_UPLOAD_BYTES", "invalid")
    with pytest.raises(ValueError, match="positive integer"):
        configured_max_upload_bytes()

    monkeypatch.setenv("VOX_MAX_UPLOAD_BYTES", "0")
    with pytest.raises(ValueError, match="positive integer"):
        configured_max_upload_bytes()


@pytest.mark.asyncio
async def test_read_upload_limited_accepts_exact_boundary():
    upload = UploadFile(file=BytesIO(b"12345678"), size=None, filename="audio.wav")

    assert await read_upload_limited(upload, max_bytes=8) == b"12345678"


@pytest.mark.asyncio
async def test_read_upload_limited_rejects_stream_past_boundary_without_declared_size():
    upload = UploadFile(file=BytesIO(b"123456789"), size=None, filename="audio.wav")

    with pytest.raises(HTTPException) as exc_info:
        await read_upload_limited(upload, max_bytes=8)

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail == "upload exceeds the 8 byte limit"


@pytest.mark.asyncio
async def test_upload_middleware_rejects_declared_oversize_before_route_runs():
    route_called = False
    sent: list[dict] = []

    async def route(_scope, _receive, _send) -> None:
        nonlocal route_called
        route_called = True

    async def receive() -> dict:
        raise AssertionError("oversized request body was read")

    async def send(message: dict) -> None:
        sent.append(message)

    middleware = UploadSizeLimitMiddleware(route, multipart_overhead_bytes=0)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "headers": [(b"content-length", b"9")],
        "app": SimpleNamespace(state=SimpleNamespace(max_upload_bytes=8)),
    }

    await middleware(scope, receive, send)

    assert route_called is False
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"detail": "upload exceeds the 8 byte limit"}


@pytest.mark.asyncio
async def test_upload_middleware_stops_chunked_body_at_ingress_limit():
    sent: list[dict] = []
    messages = iter(
        [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": True},
            {"type": "http.request", "body": b"ignored", "more_body": False},
        ]
    )
    reads = 0

    async def route(_scope, receive, _send) -> None:
        while True:
            message = await receive()
            if not message.get("more_body"):
                return

    async def receive() -> dict:
        nonlocal reads
        reads += 1
        return next(messages)

    async def send(message: dict) -> None:
        sent.append(message)

    middleware = UploadSizeLimitMiddleware(route, multipart_overhead_bytes=0)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "headers": [],
        "app": SimpleNamespace(state=SimpleNamespace(max_upload_bytes=8)),
    }

    await middleware(scope, receive, send)

    assert reads == 2
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"detail": "upload exceeds the 8 byte limit"}


async def _multipart_response(path: str) -> tuple[list[dict], int]:
    app = FastAPI()
    app.state.max_upload_bytes = 8
    route_calls = 0

    @app.post(path)
    async def upload(request: Request):
        nonlocal route_calls
        route_calls += 1
        await request.form()
        return JSONResponse({"ok": True})

    middleware = UploadSizeLimitMiddleware(app, multipart_overhead_bytes=0)
    boundary = b"vox-boundary"
    body = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        + b"Content-Type: audio/wav\r\n\r\n123456789\r\n--"
        + boundary
        + b"--\r\n"
    )
    messages = iter(
        [
            {"type": "http.request", "body": body[:7], "more_body": True},
            {"type": "http.request", "body": body[7:16], "more_body": True},
            {"type": "http.request", "body": body[16:], "more_body": False},
        ]
    )
    sent: list[dict] = []

    async def receive() -> dict:
        return next(messages)

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"multipart/form-data; boundary=" + boundary)],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "app": app,
    }

    await middleware(scope, receive, send)
    return sent, route_calls


@pytest.mark.asyncio
async def test_upload_middleware_replaces_chunked_multipart_parser_error_with_413():
    sent, route_calls = await _multipart_response("/v1/audio/transcriptions")

    assert route_calls == 1
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"detail": "upload exceeds the 8 byte limit"}


@pytest.mark.asyncio
async def test_upload_middleware_protects_voice_clone_route():
    sent, route_calls = await _multipart_response("/v1/audio/voices")

    assert route_calls == 1
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"detail": "upload exceeds the 8 byte limit"}
