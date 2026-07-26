from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, UploadFile
from starlette.responses import JSONResponse

DEFAULT_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
UPLOAD_PATHS = frozenset(
    {
        "/v1/audio/transcriptions",
        "/v1/audio/voices",
        "/v1/voices",
    }
)


class _UploadLimitExceeded(Exception):
    pass


class UploadSizeLimitMiddleware:
    def __init__(
        self,
        app: Any,
        *,
        multipart_overhead_bytes: int = MULTIPART_OVERHEAD_BYTES,
    ) -> None:
        self._app = app
        self._multipart_overhead_bytes = max(0, int(multipart_overhead_bytes))

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in UPLOAD_PATHS:
            await self._app(scope, receive, send)
            return
        app = scope["app"]
        upload_limit = int(app.state.max_upload_bytes)
        request_limit = upload_limit + self._multipart_overhead_bytes
        content_length = _content_length(scope.get("headers", ()))
        if content_length is not None and content_length > request_limit:
            await _upload_limit_response(upload_limit, scope, receive, send)
            return

        received = 0
        exceeded = False
        response_messages: list[dict] = []

        async def limited_receive() -> dict:
            nonlocal exceeded, received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > request_limit:
                    exceeded = True
                    raise _UploadLimitExceeded
            return message

        async def limited_send(message: dict) -> None:
            response_messages.append(message)

        try:
            await self._app(scope, limited_receive, limited_send)
        except _UploadLimitExceeded:
            exceeded = True
        if exceeded:
            await _upload_limit_response(upload_limit, scope, receive, send)
            return
        for message in response_messages:
            await send(message)


def configured_max_upload_bytes() -> int:
    raw = os.environ.get("VOX_MAX_UPLOAD_BYTES")
    if raw is None:
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("VOX_MAX_UPLOAD_BYTES must be a positive integer") from exc
    if value <= 0:
        raise ValueError("VOX_MAX_UPLOAD_BYTES must be a positive integer")
    return value


def _upload_too_large(limit: int) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"upload exceeds the {limit} byte limit",
    )


async def read_upload_limited(
    upload: UploadFile,
    *,
    max_bytes: int | None = None,
) -> bytes:
    limit = configured_max_upload_bytes() if max_bytes is None else max_bytes
    if limit <= 0:
        raise ValueError("max_bytes must be positive")
    if upload.size is not None and upload.size > limit:
        raise _upload_too_large(limit)
    data = bytearray()
    while True:
        remaining = limit - len(data)
        chunk = await upload.read(min(UPLOAD_READ_CHUNK_BYTES, remaining + 1))
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if len(data) > limit:
            raise _upload_too_large(limit)


def _content_length(headers: tuple[tuple[bytes, bytes], ...] | list[tuple[bytes, bytes]]) -> int | None:
    for name, value in headers:
        if name.lower() != b"content-length":
            continue
        try:
            return max(0, int(value))
        except ValueError:
            return None
    return None


async def _upload_limit_response(
    limit: int,
    scope: dict,
    receive: Any,
    send: Any,
) -> None:
    response = JSONResponse(
        {"detail": f"upload exceeds the {limit} byte limit"},
        status_code=413,
    )
    await response(scope, receive, send)
