from __future__ import annotations

import pytest

from vox.operations.errors import UnknownMessageTypeError
from vox.server.websocket import safe_send_ws_error, send_ws_error, send_ws_operation_error


class RecordingWebSocket:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("closed")
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_send_ws_error_uses_shared_error_envelope():
    websocket = RecordingWebSocket()

    await send_ws_error(websocket, "boom")

    assert websocket.sent == [{"type": "error", "message": "boom"}]


@pytest.mark.asyncio
async def test_send_ws_operation_error_uses_shared_error_envelope():
    websocket = RecordingWebSocket()

    await send_ws_operation_error(websocket, UnknownMessageTypeError("bad.type"))

    assert websocket.sent == [{
        "type": "error",
        "message": "Unknown message type: bad.type",
    }]


@pytest.mark.asyncio
async def test_safe_send_ws_error_suppresses_closed_socket_errors():
    await safe_send_ws_error(RecordingWebSocket(fail=True), "boom")
