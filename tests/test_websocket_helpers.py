from __future__ import annotations

import pytest

from vox.operations.errors import UnknownMessageTypeError
from vox.server.websocket import (
    iter_ws_json_messages,
    safe_send_ws_error,
    send_ws_error,
    send_ws_operation_error,
)


class RecordingWebSocket:
    def __init__(self, *, fail: bool = False, incoming: list[dict] | None = None) -> None:
        self.fail = fail
        self.incoming = list(incoming or [])
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("closed")
        self.sent.append(payload)

    async def receive(self) -> dict:
        if self.incoming:
            return self.incoming.pop(0)
        return {"type": "websocket.disconnect"}


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


@pytest.mark.asyncio
async def test_iter_ws_json_messages_yields_valid_json_until_disconnect():
    websocket = RecordingWebSocket(
        incoming=[
            {"text": '{"type":"response.start"}'},
            {"type": "websocket.disconnect"},
        ]
    )

    messages = [message async for message in iter_ws_json_messages(websocket)]

    assert messages == [{"type": "response.start"}]
    assert websocket.sent == []


@pytest.mark.asyncio
async def test_iter_ws_json_messages_reports_non_text_frames_and_continues():
    websocket = RecordingWebSocket(
        incoming=[
            {"bytes": b"not-json"},
            {"text": '{"type":"response.commit"}'},
            {"type": "websocket.disconnect"},
        ]
    )

    messages = [message async for message in iter_ws_json_messages(websocket)]

    assert messages == [{"type": "response.commit"}]
    assert websocket.sent == [{"type": "error", "message": "only JSON text frames are supported"}]


@pytest.mark.asyncio
async def test_iter_ws_json_messages_reports_invalid_json_and_continues():
    websocket = RecordingWebSocket(
        incoming=[
            {"text": "{"},
            {"text": '{"type":"response.cancel"}'},
            {"type": "websocket.disconnect"},
        ]
    )

    messages = [message async for message in iter_ws_json_messages(websocket)]

    assert messages == [{"type": "response.cancel"}]
    assert websocket.sent
    assert websocket.sent[0]["type"] == "error"
    assert "invalid JSON" in websocket.sent[0]["message"]
