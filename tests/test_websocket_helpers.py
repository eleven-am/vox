from __future__ import annotations

import asyncio

import pytest
from starlette.websockets import WebSocketDisconnect

from vox.operations.errors import EmptyInputError, UnknownMessageTypeError
from vox.logging_context import request_id_var
from vox.server.websocket import (
    WsBytesFrame,
    WsDisconnectFrame,
    WsTextFrame,
    bind_websocket_request_id,
    iter_ws_json_messages,
    parse_ws_frame,
    parse_ws_json_text_frame,
    receive_required_ws_config,
    receive_ws_frame,
    receive_ws_json_message,
    reset_websocket_request_id,
    emit_ws_session_events,
    safe_close_websocket,
    safe_send_ws_error,
    send_ws_error,
    send_ws_operation_error,
    send_ws_unknown_message_type,
    websocket_connection_scope,
    websocket_route_error_scope,
    websocket_session_event_scope,
    ws_operation_result_or_error,
)


class RecordingWebSocket:
    def __init__(
        self,
        *,
        fail: bool = False,
        incoming: list[dict] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.fail = fail
        self.incoming = list(incoming or [])
        self.headers = headers or {}
        self.sent: list[dict] = []
        self.sent_bytes: list[bytes] = []
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("closed")
        self.sent.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        if self.fail:
            raise RuntimeError("closed")
        self.sent_bytes.append(payload)

    async def close(self) -> None:
        if self.fail:
            raise RuntimeError("closed")
        self.closed = True

    async def receive(self) -> dict:
        if self.incoming:
            return self.incoming.pop(0)
        return {"type": "websocket.disconnect"}


class RecordingSession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.exceptions: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def exception(self, message: str) -> None:
        self.exceptions.append(message)


def test_bind_websocket_request_id_uses_header_and_resets_previous_value():
    outer = request_id_var.set("outer")
    try:
        token = bind_websocket_request_id(
            RecordingWebSocket(headers={"x-request-id": "  websocket-rid  "})
        )
        assert request_id_var.get() == "websocket-rid"

        reset_websocket_request_id(token)

        assert request_id_var.get() == "outer"
    finally:
        request_id_var.reset(outer)


def test_bind_websocket_request_id_generates_missing_header_value():
    outer = request_id_var.set("outer")
    try:
        token = bind_websocket_request_id(RecordingWebSocket())
        generated = request_id_var.get()

        assert generated
        assert generated != "outer"

        reset_websocket_request_id(token)

        assert request_id_var.get() == "outer"
    finally:
        request_id_var.reset(outer)


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
async def test_send_ws_unknown_message_type_uses_shared_operation_error_envelope():
    websocket = RecordingWebSocket()

    await send_ws_unknown_message_type(websocket, "not.supported")

    assert websocket.sent == [{
        "type": "error",
        "message": "Unknown message type: not.supported",
    }]


@pytest.mark.asyncio
async def test_ws_operation_result_or_error_returns_successful_operation_value():
    websocket = RecordingWebSocket()

    result = await ws_operation_result_or_error(websocket, lambda: {"ok": True})

    assert result == {"ok": True}
    assert websocket.sent == []


@pytest.mark.asyncio
async def test_ws_operation_result_or_error_reports_operation_error():
    websocket = RecordingWebSocket()

    def build_config():
        raise EmptyInputError()

    result = await ws_operation_result_or_error(websocket, build_config)

    assert result is None
    assert websocket.sent == [{
        "type": "error",
        "message": "No input text provided",
    }]


@pytest.mark.asyncio
async def test_safe_send_ws_error_suppresses_closed_socket_errors():
    await safe_send_ws_error(RecordingWebSocket(fail=True), "boom")


@pytest.mark.asyncio
async def test_safe_close_websocket_suppresses_closed_socket_errors():
    websocket = RecordingWebSocket()

    await safe_close_websocket(websocket)
    await safe_close_websocket(RecordingWebSocket(fail=True))

    assert websocket.closed is True


@pytest.mark.asyncio
async def test_websocket_connection_scope_accepts_closes_and_resets_request_id_on_error():
    outer = request_id_var.set("outer")
    websocket = RecordingWebSocket(headers={"x-request-id": "scope-rid"})
    try:
        with pytest.raises(RuntimeError, match="boom"):
            async with websocket_connection_scope(websocket):
                assert websocket.accepted is True
                assert request_id_var.get() == "scope-rid"
                raise RuntimeError("boom")

        assert websocket.closed is True
        assert request_id_var.get() == "outer"
    finally:
        request_id_var.reset(outer)


@pytest.mark.asyncio
async def test_websocket_route_error_scope_logs_disconnect_and_closed_messages():
    websocket = RecordingWebSocket()
    logger = RecordingLogger()

    async with websocket_route_error_scope(
        websocket,
        logger=logger,
        disconnect_log_message="disconnected",
        error_log_message="errored",
        closed_log_message="closed",
    ):
        raise WebSocketDisconnect()

    assert websocket.sent == []
    assert logger.infos == ["disconnected", "closed"]
    assert logger.exceptions == []


@pytest.mark.asyncio
async def test_websocket_route_error_scope_sends_standard_error_for_unexpected_exception():
    websocket = RecordingWebSocket()
    logger = RecordingLogger()

    async with websocket_route_error_scope(
        websocket,
        logger=logger,
        disconnect_log_message="disconnected",
        error_log_message="errored",
        closed_log_message="closed",
    ):
        raise RuntimeError("boom")

    assert websocket.sent == [{"type": "error", "message": "boom"}]
    assert logger.infos == ["closed"]
    assert logger.exceptions == ["errored"]


@pytest.mark.asyncio
async def test_websocket_session_event_scope_drains_events_before_closing_session():
    session = RecordingSession()
    started = asyncio.Event()
    done = asyncio.Event()
    drained = False

    async def emit_events() -> None:
        nonlocal drained
        started.set()
        await done.wait()
        drained = True

    async with websocket_session_event_scope(session, emit_events):
        await started.wait()
        assert session.closed is False
        done.set()

    assert drained is True
    assert session.closed is True


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


@pytest.mark.asyncio
async def test_receive_ws_json_message_returns_none_on_disconnect():
    websocket = RecordingWebSocket(incoming=[{"type": "websocket.disconnect"}])

    assert await receive_ws_json_message(websocket) is None
    assert websocket.sent == []


@pytest.mark.asyncio
async def test_receive_ws_frame_classifies_text_binary_and_disconnect_frames():
    websocket = RecordingWebSocket(
        incoming=[
            {"text": '{"type":"config"}'},
            {"bytes": b"pcm"},
            {"type": "websocket.disconnect"},
        ]
    )

    text_frame = await receive_ws_frame(websocket)
    bytes_frame = await receive_ws_frame(websocket)
    disconnect_frame = await receive_ws_frame(websocket)

    assert isinstance(text_frame, WsTextFrame)
    assert text_frame.message == {"type": "config"}
    assert isinstance(bytes_frame, WsBytesFrame)
    assert bytes_frame.data == b"pcm"
    assert isinstance(disconnect_frame, WsDisconnectFrame)
    assert websocket.sent == []


@pytest.mark.asyncio
async def test_parse_ws_frame_reports_invalid_text_frame_and_returns_none():
    websocket = RecordingWebSocket()

    assert await parse_ws_frame(websocket, {"text": "{"}) is None

    assert websocket.sent
    assert websocket.sent[0]["type"] == "error"
    assert "invalid JSON" in websocket.sent[0]["message"]


@pytest.mark.asyncio
async def test_receive_ws_json_message_rejects_binary_frame_and_continues():
    websocket = RecordingWebSocket(
        incoming=[
            {"bytes": b"pcm"},
            {"text": '{"type":"config"}'},
        ]
    )

    assert await receive_ws_json_message(websocket) == {"type": "config"}
    assert websocket.sent == [{"type": "error", "message": "only JSON text frames are supported"}]


@pytest.mark.asyncio
async def test_parse_ws_json_text_frame_reports_invalid_json_without_receiving_more():
    websocket = RecordingWebSocket()

    assert await parse_ws_json_text_frame(websocket, {"text": "{"}) is None

    assert websocket.sent
    assert websocket.sent[0]["type"] == "error"
    assert "invalid JSON" in websocket.sent[0]["message"]


@pytest.mark.asyncio
async def test_receive_required_ws_config_rejects_bad_frames_until_config_arrives():
    websocket = RecordingWebSocket(
        incoming=[
            {"bytes": b"not-json"},
            {"text": "{"},
            {"text": '{"type":"text","text":"too soon"}'},
            {"text": '{"type":"config","model":"tts"}'},
        ]
    )

    config = await receive_required_ws_config(websocket, "text input")

    assert config == {"type": "config", "model": "tts"}
    assert websocket.sent[0] == {
        "type": "error",
        "message": "only JSON text frames are supported",
    }
    assert websocket.sent[1]["type"] == "error"
    assert "invalid JSON" in websocket.sent[1]["message"]
    assert websocket.sent[2] == {
        "type": "error",
        "message": "Configuration message required before text input",
    }


@pytest.mark.asyncio
async def test_receive_required_ws_config_returns_none_when_client_disconnects_first():
    websocket = RecordingWebSocket(incoming=[{"type": "websocket.disconnect"}])

    assert await receive_required_ws_config(websocket, "audio") is None
    assert websocket.sent == []


class _ReadyEvent:
    pass


class _AudioEvent:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _DoneEvent:
    pass


async def _events(*events):
    for event in events:
        yield event


def _payload(event) -> dict | None:
    if isinstance(event, _ReadyEvent):
        return {"type": "ready"}
    if isinstance(event, _DoneEvent):
        return {"type": "done"}
    return None


def _binary_payload(event) -> bytes | None:
    if isinstance(event, _AudioEvent):
        return event.data
    return None


@pytest.mark.asyncio
async def test_emit_ws_session_events_sends_payloads_and_stops_on_terminal_event():
    websocket = RecordingWebSocket()

    await emit_ws_session_events(
        websocket,
        _events(_ReadyEvent(), _DoneEvent(), _ReadyEvent()),
        json_payload=_payload,
        terminal_types=(_DoneEvent,),
    )

    assert websocket.sent == [{"type": "ready"}, {"type": "done"}]


@pytest.mark.asyncio
async def test_emit_ws_session_events_sends_binary_payloads_before_json_mapping():
    websocket = RecordingWebSocket()

    await emit_ws_session_events(
        websocket,
        _events(_AudioEvent(b"audio"), _DoneEvent()),
        json_payload=_payload,
        binary_payload=_binary_payload,
        terminal_types=(_DoneEvent,),
    )

    assert websocket.sent_bytes == [b"audio"]
    assert websocket.sent == [{"type": "done"}]


@pytest.mark.asyncio
async def test_emit_ws_session_events_can_suppress_send_failures():
    websocket = RecordingWebSocket(fail=True)

    await emit_ws_session_events(
        websocket,
        _events(_ReadyEvent(), _DoneEvent()),
        json_payload=_payload,
        terminal_types=(_DoneEvent,),
        suppress_send_errors=True,
    )
