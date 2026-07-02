from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from vox.server.rtc_control import (
    close_rtc_control_runtime,
    receive_rtc_control_commands,
    rtc_session_attached_wire,
)
from vox.server.rtc_registry import RtcSessionRegistry


def test_rtc_session_attached_wire_includes_session_id():
    assert rtc_session_attached_wire("rtc_123") == {
        "type": "rtc.session.attached",
        "session_id": "rtc_123",
    }


class FakeWebSocket:
    def __init__(self, incoming: list[dict] | None = None) -> None:
        self.incoming = list(incoming or [])
        self.sent_json = []
        self.closed = False

    async def receive(self):
        if self.incoming:
            return self.incoming.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def close(self, *_, **__):
        self.closed = True


class FakeOrchestrator:
    def __init__(self) -> None:
        self.config = object()
        self.calls = []
        self.ended = False
        self.closed = False

    async def start_response(self, *, allow_interruptions: bool = True) -> None:
        self.calls.append(("start_response", allow_interruptions))

    async def append_response_text(self, text: str, *, allow_interruptions: bool = True) -> None:
        self.calls.append(("append_response_text", text, allow_interruptions))

    async def commit_response(self) -> None:
        self.calls.append(("commit_response",))

    async def end_of_stream(self, *, flush_response: bool = True) -> None:
        self.ended = flush_response

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_receive_rtc_control_commands_executes_shared_control_commands():
    websocket = FakeWebSocket(
        [
            {"text": json.dumps({"type": "response.start", "allow_interruptions": False})},
            {"text": json.dumps({"type": "response.delta", "delta": "hello"})},
            {"text": json.dumps({"type": "response.commit"})},
            {"type": "websocket.disconnect"},
        ]
    )
    record = SimpleNamespace(data_channel=None, pending_client_events=[])
    orchestrator = FakeOrchestrator()

    await receive_rtc_control_commands(websocket, record=record, orchestrator=orchestrator)

    assert orchestrator.calls == [
        ("start_response", False),
        ("append_response_text", "hello", True),
        ("commit_response",),
    ]
    assert websocket.sent_json == []


@pytest.mark.asyncio
async def test_receive_rtc_control_commands_reports_invalid_json():
    websocket = FakeWebSocket(
        [
            {"text": "{"},
            {"type": "websocket.disconnect"},
        ]
    )
    record = SimpleNamespace(data_channel=None, pending_client_events=[])
    orchestrator = FakeOrchestrator()

    await receive_rtc_control_commands(websocket, record=record, orchestrator=orchestrator)

    assert websocket.sent_json
    assert websocket.sent_json[0]["type"] == "error"
    assert "invalid JSON" in websocket.sent_json[0]["message"]


@pytest.mark.asyncio
async def test_close_rtc_control_runtime_clears_record_and_closes_session():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session()
    assert registry.attach_control(record.session_id) is record
    record.orchestrator = object()
    record.data_channel = object()
    record.audio_output = asyncio.Queue()
    record.media_events = asyncio.Queue()
    orchestrator = FakeOrchestrator()
    websocket = FakeWebSocket()

    async def noop():
        return None

    emit_task = asyncio.create_task(noop())
    client_event_task = asyncio.create_task(noop())

    await close_rtc_control_runtime(
        websocket=websocket,
        session_id=record.session_id,
        registry=registry,
        record=record,
        orchestrator=orchestrator,
        emit_task=emit_task,
        client_event_task=client_event_task,
    )

    assert orchestrator.ended is False
    assert orchestrator.closed is True
    assert record.orchestrator is None
    assert record.data_channel is None
    assert await record.audio_output.get() is None
    assert await record.media_events.get() is None
    assert registry.get(record.session_id) is None
    assert websocket.closed is True
