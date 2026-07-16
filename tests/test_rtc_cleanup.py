from __future__ import annotations

import asyncio

import pytest

from vox.server.rtc_cleanup import close_attached_rtc_resources
from vox.server.rtc_registry import RtcSessionRegistry


class FakeOrchestrator:
    def __init__(self) -> None:
        self.ended = None
        self.closed = False

    async def end_of_stream(self, *, flush_response: bool = True) -> None:
        self.ended = flush_response

    async def close(self) -> None:
        self.closed = True


class FakePeer:
    def __init__(self) -> None:
        self.closed = False
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1
        self.closed = True


@pytest.mark.asyncio
async def test_close_attached_rtc_resources_owns_shared_media_teardown():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    assert registry.attach_control(record.session_id, transport="pondsocket") is record
    record.orchestrator = object()
    record.data_channel = object()
    record.audio_output = asyncio.Queue()
    record.media_events = asyncio.Queue()
    peer = FakePeer()
    record.rtc_peer = peer
    orchestrator = FakeOrchestrator()

    await close_attached_rtc_resources(
        session_id=record.session_id,
        registry=registry,
        record=record,
        orchestrator=orchestrator,
    )

    assert orchestrator.ended is None
    assert orchestrator.closed is True
    assert record.orchestrator is None
    assert record.data_channel is None
    assert await record.audio_output.get() is None
    assert await record.media_events.get() is None
    assert peer.close_count == 1
    assert record.rtc_peer is None
    assert registry.get(record.session_id) is None
