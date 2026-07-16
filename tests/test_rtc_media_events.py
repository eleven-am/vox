from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vox.server.rtc_media_events import emit_media_event


@pytest.mark.asyncio
async def test_emit_media_event_is_noop_without_queue():
    record = SimpleNamespace(media_events=None)

    await emit_media_event(record, {"type": "rtc.connection_state"})


@pytest.mark.asyncio
async def test_emit_media_event_queues_when_available():
    record = SimpleNamespace(media_events=asyncio.Queue())

    event = {"type": "rtc.connection_state", "state": "connected"}
    await emit_media_event(record, event)

    assert await record.media_events.get() == event
