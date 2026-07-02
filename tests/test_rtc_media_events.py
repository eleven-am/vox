from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vox.server.rtc_media_events import emit_media_event, iter_media_sse, media_sse_message


def test_media_sse_message_uses_event_type_and_json_data():
    assert media_sse_message({"type": "rtc.connection_state", "state": "connected"}) == (
        'event: rtc.connection_state\ndata: {"type": "rtc.connection_state", "state": "connected"}\n\n'
    )


def test_media_sse_message_defaults_event_name_for_untyped_payload():
    assert media_sse_message({"state": "connected"}).startswith("event: message\n")


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


@pytest.mark.asyncio
async def test_iter_media_sse_yields_events_until_close_sentinel():
    record = SimpleNamespace(media_events=asyncio.Queue())
    await record.media_events.put({"type": "rtc.ice_candidate", "candidate": None})
    await record.media_events.put(None)

    stream = iter_media_sse(record)

    first = await anext(stream)
    second = await anext(stream)
    assert first == 'event: rtc.ice_candidate\ndata: {"type": "rtc.ice_candidate", "candidate": null}\n\n'
    assert second == "event: close\ndata: {}\n\n"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
