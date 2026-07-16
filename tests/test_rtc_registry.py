from __future__ import annotations

import asyncio

import pytest

from vox.operations.conversation import WIRE_BROWSER_EVENT, WIRE_RTC_CLIENT_DISCONNECTED
from vox.server.rtc_client_events import (
    BrowserDataChannelEvent,
    emit_client_disconnected_to_control,
    flush_pending_client_events,
    handle_browser_data_channel_message,
    parse_browser_data_channel_message,
    send_client_event_to_browser,
)
from vox.server.rtc_registry import RtcSessionRegistry


def test_create_session_returns_lookupable_record():
    registry = RtcSessionRegistry(attach_ttl_s=120)
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    assert record.session_id.startswith("rtc_")
    assert record.expires_at == 1120.0
    assert record.expected_control_transport == "pondsocket"
    assert registry.get(record.session_id, now=1001.0) is record


def test_control_owned_browser_attach_is_single_use():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    attached = registry.attach_browser_session(record.session_id, now=1001.0)

    assert attached is record
    assert attached.browser_attached is True
    assert record.media_events is not None
    assert record.audio_output is not None
    assert registry.attach_browser_session(record.session_id, now=1002.0) is None


def test_unused_session_expires_but_attached_browser_survives_token_ttl():
    registry = RtcSessionRegistry(attach_ttl_s=10)
    unused = registry.create_session(control_transport="pondsocket", now=1000.0)
    active = registry.create_session(control_transport="pondsocket", now=1000.0)

    assert registry.attach_browser_session(active.session_id, now=1001.0) is active

    assert registry.get(unused.session_id, now=1011.0) is None
    assert registry.get(active.session_id, now=1011.0) is active


def test_control_attach_is_exclusive_and_transport_is_lifetime_stable():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    assert registry.attach_control(record.session_id, transport="pondsocket", now=1001.0) is record
    assert registry.attach_control(record.session_id, transport="grpc", now=1001.0) is None

    registry.detach_control(record.session_id)
    assert record.attached_control_transport is None
    assert registry.attach_control(record.session_id, transport="grpc", now=1001.0) is None
    assert registry.attach_control(record.session_id, transport="pondsocket", now=1001.0) is record


@pytest.mark.asyncio
async def test_close_tracks_peer_teardown_with_shared_task_owner():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    closed = asyncio.Event()

    class Peer:
        async def close(self) -> None:
            closed.set()

    record.rtc_peer = Peer()

    registry.close(record.session_id)

    await asyncio.wait_for(closed.wait(), timeout=0.1)
    await asyncio.sleep(0)
    assert registry._teardown_tasks == set()


@pytest.mark.asyncio
async def test_close_awaits_cancelled_media_task_cleanup():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def media_worker() -> None:
        try:
            await asyncio.Future()
        finally:
            cleanup_started.set()
            await cleanup_release.wait()

    task = asyncio.create_task(media_worker())
    record.media_tasks.add(task)
    await asyncio.sleep(0)

    registry.close(record.session_id)
    await asyncio.wait_for(cleanup_started.wait(), timeout=0.1)
    assert not task.done()

    cleanup_release.set()
    await registry.drain_teardowns()

    assert task.done()
    assert registry._teardown_tasks == set()


@pytest.mark.asyncio
async def test_client_disconnect_control_event_is_deduped():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    await emit_client_disconnected_to_control(
        record,
        record.session_id,
        reason="peer_connection_closed",
        connection_state="closed",
        ice_connection_state="closed",
        data_channel_state="closed",
    )
    await emit_client_disconnected_to_control(
        record,
        record.session_id,
        reason="ice_connection_failed",
        connection_state="failed",
        ice_connection_state="failed",
    )

    assert record.control_events is not None
    event = await asyncio.wait_for(record.control_events.get(), timeout=0.1)

    assert event == {
        "type": WIRE_RTC_CLIENT_DISCONNECTED,
        "session_id": record.session_id,
        "reason": "peer_connection_closed",
        "connection_state": "closed",
        "ice_connection_state": "closed",
        "data_channel_state": "closed",
    }
    assert record.control_events.empty()


@pytest.mark.asyncio
async def test_data_channel_message_emits_browser_event_to_control():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    await handle_browser_data_channel_message(
        record,
        record.session_id,
        '{"event":"ui.select","payload":{"id":"choice-a"}}',
    )

    assert record.control_events is not None
    event = await asyncio.wait_for(record.control_events.get(), timeout=0.1)

    assert event == {
        "type": WIRE_BROWSER_EVENT,
        "session_id": record.session_id,
        "event": "ui.select",
        "payload": {"id": "choice-a"},
    }


def test_parse_browser_data_channel_message_accepts_text_and_bytes():
    assert parse_browser_data_channel_message(
        '{"event":"ui.select","payload":{"id":"choice-a"}}',
        session_id="rtc_1",
    ) == BrowserDataChannelEvent(name="ui.select", payload={"id": "choice-a"})
    assert parse_browser_data_channel_message(
        b'{"event":"ui.select","payload":{"id":"choice-a"}}',
        session_id="rtc_1",
    ) == BrowserDataChannelEvent(name="ui.select", payload={"id": "choice-a"})


def test_parse_browser_data_channel_message_drops_malformed_payloads():
    assert parse_browser_data_channel_message(b"\xff", session_id="rtc_1") is None
    assert parse_browser_data_channel_message("not-json", session_id="rtc_1") is None
    assert parse_browser_data_channel_message('{"payload":{}}', session_id="rtc_1") is None


@pytest.mark.asyncio
async def test_data_channel_message_drops_malformed_browser_events():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    await handle_browser_data_channel_message(record, record.session_id, b"\xff")
    await handle_browser_data_channel_message(record, record.session_id, "not-json")
    await handle_browser_data_channel_message(record, record.session_id, '{"payload":{}}')

    assert record.control_events is not None
    assert record.control_events.empty()


def test_client_events_queue_until_data_channel_opens():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    send_client_event_to_browser(record, "ui.toast", {"message": "hi"})

    assert len(record.pending_client_events) == 1

    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, payload: str) -> None:
            self.sent.append(payload)

    channel = Channel()
    record.data_channel = channel

    flush_pending_client_events(record)

    assert record.pending_client_events == []
    assert channel.sent == ['{"event": "ui.toast", "payload": {"message": "hi"}}']


def test_flush_pending_client_events_suppresses_send_errors():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)
    send_client_event_to_browser(record, "ui.toast", {"message": "hi"})

    class FailingChannel:
        readyState = "open"

        def send(self, payload: str) -> None:
            raise RuntimeError("browser channel closed during flush")

    record.data_channel = FailingChannel()

    flush_pending_client_events(record)

    assert record.pending_client_events == []
