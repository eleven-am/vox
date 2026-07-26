from __future__ import annotations

import asyncio
import json

import pytest

from vox.operations.conversation import WIRE_BROWSER_EVENT, WIRE_RTC_CLIENT_DISCONNECTED
from vox.operations.errors import InvalidConfigError
from vox.server.rtc_registry import (
    RTC_MEDIA_EVENT_MAX_BYTES,
    RTC_MEDIA_EVENT_MAX_COUNT,
    RtcEventQueue,
    RtcSessionRegistry,
    cancel_and_drain_media_tasks,
    cancel_media_tasks,
    track_media_task,
    track_task,
)
from vox.server.rtc_session_io import (
    BrowserDataChannelEvent,
    emit_client_disconnected_to_control,
    flush_pending_client_events,
    handle_browser_data_channel_message,
    parse_browser_data_channel_message,
    send_client_event_to_browser,
)


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
async def test_teardown_failure_remains_visible_and_owned():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    class FailingOrchestrator:
        async def close(self) -> None:
            raise RuntimeError("orchestrator still live")

    orchestrator = FailingOrchestrator()
    record.orchestrator = orchestrator

    registry.close(record.session_id)

    with pytest.raises(RuntimeError, match="orchestrator still live"):
        await registry.drain_teardowns()

    assert record.orchestrator is orchestrator


@pytest.mark.asyncio
async def test_transient_teardown_failure_retries_owned_resource():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    class Peer:
        def __init__(self) -> None:
            self.close_count = 0

        async def close(self) -> None:
            self.close_count += 1
            if self.close_count == 1:
                raise RuntimeError("peer close failed once")

    peer = Peer()
    record.rtc_peer = peer
    registry.close(record.session_id)

    with pytest.raises(RuntimeError, match="peer close failed once"):
        await registry.drain_teardowns()

    await registry.drain_teardowns()

    assert peer.close_count == 2
    assert record.rtc_peer is None
    assert record.session_id not in registry._closing_sessions
    assert registry._teardown_tasks == set()
    assert registry._teardown_records == {}


@pytest.mark.asyncio
async def test_transient_teardown_failure_retries_without_manual_drain():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)
    retry_completed = asyncio.Event()

    class Peer:
        def __init__(self) -> None:
            self.close_count = 0

        async def close(self) -> None:
            self.close_count += 1
            if self.close_count == 1:
                raise RuntimeError("peer close failed once")
            retry_completed.set()

    peer = Peer()
    record.rtc_peer = peer
    registry.close(record.session_id)

    await asyncio.wait_for(retry_completed.wait(), timeout=1)
    await asyncio.sleep(0)

    assert peer.close_count == 2
    assert record.rtc_peer is None
    assert record.session_id not in registry._closing_sessions
    assert registry._teardown_tasks == set()
    assert registry._teardown_records == {}


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


@pytest.mark.asyncio
async def test_browser_event_queue_has_count_and_byte_limits():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    for index in range(128):
        await handle_browser_data_channel_message(
            record,
            record.session_id,
            json.dumps(
                {
                    "event": "ui.select",
                    "payload": {"id": str(index)},
                }
            ),
        )

    with pytest.raises(InvalidConfigError, match="RTC browser event queue"):
        await handle_browser_data_channel_message(
            record,
            record.session_id,
            '{"event":"ui.select","payload":{"id":"overflow"}}',
        )

    while not record.control_events.empty():
        await record.control_events.get()

    with pytest.raises(InvalidConfigError, match="RTC browser event queue"):
        await handle_browser_data_channel_message(
            record,
            record.session_id,
            json.dumps(
                {
                    "event": "ui.select",
                    "payload": {"id": "x" * 262_144},
                }
            ),
        )


@pytest.mark.asyncio
async def test_media_event_queue_has_count_and_byte_limits():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    for index in range(256):
        await record.media_events.put(
            {
                "type": "rtc.connection_state",
                "state": f"state-{index}",
            }
        )

    with pytest.raises(asyncio.QueueFull):
        await record.media_events.put(
            {
                "type": "rtc.connection_state",
                "state": "overflow",
            }
        )

    while not record.media_events.empty():
        await record.media_events.get()

    with pytest.raises(asyncio.QueueFull):
        await record.media_events.put(
            {
                "type": "rtc.signaling_error",
                "message": "x" * 524_288,
            }
        )


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


def test_client_event_queue_has_count_and_byte_limits():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket", now=1000.0)

    for index in range(128):
        send_client_event_to_browser(record, "ui.toast", {"message": str(index)})

    with pytest.raises(InvalidConfigError, match="pending RTC client event queue"):
        send_client_event_to_browser(record, "ui.toast", {"message": "overflow"})

    record.pending_client_events.clear()
    with pytest.raises(InvalidConfigError, match="pending RTC client event queue"):
        send_client_event_to_browser(record, "ui.toast", {"message": "x" * 262_144})


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


async def _wait_forever() -> None:
    await asyncio.Event().wait()


async def _return_value(value: str) -> str:
    return value


@pytest.mark.asyncio
async def test_track_task_registers_and_discards_completed_task():
    tasks = set()

    task = track_task(tasks, _return_value("done"))

    assert task in tasks
    assert await task == "done"
    await asyncio.sleep(0)
    assert task not in tasks


@pytest.mark.asyncio
async def test_track_media_task_registers_and_discards_completed_task():
    class Record:
        media_tasks = set()

    task = track_media_task(Record(), _return_value("done"))

    assert task in Record.media_tasks
    assert await task == "done"
    await asyncio.sleep(0)
    assert task not in Record.media_tasks


@pytest.mark.asyncio
async def test_track_media_task_allows_records_without_media_task_set():
    class Record:
        pass

    task = track_media_task(Record(), _return_value("done"))

    assert await task == "done"


@pytest.mark.asyncio
async def test_cancel_media_tasks_cancels_and_clears_tracked_tasks():
    task = asyncio.create_task(_wait_forever())

    class Record:
        media_tasks = {task}

    cancelled = cancel_media_tasks(Record())
    await asyncio.gather(*cancelled, return_exceptions=True)

    assert cancelled == [task]
    assert task.cancelled()
    assert Record.media_tasks == set()


@pytest.mark.asyncio
async def test_cancel_media_tasks_is_noop_without_tracked_tasks():
    class Record:
        pass

    assert cancel_media_tasks(Record()) == []


@pytest.mark.asyncio
async def test_cancel_and_drain_media_tasks_awaits_cancelled_tasks():
    task = asyncio.create_task(_wait_forever())

    class Record:
        media_tasks = {task}

    await cancel_and_drain_media_tasks(Record())

    assert task.cancelled()
    assert Record.media_tasks == set()


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
async def test_close_attached_owns_shared_media_teardown():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    assert registry.attach_control(record.session_id, transport="pondsocket") is record
    record.orchestrator = object()
    record.data_channel = object()
    record.audio_output = asyncio.Queue()
    record.media_events = RtcEventQueue(
        max_count=RTC_MEDIA_EVENT_MAX_COUNT,
        max_bytes=RTC_MEDIA_EVENT_MAX_BYTES,
    )
    peer = FakePeer()
    record.rtc_peer = peer
    orchestrator = FakeOrchestrator()

    await registry.close_attached(record, orchestrator=orchestrator)

    assert orchestrator.ended is None
    assert orchestrator.closed is True
    assert record.orchestrator is None
    assert record.data_channel is None
    assert await record.audio_output.get() is None
    assert await record.media_events.get() is None
    assert peer.close_count == 1
    assert record.rtc_peer is None
    assert registry.get(record.session_id) is None
