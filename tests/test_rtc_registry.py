from __future__ import annotations

import asyncio

import pytest

from vox.server.rtc_client_events import (
    WIRE_BROWSER_EVENT,
    WIRE_RTC_CLIENT_DISCONNECTED,
    emit_client_disconnected_to_control,
    handle_browser_data_channel_message,
    parse_client_event_message,
)
from vox.server.rtc_registry import RtcSessionRegistry


def test_create_session_returns_lookupable_record_and_token():
    registry = RtcSessionRegistry(join_token_ttl_s=120)
    record, token = registry.create_session(now=1000.0)

    assert record.session_id.startswith("rtc_")
    assert token.startswith("rtc_client_")
    assert record.expires_at == 1120.0
    assert registry.get(record.session_id, now=1001.0) is record


def test_client_token_is_single_use_and_consumed_on_browser_attach():
    registry = RtcSessionRegistry()
    record, token = registry.create_session(now=1000.0)

    attached = registry.consume_client_token(token, now=1001.0)

    assert attached is record
    assert attached.browser_attached is True
    assert attached.client_token_hash == ""
    assert registry.consume_client_token(token, now=1002.0) is None


def test_browser_attach_creates_media_token_and_queues():
    registry = RtcSessionRegistry()
    record, token = registry.create_session(now=1000.0)

    attached = registry.attach_browser(token, now=1001.0)

    assert attached is not None
    attached_record, media_token = attached
    assert attached_record is record
    assert media_token.startswith("rtc_media_")
    assert record.media_token_hash
    assert record.media_events is not None
    assert record.audio_output is not None
    assert registry.validate_media_token(record.session_id, media_token) is record


def test_unused_session_expires_but_attached_browser_survives_token_ttl():
    registry = RtcSessionRegistry(join_token_ttl_s=10)
    unused, _ = registry.create_session(now=1000.0)
    active, token = registry.create_session(now=1000.0)

    assert registry.consume_client_token(token, now=1001.0) is active

    assert registry.get(unused.session_id, now=1011.0) is None
    assert registry.get(active.session_id, now=1011.0) is active


def test_control_attach_is_exclusive():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session(now=1000.0)

    assert registry.attach_control(record.session_id, now=1001.0) is record
    assert registry.attach_control(record.session_id, now=1001.0) is None

    registry.detach_control(record.session_id)
    assert registry.attach_control(record.session_id, now=1001.0) is record


def test_parse_client_event_message_uses_shared_command_validation():
    assert parse_client_event_message({"event": "render.url", "payload": {"url": "https://example.com"}}) == (
        "render.url",
        {"url": "https://example.com"},
    )

    with pytest.raises(ValueError, match="client.event requires a JSON object"):
        parse_client_event_message("not an object")

    with pytest.raises(ValueError, match="client.event requires a non-empty string 'event'"):
        parse_client_event_message({"payload": {}})


@pytest.mark.asyncio
async def test_client_disconnect_control_event_is_deduped():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session(now=1000.0)

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
    record, _ = registry.create_session(now=1000.0)

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
async def test_data_channel_message_drops_malformed_browser_events():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session(now=1000.0)

    await handle_browser_data_channel_message(record, record.session_id, b"\xff")
    await handle_browser_data_channel_message(record, record.session_id, "not-json")
    await handle_browser_data_channel_message(record, record.session_id, '{"payload":{}}')

    assert record.control_events is not None
    assert record.control_events.empty()
