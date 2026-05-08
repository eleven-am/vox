from __future__ import annotations

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
