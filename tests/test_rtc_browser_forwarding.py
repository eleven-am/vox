from __future__ import annotations

import json
from unittest.mock import MagicMock

from vox.conversation.session import (
    WIRE_AUDIO_CLEAR,
    WIRE_AUDIO_DELTA,
    WIRE_STATE_CHANGED,
    WIRE_TRANSCRIPT_DELTA,
)
from vox.server.rtc_conversation import (
    BROWSER_FORWARDED_EVENT_TYPES,
    forward_wire_event_to_browser,
)
from vox.server.rtc_registry import RtcSessionRecord


def _record_with_open_channel() -> tuple[RtcSessionRecord, MagicMock]:
    channel = MagicMock()
    channel.readyState = "open"
    record = RtcSessionRecord(
        session_id="rtc_test",
        created_at=0.0,
        expires_at=10.0,
        expected_control_transport="pondsocket",
        data_channel=channel,
    )
    return record, channel


def test_forwards_curated_event_to_open_data_channel():
    record, channel = _record_with_open_channel()

    forward_wire_event_to_browser(
        record,
        {"type": WIRE_STATE_CHANGED, "state": "thinking", "previous_state": "listening"},
    )

    channel.send.assert_called_once()
    sent = json.loads(channel.send.call_args[0][0])
    assert sent["event"] == WIRE_STATE_CHANGED
    assert sent["payload"]["state"] == "thinking"


def test_forwards_transcript_delta_and_audio_clear():
    record, channel = _record_with_open_channel()

    forward_wire_event_to_browser(record, {"type": WIRE_TRANSCRIPT_DELTA, "delta": "hi", "start_ms": 0, "end_ms": 700})
    forward_wire_event_to_browser(record, {"type": WIRE_AUDIO_CLEAR, "response_id": "resp_1"})

    assert channel.send.call_count == 2
    events = [json.loads(call.args[0])["event"] for call in channel.send.call_args_list]
    assert events == [WIRE_TRANSCRIPT_DELTA, WIRE_AUDIO_CLEAR]


def test_audio_delta_never_forwarded():
    record, channel = _record_with_open_channel()

    assert WIRE_AUDIO_DELTA not in BROWSER_FORWARDED_EVENT_TYPES
    forward_wire_event_to_browser(record, {"type": WIRE_AUDIO_DELTA, "audio": "AAAA", "sample_rate": 24000})

    channel.send.assert_not_called()


def test_not_forwarded_when_channel_closed_and_not_buffered():
    record, channel = _record_with_open_channel()
    channel.readyState = "closed"

    forward_wire_event_to_browser(record, {"type": WIRE_STATE_CHANGED, "state": "idle", "previous_state": "speaking"})

    channel.send.assert_not_called()
    assert record.pending_client_events == []


def test_not_forwarded_when_opted_out():
    record, channel = _record_with_open_channel()
    record.forward_browser_events = False

    forward_wire_event_to_browser(record, {"type": WIRE_STATE_CHANGED, "state": "idle", "previous_state": "speaking"})

    channel.send.assert_not_called()


def test_none_wire_and_none_record_are_ignored():
    record, _ = _record_with_open_channel()
    forward_wire_event_to_browser(record, None)
    forward_wire_event_to_browser(None, {"type": WIRE_STATE_CHANGED})
