from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvStateChangedEvent,
)
from vox.server.rtc_conversation import (
    clear_rtc_audio_if_needed,
    create_rtc_orchestrator,
    enqueue_rtc_audio,
    prepare_rtc_control_event,
    wait_until_rtc_audio_drained,
)
from vox.server.rtc_registry import RtcSessionRecord


class FakeAudioTrack:
    def __init__(self) -> None:
        self.enqueued: list[tuple[bytes, int]] = []
        self.clear_count = 0
        self.drain_count = 0

    async def enqueue(self, pcm: bytes, sample_rate: int) -> None:
        self.enqueued.append((pcm, sample_rate))

    async def wait_until_drained(self) -> None:
        self.drain_count += 1

    def clear(self) -> None:
        self.clear_count += 1


@pytest.mark.asyncio
async def test_rtc_audio_helpers_are_noops_without_media_track():
    record = SimpleNamespace(audio_output_track=None, orchestrator=None)
    event = ConvAudioDeltaEvent(
        audio_b64=base64.b64encode(b"pcm").decode(),
        sample_rate=16_000,
        audio_format="pcm16",
    )

    await enqueue_rtc_audio(record, event)
    await wait_until_rtc_audio_drained(record)

    assert clear_rtc_audio_if_needed(record, ConvAudioClearEvent(response_id="r1")) is False


@pytest.mark.asyncio
async def test_rtc_audio_helpers_decode_enqueue_drain_and_clear_track():
    track = FakeAudioTrack()
    record = SimpleNamespace(audio_output_track=track, orchestrator=None)
    event = ConvAudioDeltaEvent(
        audio_b64=base64.b64encode(b"abc123").decode(),
        sample_rate=24_000,
        audio_format="pcm16",
    )

    await enqueue_rtc_audio(record, event)
    await wait_until_rtc_audio_drained(record)

    assert clear_rtc_audio_if_needed(record, ConvAudioClearEvent(response_id="r1")) is True
    assert track.enqueued == [(b"abc123", 24_000)]
    assert track.drain_count == 1
    assert track.clear_count == 1


def test_create_rtc_orchestrator_attaches_orchestrator_to_record():
    record = SimpleNamespace(audio_output_track=FakeAudioTrack(), orchestrator=None)

    orchestrator = create_rtc_orchestrator(scheduler=object(), record=record)

    assert record.orchestrator is orchestrator


def test_prepare_rtc_control_event_serializes_and_forwards_browser_event():
    channel = MagicMock()
    channel.readyState = "open"
    record = RtcSessionRecord(
        session_id="rtc_1",
        client_token_hash="",
        created_at=0.0,
        expires_at=10.0,
        data_channel=channel,
    )

    prepared = prepare_rtc_control_event(
        record=record,
        session_id="rtc_1",
        event=ConvStateChangedEvent(state="thinking", previous_state="listening"),
    )

    assert prepared.done is False
    assert prepared.wire == {
        "type": "turn.state_changed",
        "state": "thinking",
        "previous_state": "listening",
        "session_id": "rtc_1",
    }
    channel.send.assert_called_once()
    sent = json.loads(channel.send.call_args[0][0])
    assert sent["event"] == "turn.state_changed"
    assert sent["payload"]["session_id"] == "rtc_1"


def test_prepare_rtc_control_event_applies_audio_clear_side_effect():
    track = FakeAudioTrack()
    record = SimpleNamespace(audio_output_track=track, data_channel=None)

    prepared = prepare_rtc_control_event(
        record=record,
        session_id="rtc_1",
        event=ConvAudioClearEvent(response_id="resp_1"),
    )

    assert prepared.done is False
    assert prepared.wire == {
        "type": "response.audio.clear",
        "response_id": "resp_1",
        "session_id": "rtc_1",
    }
    assert track.clear_count == 1


def test_prepare_rtc_control_event_suppresses_audio_delta_when_media_track_exists():
    track = FakeAudioTrack()
    record = SimpleNamespace(audio_output_track=track, data_channel=None)

    prepared = prepare_rtc_control_event(
        record=record,
        session_id="rtc_1",
        event=ConvAudioDeltaEvent(
            audio_b64=base64.b64encode(b"pcm").decode(),
            sample_rate=24_000,
            audio_format="pcm16",
            response_id="resp_1",
            sequence=2,
        ),
    )

    assert prepared.done is False
    assert prepared.wire is None


def test_prepare_rtc_control_event_preserves_audio_delta_without_media_track():
    record = SimpleNamespace(audio_output_track=None, data_channel=None)

    prepared = prepare_rtc_control_event(
        record=record,
        session_id="rtc_1",
        event=ConvAudioDeltaEvent(
            audio_b64=base64.b64encode(b"pcm").decode(),
            sample_rate=24_000,
            audio_format="pcm16",
            response_id="resp_1",
            sequence=2,
        ),
    )

    assert prepared.done is False
    assert prepared.wire == {
        "type": "response.audio.delta",
        "audio": base64.b64encode(b"pcm").decode(),
        "sample_rate": 24_000,
        "audio_format": "pcm16",
        "response_id": "resp_1",
        "sequence": 2,
        "session_id": "rtc_1",
    }


def test_prepare_rtc_control_event_marks_done_without_wire_event():
    record = SimpleNamespace(audio_output_track=None, data_channel=None)

    prepared = prepare_rtc_control_event(
        record=record,
        session_id="rtc_1",
        event=ConvDoneEvent(),
    )

    assert prepared.done is True
    assert prepared.wire is None
