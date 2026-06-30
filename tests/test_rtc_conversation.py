from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from vox.operations.conversation import ConvAudioClearEvent, ConvAudioDeltaEvent
from vox.server.rtc_conversation import (
    clear_rtc_audio_if_needed,
    create_rtc_orchestrator,
    enqueue_rtc_audio,
    wait_until_rtc_audio_drained,
)


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
