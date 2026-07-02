from __future__ import annotations

import base64
from typing import Any

from vox.conversation.session import (
    WIRE_AUDIO_CLEAR,
    WIRE_INTERRUPTION_DETECTED,
    WIRE_INTERRUPTION_FALSE_POSITIVE,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_CREATED,
    WIRE_RESPONSE_DONE,
    WIRE_SPEECH_STARTED,
    WIRE_SPEECH_STOPPED,
    WIRE_STATE_CHANGED,
    WIRE_TRANSCRIPT_DELTA,
    WIRE_TRANSCRIPT_DONE,
)
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConversationOrchestrator,
)
from vox.server.rtc_client_events import send_client_event_to_browser

BROWSER_FORWARDED_EVENT_TYPES = frozenset(
    {
        WIRE_STATE_CHANGED,
        WIRE_SPEECH_STARTED,
        WIRE_SPEECH_STOPPED,
        WIRE_TRANSCRIPT_DELTA,
        WIRE_TRANSCRIPT_DONE,
        WIRE_INTERRUPTION_DETECTED,
        WIRE_INTERRUPTION_FALSE_POSITIVE,
        WIRE_RESPONSE_CREATED,
        WIRE_RESPONSE_DONE,
        WIRE_RESPONSE_CANCELLED,
        WIRE_AUDIO_CLEAR,
    }
)


def forward_wire_event_to_browser(record: Any, wire: dict | None) -> None:
    if record is None or wire is None:
        return
    if not getattr(record, "forward_browser_events", True):
        return
    event_type = wire.get("type")
    if event_type not in BROWSER_FORWARDED_EVENT_TYPES:
        return
    channel = getattr(record, "data_channel", None)
    if channel is None or getattr(channel, "readyState", None) != "open":
        return
    payload = {key: value for key, value in wire.items() if key != "type"}
    send_client_event_to_browser(record, str(event_type), payload)


def create_rtc_orchestrator(*, scheduler: Any, record: Any) -> ConversationOrchestrator:
    return create_rtc_orchestrator_with(
        scheduler=scheduler,
        record=record,
        orchestrator_cls=ConversationOrchestrator,
    )


def create_rtc_orchestrator_with(
    *,
    scheduler: Any,
    record: Any,
    orchestrator_cls: type[ConversationOrchestrator],
) -> ConversationOrchestrator:
    async def send_rtc_audio(event: ConvAudioDeltaEvent) -> None:
        await enqueue_rtc_audio(record, event)

    async def wait_for_rtc_playout() -> None:
        await wait_until_rtc_audio_drained(record)

    orchestrator = orchestrator_cls(
        scheduler=scheduler,
        pace_response_done_to_audio=True,
        audio_sink=send_rtc_audio,
        wait_for_output_playout=wait_for_rtc_playout,
    )
    record.orchestrator = orchestrator
    return orchestrator


async def enqueue_rtc_audio(record: Any, event: ConvAudioDeltaEvent) -> None:
    if record is None or record.audio_output_track is None:
        return
    await record.audio_output_track.enqueue(base64.b64decode(event.audio_b64), event.sample_rate)


async def wait_until_rtc_audio_drained(record: Any) -> None:
    if record is None or record.audio_output_track is None:
        return
    await record.audio_output_track.wait_until_drained()


def clear_rtc_audio_if_needed(record: Any, event: object) -> bool:
    if (
        record is None
        or not isinstance(event, ConvAudioClearEvent)
        or record.audio_output_track is None
    ):
        return False
    record.audio_output_track.clear()
    return True
