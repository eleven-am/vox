from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from vox.conversation.session import (
    WIRE_AUDIO_CLEAR,
    WIRE_AUDIO_RESUME,
    WIRE_AUDIO_SUSPEND,
    WIRE_INTERRUPTION_DETECTED,
    WIRE_INTERRUPTION_FALSE_POSITIVE,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_CREATED,
    WIRE_RESPONSE_DONE,
    WIRE_RESPONSE_SPOKEN_TEXT,
    WIRE_SPEECH_STARTED,
    WIRE_SPEECH_STOPPED,
    WIRE_STATE_CHANGED,
    WIRE_TRANSCRIPT_DELTA,
    WIRE_TRANSCRIPT_DONE,
)
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvAudioResumeEvent,
    ConvAudioSuspendEvent,
    ConvDoneEvent,
    ConversationOrchestrator,
    ConvEvent,
    browser_event_wire,
    client_disconnected_wire,
    client_event_payload_json,
    conversation_wire_event_payload,
    parse_client_event_command,
    serialize_conversation_event,
)
from vox.operations.errors import InvalidConfigError, OperationError
from vox.server.rtc_registry import RtcSessionRecord
from vox.speech_context.service import SpeechContextService

logger = logging.getLogger(__name__)
MAX_PENDING_CLIENT_EVENTS = 128
MAX_PENDING_CLIENT_EVENT_BYTES = 262_144

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
        WIRE_RESPONSE_SPOKEN_TEXT,
        WIRE_AUDIO_CLEAR,
        WIRE_AUDIO_SUSPEND,
        WIRE_AUDIO_RESUME,
    }
)


@dataclass(frozen=True)
class BrowserDataChannelEvent:
    name: str
    payload: Any


async def emit_client_disconnected_to_control(
    record: RtcSessionRecord,
    session_id: str,
    *,
    reason: str,
    connection_state: str | None = None,
    ice_connection_state: str | None = None,
    data_channel_state: str | None = None,
) -> None:
    if record.browser_disconnect_emitted:
        return
    record.browser_disconnect_emitted = True
    if record.control_events is None:
        return
    record.control_events.put_terminal_nowait(
        client_disconnected_wire(
            session_id,
            reason=reason,
            connection_state=connection_state,
            ice_connection_state=ice_connection_state,
            data_channel_state=data_channel_state,
        )
    )


def _data_channel_is_open(record: RtcSessionRecord) -> bool:
    channel = record.data_channel
    return channel is not None and getattr(channel, "readyState", None) == "open"


def _send_raw_client_event_to_browser(record: RtcSessionRecord, raw: str) -> bool:
    channel = record.data_channel
    if channel is not None and getattr(channel, "readyState", None) == "open":
        channel.send(raw)
        return True
    return False


def send_client_event_to_browser(record: RtcSessionRecord, event_name: str, payload: Any) -> None:
    raw = client_event_payload_json(event_name, payload)
    if _send_raw_client_event_to_browser(record, raw):
        return
    pending = record.pending_client_events
    pending_bytes = sum(len(item.encode("utf-8")) for item in pending)
    if (
        len(pending) >= MAX_PENDING_CLIENT_EVENTS
        or pending_bytes + len(raw.encode("utf-8")) > MAX_PENDING_CLIENT_EVENT_BYTES
    ):
        raise InvalidConfigError("pending RTC client event queue capacity exceeded")
    record.pending_client_events.append(raw)


def flush_pending_client_events(record: RtcSessionRecord) -> None:
    if not _data_channel_is_open(record):
        return
    pending = list(record.pending_client_events)
    record.pending_client_events.clear()
    for raw in pending:
        with suppress(Exception):
            _send_raw_client_event_to_browser(record, raw)


async def emit_browser_event_to_control(
    record: RtcSessionRecord,
    session_id: str,
    event_name: str,
    payload: Any,
) -> None:
    emit_browser_event_to_control_nowait(record, session_id, event_name, payload)


def emit_browser_event_to_control_nowait(
    record: RtcSessionRecord,
    session_id: str,
    event_name: str,
    payload: Any,
) -> None:
    if record.control_events is None:
        return
    try:
        record.control_events.put_nowait(browser_event_wire(session_id, event_name, payload))
    except asyncio.QueueFull as exc:
        raise InvalidConfigError("RTC browser event queue capacity exceeded") from exc


async def handle_browser_data_channel_message(record: RtcSessionRecord, session_id: str, message: Any) -> None:
    handle_browser_data_channel_message_nowait(record, session_id, message)


def handle_browser_data_channel_message_nowait(
    record: RtcSessionRecord,
    session_id: str,
    message: Any,
) -> None:
    event = parse_browser_data_channel_message(message, session_id=session_id)
    if event is None:
        return

    emit_browser_event_to_control_nowait(record, session_id, event.name, event.payload)


def parse_browser_data_channel_message(
    message: Any,
    *,
    session_id: str,
) -> BrowserDataChannelEvent | None:
    if isinstance(message, bytes):
        try:
            text = message.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("dropping non-UTF-8 RTC data channel message for %s", session_id)
            return
    else:
        text = str(message)

    try:
        message_obj = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("dropping non-JSON RTC data channel message for %s", session_id)
        return

    try:
        event_name, payload = parse_client_event_command(message_obj)
    except OperationError:
        logger.warning("dropping malformed RTC browser.event payload for %s", session_id)
        return None

    return BrowserDataChannelEvent(name=event_name, payload=payload)


@dataclass(frozen=True, slots=True)
class RtcControlEvent:
    wire: dict | None
    done: bool


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
    event_name, payload = conversation_wire_event_payload(wire)
    send_client_event_to_browser(record, event_name, payload)


def prepare_rtc_control_event(
    *,
    record: Any,
    session_id: str,
    event: ConvEvent,
) -> RtcControlEvent:
    apply_rtc_audio_control_if_needed(record, event)
    if (
        isinstance(event, ConvAudioDeltaEvent)
        and record is not None
        and getattr(record, "audio_output_track", None) is not None
    ):
        return RtcControlEvent(wire=None, done=False)
    wire = serialize_conversation_event(event)
    if wire is not None:
        wire.setdefault("session_id", session_id)
        forward_wire_event_to_browser(record, wire)
    return RtcControlEvent(wire=wire, done=isinstance(event, ConvDoneEvent))


def create_rtc_orchestrator(
    *,
    scheduler: Any,
    store: Any | None = None,
    record: Any,
    speech_context_service: SpeechContextService | None = None,
) -> ConversationOrchestrator:
    async def send_rtc_audio(event: ConvAudioDeltaEvent) -> None:
        await enqueue_rtc_audio(record, event)

    async def wait_for_rtc_playout() -> None:
        await wait_until_rtc_audio_drained(record)

    orchestrator = ConversationOrchestrator(
        scheduler=scheduler,
        store=store,
        pace_response_done_to_audio=True,
        audio_sink=send_rtc_audio,
        wait_for_output_playout=wait_for_rtc_playout,
        output_playout_observed=True,
        speech_context_service=speech_context_service,
    )
    record.orchestrator = orchestrator
    return orchestrator


def observe_rtc_audio_playout(record: Any, pcm16: bytes, sample_rate: int) -> None:
    orchestrator = getattr(record, "orchestrator", None)
    if orchestrator is not None:
        orchestrator.observe_output_playout(pcm16, sample_rate)


async def enqueue_rtc_audio(record: Any, event: ConvAudioDeltaEvent) -> None:
    if record is None or record.audio_output_track is None:
        return
    await record.audio_output_track.enqueue(event.audio, event.sample_rate)


async def wait_until_rtc_audio_drained(record: Any) -> None:
    if record is None or record.audio_output_track is None:
        return
    await record.audio_output_track.wait_until_drained()


def clear_rtc_audio_if_needed(record: Any, event: object) -> bool:
    if record is None or not isinstance(event, ConvAudioClearEvent) or record.audio_output_track is None:
        return False
    record.audio_output_track.clear()
    return True


def apply_rtc_audio_control_if_needed(record: Any, event: object) -> bool:
    if record is None or record.audio_output_track is None:
        return False
    if isinstance(event, ConvAudioClearEvent):
        record.audio_output_track.clear()
        return True
    if isinstance(event, ConvAudioSuspendEvent):
        return bool(record.audio_output_track.suspend(event.candidate_id))
    if isinstance(event, ConvAudioResumeEvent):
        return bool(record.audio_output_track.resume(event.candidate_id))
    return False
