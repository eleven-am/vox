from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from vox.operations.conversation import (
    browser_event_wire,
    client_disconnected_wire,
    client_event_payload_json,
    parse_client_event_command,
)
from vox.operations.errors import OperationError
from vox.server.rtc_registry import RtcSessionRecord

logger = logging.getLogger(__name__)


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
    await record.control_events.put(
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
    if _data_channel_is_open(record):
        record.data_channel.send(raw)
        return True
    return False


def send_client_event_to_browser(record: RtcSessionRecord, event_name: str, payload: Any) -> None:
    raw = client_event_payload_json(event_name, payload)
    if _send_raw_client_event_to_browser(record, raw):
        return
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
    if record.control_events is None:
        return
    await record.control_events.put(browser_event_wire(session_id, event_name, payload))


async def handle_browser_data_channel_message(record: RtcSessionRecord, session_id: str, message: Any) -> None:
    event = parse_browser_data_channel_message(message, session_id=session_id)
    if event is None:
        return

    await emit_browser_event_to_control(record, session_id, event.name, event.payload)


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
