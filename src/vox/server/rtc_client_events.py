from __future__ import annotations

import json
import logging
from contextlib import suppress
from typing import Any

from vox.operations.conversation import parse_client_event_command
from vox.operations.errors import OperationError
from vox.server.rtc_registry import RtcSessionRecord

WIRE_CLIENT_EVENT = "client.event"
WIRE_BROWSER_EVENT = "browser.event"
WIRE_RTC_CLIENT_DISCONNECTED = "rtc.client.disconnected"

logger = logging.getLogger(__name__)


def parse_client_event_message(message: Any) -> tuple[str, Any]:
    if not isinstance(message, dict):
        raise ValueError("client.event requires a JSON object")
    event_name = message.get("event")
    if not isinstance(event_name, str) or not event_name.strip():
        raise ValueError("client.event requires a non-empty string 'event'")
    return event_name.strip(), message.get("payload")


def client_event_payload(event_name: str, payload: Any) -> dict:
    return {
        "event": event_name,
        "payload": payload,
    }


def client_event_wire(session_id: str, event_name: str, payload: Any) -> dict:
    return {
        "type": WIRE_CLIENT_EVENT,
        "session_id": session_id,
        "event": event_name,
        "payload": payload,
    }


def browser_event_wire(session_id: str, event_name: str, payload: Any) -> dict:
    return {
        "type": WIRE_BROWSER_EVENT,
        "session_id": session_id,
        "event": event_name,
        "payload": payload,
    }


def control_event_as_client_event(event: dict) -> tuple[str, Any]:
    event_type = event.get("type")
    if event_type == WIRE_CLIENT_EVENT:
        return parse_client_event_message(event)
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("control event requires a non-empty string 'type'")
    return event_type.strip(), {key: value for key, value in event.items() if key != "type"}


def client_disconnected_wire(
    session_id: str,
    *,
    reason: str,
    connection_state: str | None = None,
    ice_connection_state: str | None = None,
    data_channel_state: str | None = None,
) -> dict:
    payload = {
        "type": WIRE_RTC_CLIENT_DISCONNECTED,
        "session_id": session_id,
        "reason": reason,
    }
    if connection_state is not None:
        payload["connection_state"] = connection_state
    if ice_connection_state is not None:
        payload["ice_connection_state"] = ice_connection_state
    if data_channel_state is not None:
        payload["data_channel_state"] = data_channel_state
    return payload


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


def send_client_event_to_browser(record: RtcSessionRecord, event_name: str, payload: Any) -> None:
    raw = json.dumps(client_event_payload(event_name, payload))
    if _data_channel_is_open(record):
        record.data_channel.send(raw)
        return
    record.pending_client_events.append(raw)


def flush_pending_client_events(record: RtcSessionRecord) -> None:
    if not _data_channel_is_open(record):
        return
    pending = list(record.pending_client_events)
    record.pending_client_events.clear()
    for raw in pending:
        with suppress(Exception):
            record.data_channel.send(raw)


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
        return

    await emit_browser_event_to_control(record, session_id, event_name, payload)
