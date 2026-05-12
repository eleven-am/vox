from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from vox.server.rtc_registry import RtcSessionRecord

WIRE_CLIENT_EVENT = "client.event"


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


async def emit_client_event_to_control(
    record: RtcSessionRecord,
    session_id: str,
    event_name: str,
    payload: Any,
) -> None:
    if record.control_events is None:
        return
    await record.control_events.put(client_event_wire(session_id, event_name, payload))
