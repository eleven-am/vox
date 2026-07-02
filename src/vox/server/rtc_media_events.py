from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


async def emit_media_event(record: Any, event: dict) -> None:
    if getattr(record, "media_events", None) is not None:
        await record.media_events.put(event)


def media_sse_message(event: dict) -> str:
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event)}\n\n"


async def iter_media_sse(record: Any) -> AsyncIterator[str]:
    while True:
        event = await record.media_events.get()
        if event is None:
            yield "event: close\ndata: {}\n\n"
            return
        yield media_sse_message(event)
