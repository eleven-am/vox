from __future__ import annotations

from typing import Any


async def emit_media_event(record: Any, event: dict) -> None:
    if getattr(record, "media_events", None) is not None:
        await record.media_events.put(event)
