from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry
from vox.server.rtc_tasks import cancel_and_drain_media_tasks


async def close_attached_rtc_resources(
    *,
    session_id: str,
    registry: RtcSessionRegistry,
    record: RtcSessionRecord,
    orchestrator: Any | None,
) -> None:
    if orchestrator is not None:
        with suppress(Exception):
            await orchestrator.close()
    record.orchestrator = None
    record.data_channel = None
    if record.audio_output is not None:
        if record.audio_output_track is not None:
            record.audio_output_track.clear()
        while True:
            try:
                record.audio_output.get_nowait()
            except asyncio.QueueEmpty:
                break
        await record.audio_output.put(None)
    if record.media_events is not None:
        await record.media_events.put(None)
    await cancel_and_drain_media_tasks(record)
    if record.rtc_peer is not None:
        peer = record.rtc_peer
        record.rtc_peer = None
        with suppress(Exception):
            await peer.close()
    registry.detach_control(session_id)
    registry.close(session_id)
