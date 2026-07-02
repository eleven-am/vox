from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from vox.core.tasks import drain_task
from vox.server.rtc_media import cancel_and_drain_media_tasks
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry


async def close_rtc_runtime_resources(
    *,
    session_id: str,
    registry: RtcSessionRegistry,
    record: RtcSessionRecord,
    orchestrator: Any,
    emit_task: asyncio.Task,
    client_event_task: asyncio.Task,
) -> None:
    await orchestrator.end_of_stream(flush_response=False)
    await drain_task(emit_task)
    if record.control_events is not None:
        await record.control_events.put(None)
    await drain_task(client_event_task)
    await orchestrator.close()
    record.orchestrator = None
    record.data_channel = None
    if record.audio_output is not None:
        await record.audio_output.put(None)
    if record.media_events is not None:
        await record.media_events.put(None)
    await cancel_and_drain_media_tasks(record)
    if record.rtc_peer is not None:
        with suppress(Exception):
            await record.rtc_peer.close()
    registry.detach_control(session_id)
    registry.close(session_id)
