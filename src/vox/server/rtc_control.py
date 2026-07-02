"""Developer control channel for RTC sessions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from vox.core.tasks import drain_task
from vox.operations.conversation import (
    ConvDoneEvent,
    ConversationOrchestrator,
    execute_conversation_command,
    serialize_conversation_event,
)
from vox.operations.errors import OperationError
from vox.server.rtc_client_events import send_client_event_to_browser
from vox.server.rtc_conversation import (
    clear_rtc_audio_if_needed,
    create_rtc_orchestrator,
    forward_wire_event_to_browser,
)
from vox.server.rtc_media import cancel_and_drain_media_tasks
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry
from vox.server.rtc_timeline import RtcTurnTimeline, rtc_audio_stats
from vox.server.websocket import safe_send_ws_error, send_ws_error, send_ws_operation_error

logger = logging.getLogger(__name__)


def rtc_session_attached_wire(session_id: str) -> dict:
    return {
        "type": "rtc.session.attached",
        "session_id": session_id,
    }


async def handle_rtc_control_ws(
    websocket: WebSocket,
    session_id: str,
    *,
    registry: RtcSessionRegistry,
    scheduler: Any,
    orchestrator_factory: Callable[..., ConversationOrchestrator] = create_rtc_orchestrator,
) -> None:
    record = registry.attach_control(session_id)
    if record is None:
        await websocket.close(code=1008, reason="unknown, expired, or already attached RTC session")
        return

    await websocket.accept()

    orchestrator = orchestrator_factory(scheduler=scheduler, record=record)
    timeline = RtcTurnTimeline(session_id=session_id)
    emit_task = asyncio.create_task(
        emit_rtc_orchestrator_events(
            websocket=websocket,
            session_id=session_id,
            record=record,
            orchestrator=orchestrator,
            timeline=timeline,
        )
    )
    client_event_task = asyncio.create_task(
        emit_rtc_control_events(websocket=websocket, record=record)
    )

    try:
        await websocket.send_json(rtc_session_attached_wire(session_id))
        await receive_rtc_control_commands(websocket, record=record, orchestrator=orchestrator)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("RTC control WS error")
        await safe_send_ws_error(websocket, "internal error; closing")
    finally:
        await close_rtc_control_runtime(
            websocket=websocket,
            session_id=session_id,
            registry=registry,
            record=record,
            orchestrator=orchestrator,
            emit_task=emit_task,
            client_event_task=client_event_task,
        )


async def emit_rtc_orchestrator_events(
    *,
    websocket: WebSocket,
    session_id: str,
    record: RtcSessionRecord,
    orchestrator: ConversationOrchestrator,
    timeline: RtcTurnTimeline,
) -> None:
    async for event in orchestrator.events():
        clear_rtc_audio_if_needed(record, event)
        wire = serialize_conversation_event(event)
        if wire is not None:
            wire.setdefault("session_id", session_id)
            forward_wire_event_to_browser(record, wire)
            with suppress(Exception):
                await websocket.send_json(wire)
            timing = timeline.observe(wire, audio_stats=rtc_audio_stats(record))
            if timing is not None:
                with suppress(Exception):
                    await websocket.send_json(timing)
        if isinstance(event, ConvDoneEvent):
            return


async def emit_rtc_control_events(*, websocket: WebSocket, record: RtcSessionRecord) -> None:
    if record.control_events is None:
        return
    while True:
        event = await record.control_events.get()
        if event is None:
            return
        with suppress(Exception):
            await websocket.send_json(event)


async def receive_rtc_control_commands(
    websocket: WebSocket,
    *,
    record: RtcSessionRecord,
    orchestrator: ConversationOrchestrator,
) -> None:
    while True:
        raw = await websocket.receive()
        if raw.get("type") == "websocket.disconnect":
            break
        if "text" not in raw or raw["text"] is None:
            await send_ws_error(websocket, "only JSON text frames are supported")
            continue

        try:
            msg = json.loads(raw["text"])
        except json.JSONDecodeError as exc:
            await send_ws_error(websocket, f"invalid JSON: {exc}")
            continue

        try:
            await execute_conversation_command(
                orchestrator,
                msg,
                allow_input_audio=False,
                client_event_handler=lambda event_name, payload: send_client_event_to_browser(
                    record,
                    event_name,
                    payload,
                ),
                unknown_message_label="unknown control message type",
            )
        except OperationError as exc:
            await send_ws_operation_error(websocket, exc)


async def close_rtc_control_runtime(
    *,
    websocket: WebSocket,
    session_id: str,
    registry: RtcSessionRegistry,
    record: RtcSessionRecord,
    orchestrator: ConversationOrchestrator,
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
    with suppress(Exception):
        await websocket.close()
