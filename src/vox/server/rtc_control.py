"""Developer control channel for RTC sessions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from vox.operations.conversation import (
    ConversationOrchestrator,
    execute_conversation_command,
)
from vox.operations.errors import OperationError
from vox.server.rtc_cleanup import close_rtc_runtime_resources
from vox.server.rtc_client_events import send_client_event_to_browser
from vox.server.rtc_conversation import (
    create_rtc_orchestrator,
    prepare_rtc_control_event,
)
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry
from vox.server.rtc_timeline import RtcTurnTimeline, rtc_audio_stats
from vox.server.websocket import iter_ws_json_messages, safe_send_ws_error, send_ws_operation_error

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
        prepared = prepare_rtc_control_event(record=record, session_id=session_id, event=event)
        wire = prepared.wire
        if wire is not None:
            with suppress(Exception):
                await websocket.send_json(wire)
            timing = timeline.observe(wire, audio_stats=rtc_audio_stats(record))
            if timing is not None:
                with suppress(Exception):
                    await websocket.send_json(timing)
        if prepared.done:
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
    async for msg in iter_ws_json_messages(websocket):
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
    await close_rtc_runtime_resources(
        session_id=session_id,
        registry=registry,
        record=record,
        orchestrator=orchestrator,
        emit_task=emit_task,
        client_event_task=client_event_task,
    )
    with suppress(Exception):
        await websocket.close()
