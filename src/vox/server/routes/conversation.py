"""WebSocket endpoint for agent-facing voice conversations.

Wire shape is OpenAI Realtime-compatible so existing SDKs can point at Vox with
a URL swap. See src/vox/conversation/session.py for the wire event names.

This route is NOT intended for direct browser use. Browsers / phones terminate
WebRTC (or SIP) at an *agent* process, which then speaks this WS on behalf of
the user. That keeps Vox's scope limited to speech inference + turn orchestration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vox.core.tasks import drain_task
from vox.logging_context import new_request_id, request_id_var
from vox.operations.conversation import (
    ConvDoneEvent,
    ConversationOrchestrator,
    execute_conversation_command,
    serialize_conversation_event,
)
from vox.operations.errors import OperationError
from vox.server.websocket import safe_send_ws_error, send_ws_error

logger = logging.getLogger(__name__)
router = APIRouter()
legacy_router = APIRouter()


@legacy_router.websocket("/v1/conversation")
async def conversation_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    scheduler = websocket.app.state.scheduler

    incoming = websocket.headers.get("x-request-id")
    rid = incoming.strip() if incoming and incoming.strip() else new_request_id()
    token = request_id_var.set(rid)

    logger.info("conversation ws connected")

    orchestrator = ConversationOrchestrator(scheduler=scheduler)

    async def emit_events() -> None:
        async for event in orchestrator.events():
            wire = serialize_conversation_event(event)
            if wire is not None:
                with suppress(Exception):
                    await websocket.send_json(wire)
            if isinstance(event, ConvDoneEvent):
                return

    emit_task = asyncio.create_task(emit_events())

    try:
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
                await execute_conversation_command(orchestrator, msg)
            except OperationError as exc:
                await send_ws_error(websocket, str(exc))

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("conversation WS error")
        await safe_send_ws_error(websocket, "internal error; closing")
    finally:
        await orchestrator.end_of_stream()
        await drain_task(emit_task)
        await orchestrator.close()
        with suppress(Exception):
            await websocket.close()
        logger.info("conversation ws closed")
        request_id_var.reset(token)
