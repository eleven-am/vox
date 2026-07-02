from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, WebSocket

from vox.core.tasks import drain_task
from vox.operations.conversation import (
    ConvDoneEvent,
    ConversationOrchestrator,
    execute_conversation_command,
    execute_rtc_control_command,
    pondsocket_event_to_conversation_command,
    serialize_conversation_event,
)
from vox.operations.errors import OperationError
from vox.server.auth import configured_api_key, extract_api_key_from_connection, is_api_key_authorized
from vox.server.rtc_cleanup import close_rtc_runtime_resources
from vox.server.rtc_client_events import send_client_event_to_browser
from vox.server.rtc_conversation import (
    create_rtc_orchestrator,
    prepare_rtc_control_event,
)
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry

if TYPE_CHECKING:
    from pondsocket import Channel, EventContext, JoinContext, LeaveContext

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass(slots=True)
class _ConversationRuntime:
    channel_name: str
    session_id: str
    user_id: str
    orchestrator: ConversationOrchestrator
    emit_task: asyncio.Task[None]


@dataclass(slots=True)
class _RtcRuntime:
    channel_name: str
    session_id: str
    user_id: str
    record: RtcSessionRecord
    orchestrator: ConversationOrchestrator
    emit_task: asyncio.Task[None]
    client_event_task: asyncio.Task[None]


def _http_to_ws_close_code(http_code: int) -> int:
    if 1000 <= http_code <= 4999:
        return http_code
    if 100 <= http_code <= 999:
        return 4000 + http_code
    return 1008


@router.websocket("/v1/socket")
async def pondsocket_ws(websocket: WebSocket) -> None:
    pond = getattr(websocket.app.state, "pondsocket", None)
    if pond is None:
        await websocket.close(code=4404, reason="PondSocket gateway not enabled")
        return

    from pondsocket import Event, SystemEntity
    from pondsocket_asgi._scope import build_incoming_connection
    from pondsocket_asgi.transport import ASGIWebSocketTransport
    from pondsocket_common import ServerActions, uuid

    match = pond.match_endpoint("/")
    if match is None:
        await websocket.close(code=4404, reason="endpoint not found")
        return

    endpoint = match.endpoint
    route = match.route
    user_id = uuid()
    incoming = build_incoming_connection(user_id=user_id, scope=websocket.scope, route=route)
    ctx = await endpoint.request_connection(incoming, user_id=user_id)
    if ctx.is_declined:
        code, message = ctx.decline_info
        await websocket.close(code=_http_to_ws_close_code(code), reason=message)
        return

    await websocket.accept()
    transport = ASGIWebSocketTransport(
        id=ctx.user_id,
        send=websocket.send,
        receive=websocket.receive,
        assigns=ctx.assigns,
    )
    await endpoint.register_transport(transport)

    if ctx.pending_reply is not None:
        name, payload = ctx.pending_reply
        await transport.send_event(
            Event(
                action=ServerActions.SYSTEM.value,
                channel_name=SystemEntity.GATEWAY.value,
                request_id=uuid(),
                event=name,
                payload=payload,
            )
        )

    await transport.wait_until_closed()


def install_pondsocket_gateway(app: FastAPI, *, mount_path: str = "/v1/socket") -> bool:
    try:
        from pondsocket import ConnectionContext, PondSocket
    except ImportError as exc:
        logger.warning("PondSocket gateway requested but dependency is unavailable: %s", exc)
        return False

    scheduler = app.state.scheduler
    rtc_registry: RtcSessionRegistry = app.state.rtc_registry
    conversation_runtimes: dict[str, _ConversationRuntime] = {}
    rtc_runtimes: dict[str, _RtcRuntime] = {}

    async def auth(ctx: ConnectionContext) -> None:
        expected = configured_api_key()
        if expected is None:
            ctx.accept()
            return
        if is_api_key_authorized(extract_api_key_from_connection(ctx)):
            ctx.accept()
            return
        ctx.decline(401, "missing or invalid API key")

    pond = PondSocket()
    endpoint = pond.create_endpoint("/", auth)

    async def emit_wire_to_user(channel: Channel, user_id: str, wire: dict[str, Any]) -> None:
        payload = dict(wire)
        event_name = str(payload.pop("type"))
        await channel.broadcast_to(event_name, payload, user_id)

    async def reply_error(ctx: EventContext, message: str) -> None:
        if ctx.has_replied():
            return
        with suppress(Exception):
            await ctx.reply("error", {"message": message})

    async def close_conversation_runtime(runtime: _ConversationRuntime) -> None:
        await runtime.orchestrator.end_of_stream()
        await drain_task(runtime.emit_task)
        await runtime.orchestrator.close()

    async def close_rtc_runtime(runtime: _RtcRuntime) -> None:
        await close_rtc_runtime_resources(
            session_id=runtime.session_id,
            registry=rtc_registry,
            record=runtime.record,
            orchestrator=runtime.orchestrator,
            emit_task=runtime.emit_task,
            client_event_task=runtime.client_event_task,
        )

    def conversation_session_id(ctx: JoinContext | EventContext | LeaveContext) -> str:
        return str(ctx.route.params.get("session_id") or ctx.channel.name.rsplit("/", 1)[-1])

    def rtc_session_id(ctx: JoinContext | EventContext | LeaveContext) -> str:
        return str(ctx.route.params.get("session_id") or ctx.channel.name.rsplit("/", 1)[-1])

    async def build_conversation_runtime(channel: Channel, user_id: str, session_id: str) -> _ConversationRuntime:
        orchestrator = ConversationOrchestrator(scheduler=scheduler)

        async def emit_events() -> None:
            async for event in orchestrator.events():
                wire = serialize_conversation_event(event)
                if wire is not None:
                    with suppress(Exception):
                        await emit_wire_to_user(channel, user_id, wire)
                if isinstance(event, ConvDoneEvent):
                    return

        emit_task = asyncio.create_task(emit_events())
        return _ConversationRuntime(
            channel_name=channel.name,
            session_id=session_id,
            user_id=user_id,
            orchestrator=orchestrator,
            emit_task=emit_task,
        )

    async def build_rtc_runtime(
        channel: Channel,
        user_id: str,
        session_id: str,
        record: RtcSessionRecord,
    ) -> _RtcRuntime:
        orchestrator = create_rtc_orchestrator(scheduler=scheduler, record=record)

        async def emit_events() -> None:
            async for event in orchestrator.events():
                prepared = prepare_rtc_control_event(
                    record=record,
                    session_id=session_id,
                    event=event,
                )
                wire = prepared.wire
                if wire is not None:
                    with suppress(Exception):
                        await emit_wire_to_user(channel, user_id, wire)
                if prepared.done:
                    return

        async def emit_client_events() -> None:
            if record.control_events is None:
                return
            while True:
                event = await record.control_events.get()
                if event is None:
                    return
                with suppress(Exception):
                    await emit_wire_to_user(channel, user_id, event)

        emit_task = asyncio.create_task(emit_events())
        client_event_task = asyncio.create_task(emit_client_events())
        return _RtcRuntime(
            channel_name=channel.name,
            session_id=session_id,
            user_id=user_id,
            record=record,
            orchestrator=orchestrator,
            emit_task=emit_task,
            client_event_task=client_event_task,
        )

    async def on_conversation_join(ctx: JoinContext) -> None:
        if await ctx.channel.user_count() > 0:
            await ctx.decline(409, "conversation channel already attached")
            return
        session_id = conversation_session_id(ctx)
        runtime = await build_conversation_runtime(ctx.channel, ctx.transport.get_id(), session_id)
        conversation_runtimes[ctx.channel.name] = runtime
        try:
            await ctx.accept({"session_id": session_id, "kind": "conversation"})
        except Exception:
            conversation_runtimes.pop(ctx.channel.name, None)
            await close_conversation_runtime(runtime)
            raise

    async def on_conversation_leave(ctx: LeaveContext) -> None:
        runtime = conversation_runtimes.pop(ctx.channel.name, None)
        if runtime is not None:
            await close_conversation_runtime(runtime)

    async def on_conversation_event(ctx: EventContext) -> None:
        runtime = conversation_runtimes.get(ctx.channel.name)
        if runtime is None:
            await reply_error(ctx, "conversation session not attached")
            return
        try:
            message = pondsocket_event_to_conversation_command(ctx.event_name, ctx.get_payload())
            await execute_conversation_command(
                runtime.orchestrator,
                message,
                unknown_message_label="unknown conversation message type",
            )
        except OperationError as exc:
            await reply_error(ctx, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("PondSocket conversation event error")
            await reply_error(ctx, str(exc))

    async def on_rtc_join(ctx: JoinContext) -> None:
        if await ctx.channel.user_count() > 0:
            await ctx.decline(409, "RTC control channel already attached")
            return
        session_id = rtc_session_id(ctx)
        record = rtc_registry.attach_control(session_id)
        if record is None:
            await ctx.decline(404, "unknown, expired, or already attached RTC session")
            return
        try:
            runtime = await build_rtc_runtime(ctx.channel, ctx.transport.get_id(), session_id, record)
        except Exception:
            rtc_registry.detach_control(session_id)
            raise
        rtc_runtimes[ctx.channel.name] = runtime
        try:
            await ctx.accept({"session_id": session_id, "kind": "rtc"})
            await ctx.reply("rtc.session.attached", {"session_id": session_id})
        except Exception:
            rtc_runtimes.pop(ctx.channel.name, None)
            await close_rtc_runtime(runtime)
            raise

    async def on_rtc_leave(ctx: LeaveContext) -> None:
        runtime = rtc_runtimes.pop(ctx.channel.name, None)
        if runtime is not None:
            await close_rtc_runtime(runtime)

    async def on_rtc_event(ctx: EventContext) -> None:
        runtime = rtc_runtimes.get(ctx.channel.name)
        if runtime is None:
            await reply_error(ctx, "RTC control session not attached")
            return
        try:
            message = pondsocket_event_to_conversation_command(ctx.event_name, ctx.get_payload())
            await execute_rtc_control_command(
                runtime.orchestrator,
                message,
                client_event_handler=lambda event_name, payload: send_client_event_to_browser(
                    runtime.record,
                    event_name,
                    payload,
                ),
                unknown_message_label="unknown RTC control message type",
            )
        except OperationError as exc:
            await reply_error(ctx, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("PondSocket RTC control event error")
            await reply_error(ctx, str(exc))

    conversation_lobby = endpoint.create_channel("/conversation/:session_id", on_conversation_join)
    conversation_lobby.on_leave(on_conversation_leave)
    conversation_lobby.on_message("*", on_conversation_event)

    rtc_lobby = endpoint.create_channel("/rtc/:session_id", on_rtc_join)
    rtc_lobby.on_leave(on_rtc_leave)
    rtc_lobby.on_message("*", on_rtc_event)

    app.state.pondsocket = pond
    app.state.pondsocket_mount_path = mount_path
    app.include_router(router)
    logger.info("Enabled PondSocket gateway at %s", mount_path)
    return True
