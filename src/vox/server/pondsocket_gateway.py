from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, WebSocket

from vox.operations.conversation import ConversationOrchestrator
from vox.operations.conversation_runtime import ConversationRuntime
from vox.operations.errors import OperationError
from vox.operations.rtc_runtime import RtcRuntime
from vox.server.app_services import (
    app_pondsocket,
    app_rtc_registry,
    app_scheduler,
    set_app_pondsocket_gateway,
)
from vox.server.auth import authorize_api_key_connection
from vox.server.operation_errors import operation_error_to_http
from vox.server.pondsocket_events import (
    broadcast_conversation_event_to_user,
    decline_if_channel_attached,
    handle_pondsocket_control_event,
    pondsocket_route_or_channel_session_id,
    try_broadcast_wire_to_user,
)
from vox.server.rtc_registry import RtcSessionRegistry

if TYPE_CHECKING:
    from pondsocket import Channel, EventContext, JoinContext, LeaveContext

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass(slots=True)
class _ConversationRuntime:
    channel_name: str
    session_id: str
    user_id: str
    conversation: ConversationRuntime

    @property
    def orchestrator(self) -> ConversationOrchestrator:
        return self.conversation.orchestrator


def _http_to_ws_close_code(http_code: int) -> int:
    if 1000 <= http_code <= 4999:
        return http_code
    if 100 <= http_code <= 999:
        return 4000 + http_code
    return 1008


@router.websocket("/v1/socket")
async def pondsocket_ws(websocket: WebSocket) -> None:
    pond = app_pondsocket(websocket)
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

    scheduler = app_scheduler(app)
    rtc_registry: RtcSessionRegistry = app_rtc_registry(app)
    conversation_runtimes: dict[str, _ConversationRuntime] = {}
    rtc_runtimes: dict[str, RtcRuntime] = {}

    async def auth(ctx: ConnectionContext) -> None:
        authorize_api_key_connection(ctx)

    pond = PondSocket()
    endpoint = pond.create_endpoint("/", auth)

    async def close_conversation_runtime(runtime: _ConversationRuntime) -> None:
        await runtime.conversation.close()

    async def build_conversation_runtime(channel: Channel, user_id: str, session_id: str) -> _ConversationRuntime:
        orchestrator = ConversationOrchestrator(scheduler=scheduler)
        conversation = ConversationRuntime(
            orchestrator,
            unknown_message_label="unknown conversation message type",
        )
        conversation.start_event_pump(
            lambda event: broadcast_conversation_event_to_user(
                event,
                channel=channel,
                user_id=user_id,
                session_id=session_id,
            )
        )
        return _ConversationRuntime(
            channel_name=channel.name,
            session_id=session_id,
            user_id=user_id,
            conversation=conversation,
        )

    async def build_rtc_runtime(
        channel: Channel,
        user_id: str,
        session_id: str,
    ) -> RtcRuntime:
        return RtcRuntime(
            scheduler=scheduler,
            registry=rtc_registry,
            session_id=session_id,
            transport="pondsocket",
            emit=lambda wire: try_broadcast_wire_to_user(
                channel,
                user_id,
                wire,
                session_id=session_id,
            ),
        )

    async def on_conversation_join(ctx: JoinContext) -> None:
        if await decline_if_channel_attached(ctx, "conversation channel already attached"):
            return
        session_id = pondsocket_route_or_channel_session_id(ctx)
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
        await handle_pondsocket_control_event(
            ctx,
            runtime=None if runtime is None else runtime.conversation,
            missing_message="conversation session not attached",
            error_log_message="PondSocket conversation event error",
            logger=logger,
        )

    async def on_rtc_join(ctx: JoinContext) -> None:
        if await decline_if_channel_attached(ctx, "RTC control channel already attached"):
            return
        session_id = pondsocket_route_or_channel_session_id(ctx)
        try:
            runtime = await build_rtc_runtime(ctx.channel, ctx.transport.get_id(), session_id)
        except OperationError as exc:
            error = operation_error_to_http(exc)
            await ctx.decline(error.status_code, str(error.detail))
            return
        rtc_runtimes[ctx.channel.name] = runtime
        try:
            await ctx.accept({"session_id": session_id, "kind": "rtc"})
            await runtime.start()
        except Exception:
            rtc_runtimes.pop(ctx.channel.name, None)
            await runtime.close(reason="join_failed")
            raise

    async def on_rtc_leave(ctx: LeaveContext) -> None:
        runtime = rtc_runtimes.pop(ctx.channel.name, None)
        if runtime is not None:
            await runtime.close(reason="transport_closed")

    async def on_rtc_event(ctx: EventContext) -> None:
        runtime = rtc_runtimes.get(ctx.channel.name)
        await handle_pondsocket_control_event(
            ctx,
            runtime=runtime,
            missing_message="RTC control session not attached",
            error_log_message="PondSocket RTC control event error",
            logger=logger,
        )

    conversation_lobby = endpoint.create_channel("/conversation/:session_id", on_conversation_join)
    conversation_lobby.on_leave(on_conversation_leave)
    conversation_lobby.on_message("*", on_conversation_event)

    rtc_lobby = endpoint.create_channel("/rtc/:session_id", on_rtc_join)
    rtc_lobby.on_leave(on_rtc_leave)
    rtc_lobby.on_message("*", on_rtc_event)

    set_app_pondsocket_gateway(app, pondsocket=pond, mount_path=mount_path)
    app.include_router(router)
    logger.info("Enabled PondSocket gateway at %s", mount_path)
    return True
