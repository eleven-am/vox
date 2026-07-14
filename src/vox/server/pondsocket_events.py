from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from vox.operations.conversation import (
    conversation_wire_event_payload,
    pondsocket_event_to_conversation_command,
    serialize_conversation_event,
)
from vox.operations.errors import OperationError
from vox.server.rtc_timeline import RtcTurnTimeline, rtc_audio_stats


def pondsocket_route_or_channel_session_id(ctx: Any) -> str:
    """Return the route session_id, falling back to channel suffix for older PondSocket contexts."""
    return str(ctx.route.params.get("session_id") or ctx.channel.name.rsplit("/", 1)[-1])


async def decline_if_channel_attached(ctx: Any, message: str) -> bool:
    if await ctx.channel.user_count() <= 0:
        return False
    await ctx.decline(409, message)
    return True


async def broadcast_wire_to_user(channel: Any, user_id: str, wire: dict[str, Any]) -> None:
    event_name, payload = conversation_wire_event_payload(wire)
    await channel.broadcast_to(event_name, payload, user_id)


async def try_broadcast_wire_to_user(channel: Any, user_id: str, wire: dict[str, Any]) -> bool:
    with suppress(Exception):
        await broadcast_wire_to_user(channel, user_id, wire)
        return True
    return False


async def broadcast_conversation_events_to_user(
    *,
    orchestrator: Any,
    channel: Any,
    user_id: str,
) -> None:
    async for event in orchestrator.events():
        wire = serialize_conversation_event(event)
        if wire is not None:
            await try_broadcast_wire_to_user(channel, user_id, wire)


async def broadcast_rtc_control_events_to_user(
    *,
    orchestrator: Any,
    channel: Any,
    user_id: str,
    record: Any,
    session_id: str,
    prepare_event: Any,
) -> None:
    timeline = RtcTurnTimeline(session_id=session_id)
    async for event in orchestrator.events():
        prepared = prepare_event(record=record, session_id=session_id, event=event)
        if prepared.wire is not None:
            await try_broadcast_wire_to_user(channel, user_id, prepared.wire)
            timing = timeline.observe(
                prepared.wire,
                audio_stats=rtc_audio_stats(record),
            )
            if timing is not None:
                await try_broadcast_wire_to_user(channel, user_id, timing)
        if prepared.done:
            return


async def broadcast_rtc_client_events_to_user(
    *,
    record: Any,
    channel: Any,
    user_id: str,
) -> None:
    if record.control_events is None:
        return
    while True:
        event = await record.control_events.get()
        if event is None:
            return
        await try_broadcast_wire_to_user(channel, user_id, event)


async def reply_pondsocket_error(ctx: Any, message: str) -> None:
    if ctx.has_replied():
        return
    with suppress(Exception):
        await ctx.reply("error", {"message": message})


async def handle_pondsocket_control_event(
    ctx: Any,
    *,
    runtime: Any | None,
    missing_message: str,
    executor: Any,
    unknown_message_label: str,
    error_log_message: str,
    logger: logging.Logger,
    client_event_handler: Any = None,
) -> None:
    if runtime is None:
        await reply_pondsocket_error(ctx, missing_message)
        return
    try:
        message = pondsocket_event_to_conversation_command(ctx.event_name, ctx.get_payload())
        if client_event_handler is None:
            await executor(
                runtime.orchestrator,
                message,
                unknown_message_label=unknown_message_label,
            )
        else:
            await executor(
                runtime.orchestrator,
                message,
                client_event_handler=client_event_handler,
                unknown_message_label=unknown_message_label,
            )
    except OperationError as exc:
        await reply_pondsocket_error(ctx, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception(error_log_message)
        await reply_pondsocket_error(ctx, str(exc))
