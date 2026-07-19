from __future__ import annotations

import base64
import logging
from contextlib import suppress
from typing import Any

from vox.operations.conversation import (
    ConvEvent,
    conversation_error_fields,
    conversation_wire_event_payload,
    parse_allow_interruptions,
    parse_response_generation_id,
    parse_response_text,
    parse_session_update,
    serialize_conversation_event,
)
from vox.operations.conversation_commands import (
    AudioAppendCommand,
    ConversationCommand,
    ResponseCancelCommand,
    ResponseCommitCommand,
    ResponseDeltaCommand,
    ResponseReplaceTextCommand,
    ResponseStartCommand,
    SessionUpdateCommand,
    UnknownCommand,
    client_event_command,
)
from vox.operations.errors import InvalidConfigError, OperationError
from vox.operations.rtc_runtime import (
    LIFECYCLE_CRITICAL_WIRE_TYPES,
    RtcCandidateCommand,
    RtcCloseCommand,
    RtcCommand,
    RtcOfferCommand,
)

logger = logging.getLogger(__name__)


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


async def try_broadcast_wire_to_user(
    channel: Any,
    user_id: str,
    wire: dict[str, Any],
    *,
    session_id: str | None = None,
) -> bool:
    try:
        await broadcast_wire_to_user(channel, user_id, wire)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "PondSocket broadcast failed for event %s (session=%s, user=%s): %s",
            wire.get("type"),
            session_id,
            user_id,
            exc,
        )
        return False


async def deliver_wire_to_user(
    channel: Any,
    user_id: str,
    wire: dict[str, Any],
    *,
    session_id: str | None = None,
) -> bool:
    if await try_broadcast_wire_to_user(channel, user_id, wire, session_id=session_id):
        return True
    event_type = str(wire.get("type") or "")
    if event_type not in LIFECYCLE_CRITICAL_WIRE_TYPES:
        return False
    if await try_broadcast_wire_to_user(channel, user_id, wire, session_id=session_id):
        return True
    logger.error(
        "Lost lifecycle event %s for session %s (user=%s) after retry",
        event_type,
        session_id,
        user_id,
    )
    return False


async def broadcast_conversation_event_to_user(
    event: ConvEvent,
    *,
    channel: Any,
    user_id: str,
    session_id: str | None = None,
) -> None:
    wire = serialize_conversation_event(event)
    if wire is not None:
        await deliver_wire_to_user(channel, user_id, wire, session_id=session_id)


async def reply_pondsocket_error(
    ctx: Any,
    message: str,
    *,
    code: str = "command_invalid",
    recoverable: bool = True,
    generation_id: str | None = None,
) -> None:
    if ctx.has_replied():
        return
    payload: dict[str, Any] = {
        "message": message,
        "code": code,
        "recoverable": recoverable,
    }
    if generation_id:
        payload["generation_id"] = generation_id
    with suppress(Exception):
        await ctx.reply("error", payload)


def decode_pondsocket_command(event_name: str, payload: Any) -> ConversationCommand | RtcCommand:
    if not isinstance(payload, dict):
        raise InvalidConfigError(f"{event_name} requires an object payload")
    message = {"type": event_name, **payload}
    if event_name == "rtc.offer":
        offer = message.get("offer")
        if not isinstance(offer, dict):
            offer = message
        return RtcOfferCommand(
            offer_type=str(offer.get("type") or offer.get("sdpType") or offer.get("sdp_type") or "offer"),
            sdp=str(offer.get("sdp") or ""),
            restart=bool(message.get("restart", False)),
        )
    if event_name == "rtc.ice_candidate":
        candidate = message.get("candidate")
        if candidate is None:
            return RtcCandidateCommand(candidate=None)
        if isinstance(candidate, dict):
            return RtcCandidateCommand(
                candidate=str(candidate.get("candidate") or ""),
                sdp_mid=_optional_string(candidate.get("sdpMid") or candidate.get("sdp_mid")),
                sdp_m_line_index=_optional_int(
                    candidate.get("sdpMLineIndex")
                    if "sdpMLineIndex" in candidate
                    else candidate.get("sdp_m_line_index")
                ),
                username_fragment=_optional_string(
                    candidate.get("usernameFragment") or candidate.get("username_fragment")
                ),
            )
        return RtcCandidateCommand(candidate=str(candidate))
    if event_name == "rtc.close":
        return RtcCloseCommand(reason=str(message.get("reason") or "client_closed"))
    if event_name == "session.update":
        return SessionUpdateCommand(config=parse_session_update(message))
    if event_name == "client.event":
        return client_event_command(message.get("event"), message.get("payload"))
    if event_name == "input_audio_buffer.append":
        audio_b64 = message.get("audio")
        if not audio_b64:
            raise InvalidConfigError("audio field required")
        try:
            pcm16 = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise InvalidConfigError(f"invalid base64 audio: {exc}") from exc
        return AudioAppendCommand(
            pcm16=pcm16,
            sample_rate=int(message.get("sample_rate", 0)) or None,
        )
    if event_name == "response.start":
        return ResponseStartCommand(
            allow_interruptions=parse_allow_interruptions(message),
            generation_id=parse_response_generation_id(message),
        )
    if event_name == "response.delta":
        return ResponseDeltaCommand(
            text=parse_response_text(message, "delta") or "",
            allow_interruptions=parse_allow_interruptions(message),
            generation_id=parse_response_generation_id(message),
        )
    if event_name == "response.commit":
        return ResponseCommitCommand(generation_id=parse_response_generation_id(message))
    if event_name == "response.cancel":
        return ResponseCancelCommand(generation_id=parse_response_generation_id(message))
    if event_name == "response.replace_text":
        return ResponseReplaceTextCommand(
            text=parse_response_text(message, "text") or "",
            allow_interruptions=parse_allow_interruptions(message),
        )
    return UnknownCommand(name=event_name)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


async def handle_pondsocket_control_event(
    ctx: Any,
    *,
    runtime: Any | None,
    missing_message: str,
    error_log_message: str,
    logger: logging.Logger,
) -> None:
    if runtime is None:
        await reply_pondsocket_error(ctx, missing_message)
        return
    try:
        command = decode_pondsocket_command(ctx.event_name, ctx.get_payload())
        await runtime.dispatch(command)
    except OperationError as exc:
        code, recoverable, generation_id = conversation_error_fields(exc)
        await reply_pondsocket_error(
            ctx,
            str(exc),
            code=code,
            recoverable=recoverable,
            generation_id=generation_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(error_log_message)
        await reply_pondsocket_error(ctx, str(exc))
