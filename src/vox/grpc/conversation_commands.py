"""gRPC conversation-control command decoding."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from vox.grpc import vox_pb2
from vox.operations.conversation import ConversationOrchestrator, execute_conversation_command
from vox.operations.errors import InvalidConfigError

_POLICY_FIELDS = (
    "allow_interrupt_while_speaking",
    "min_interrupt_duration_ms",
    "max_endpointing_delay_ms",
    "stable_speaking_min_ms",
    "false_interruption_timeout_ms",
    "min_interrupt_words",
    "partial_interrupts",
    "dynamic_endpointing",
    "min_endpointing_delay_ms",
    "speaking_interrupt_min_duration_ms",
    "speaking_interrupt_min_words",
    "self_echo_min_words",
    "self_echo_min_overlap",
    "aec_warmup_ms",
    "backchannel_end_cooldown_ms",
    "vad_min_silence_ms",
)


@dataclass(frozen=True)
class GrpcConversationCommand:
    message: dict


def conversation_session_update_to_message(
    update: vox_pb2.ConversationSessionUpdate,
) -> dict:
    session: dict = {
        "stt_model": update.stt_model,
        "tts_model": update.tts_model,
    }
    if update.voice:
        session["voice"] = update.voice
    if update.language:
        session["language"] = update.language
    if update.sample_rate:
        session["sample_rate"] = update.sample_rate
    if update.turn_profile:
        session["turn_profile"] = update.turn_profile
    if update.vad_backend:
        session["vad_backend"] = update.vad_backend
    if update.turn_detector:
        session["turn_detector"] = update.turn_detector
    if update.include_word_timestamps:
        session["include_word_timestamps"] = True
    if update.HasField("policy"):
        policy_overrides: dict[str, int | float | bool] = {}
        policy_pb = update.policy
        for field_name in _POLICY_FIELDS:
            if policy_pb.HasField(field_name):
                policy_overrides[field_name] = getattr(policy_pb, field_name)
        if policy_overrides:
            session["turn_policy"] = policy_overrides
    return {"type": "session.update", "session": session}


def converse_client_message_to_command(
    client_msg: vox_pb2.ConverseClientMessage,
) -> GrpcConversationCommand:
    kind = client_msg.WhichOneof("msg")
    if kind == "session_update":
        return GrpcConversationCommand(
            message=conversation_session_update_to_message(client_msg.session_update),
        )
    if kind == "audio_append":
        return GrpcConversationCommand(
            message={
                "type": "input_audio_buffer.append",
                "audio_pcm16": client_msg.audio_append.pcm16,
                "sample_rate": client_msg.audio_append.sample_rate,
            },
        )
    return _response_command(kind, client_msg, unknown_message_label="unknown message kind")


def rtc_control_message_to_command(
    client_msg: vox_pb2.RtcControlClientMessage,
) -> GrpcConversationCommand:
    kind = client_msg.WhichOneof("msg")
    if kind == "session_update":
        return GrpcConversationCommand(
            message=conversation_session_update_to_message(client_msg.session_update),
        )
    if kind == "client_event":
        event_name = client_msg.client_event.event.strip()
        if not event_name:
            raise InvalidConfigError("client_event requires a non-empty event")
        try:
            payload = json.loads(client_msg.client_event.payload_json or "null")
        except json.JSONDecodeError as exc:
            raise InvalidConfigError(f"client_event requires valid payload JSON: {exc}") from exc
        return GrpcConversationCommand(
            message={"type": "client.event", "event": event_name, "payload": payload},
        )
    return _response_command(kind, client_msg, unknown_message_label="unknown control message kind")


async def execute_converse_client_message(
    orchestrator: ConversationOrchestrator,
    client_msg: vox_pb2.ConverseClientMessage,
) -> None:
    command = converse_client_message_to_command(client_msg)
    await execute_conversation_command(
        orchestrator,
        command.message,
        require_config_message="send session_update first",
        unknown_message_label="unknown message kind",
    )


async def execute_rtc_control_message(
    orchestrator: ConversationOrchestrator,
    client_msg: vox_pb2.RtcControlClientMessage,
    *,
    client_event_handler: Callable[[str, Any], Awaitable[None] | None],
) -> None:
    command = rtc_control_message_to_command(client_msg)
    await execute_conversation_command(
        orchestrator,
        command.message,
        allow_input_audio=False,
        client_event_handler=client_event_handler,
        require_config_message="send session_update first",
        unknown_message_label="unknown control message kind",
    )


def _response_command(
    kind: str | None,
    client_msg: vox_pb2.ConverseClientMessage | vox_pb2.RtcControlClientMessage,
    *,
    unknown_message_label: str,
) -> GrpcConversationCommand:
    if kind == "response_start":
        message: dict = {"type": "response.start"}
        if client_msg.response_start.HasField("allow_interruptions"):
            message["allow_interruptions"] = client_msg.response_start.allow_interruptions
        return GrpcConversationCommand(message=message)
    if kind == "response_delta":
        message = {
            "type": "response.delta",
            "delta": client_msg.response_delta.delta,
        }
        if client_msg.response_delta.HasField("allow_interruptions"):
            message["allow_interruptions"] = client_msg.response_delta.allow_interruptions
        return GrpcConversationCommand(message=message)
    if kind == "response_commit":
        return GrpcConversationCommand(message={"type": "response.commit"})
    if kind == "response_cancel":
        return GrpcConversationCommand(message={"type": "response.cancel"})
    if kind == "response_replace_text":
        message = {
            "type": "response.replace_text",
            "text": client_msg.response_replace_text.text,
        }
        if client_msg.response_replace_text.HasField("allow_interruptions"):
            message["allow_interruptions"] = client_msg.response_replace_text.allow_interruptions
        return GrpcConversationCommand(message=message)
    raise InvalidConfigError(f"{unknown_message_label}: {kind!r}")
