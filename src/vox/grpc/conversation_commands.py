"""gRPC conversation-control command decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from vox.conversation.profiles import DEFAULT_TURN_PROFILE, resolve_turn_policy
from vox.grpc import vox_pb2
from vox.operations.conversation import ConversationSessionConfig
from vox.operations.errors import InvalidConfigError
from vox.streaming.types import TARGET_SAMPLE_RATE

GrpcConversationCommandKind = Literal["session_update", "command"]


@dataclass(frozen=True)
class GrpcConversationCommand:
    kind: GrpcConversationCommandKind
    config: ConversationSessionConfig | None = None
    message: dict | None = None


def conversation_session_update_to_config(
    update: vox_pb2.ConversationSessionUpdate,
) -> ConversationSessionConfig:
    if not update.stt_model:
        raise InvalidConfigError("session_update requires stt_model")
    if not update.tts_model:
        raise InvalidConfigError("session_update requires tts_model")

    policy_overrides: dict[str, int | float | bool] = {}
    if update.HasField("policy"):
        policy_pb = update.policy
        for field_name in (
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
        ):
            if policy_pb.HasField(field_name):
                policy_overrides[field_name] = getattr(policy_pb, field_name)
    try:
        turn_profile, policy = resolve_turn_policy(
            update.turn_profile or DEFAULT_TURN_PROFILE,
            policy_overrides,
        )
    except ValueError as exc:
        raise InvalidConfigError(str(exc)) from exc

    return ConversationSessionConfig(
        stt_model=update.stt_model,
        tts_model=update.tts_model,
        voice=update.voice or None,
        language=update.language or "en",
        sample_rate=update.sample_rate or TARGET_SAMPLE_RATE,
        turn_profile=turn_profile,
        vad_backend=update.vad_backend or "silero",
        turn_detector=update.turn_detector or "livekit",
        policy=policy,
        include_word_timestamps=bool(update.include_word_timestamps),
    )


def converse_client_message_to_command(
    client_msg: vox_pb2.ConverseClientMessage,
) -> GrpcConversationCommand:
    kind = client_msg.WhichOneof("msg")
    if kind == "session_update":
        return GrpcConversationCommand(
            kind="session_update",
            config=conversation_session_update_to_config(client_msg.session_update),
        )
    if kind == "audio_append":
        return GrpcConversationCommand(
            kind="command",
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
            kind="session_update",
            config=conversation_session_update_to_config(client_msg.session_update),
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
            kind="command",
            message={"type": "client.event", "event": event_name, "payload": payload},
        )
    return _response_command(kind, client_msg, unknown_message_label="unknown control message kind")


def _response_command(
    kind: str | None,
    client_msg: vox_pb2.ConverseClientMessage | vox_pb2.RtcControlClientMessage,
    *,
    unknown_message_label: str,
) -> GrpcConversationCommand:
    if kind == "response_start":
        return GrpcConversationCommand(kind="command", message={"type": "response.start"})
    if kind == "response_delta":
        return GrpcConversationCommand(
            kind="command",
            message={
                "type": "response.delta",
                "delta": client_msg.response_delta.delta,
            },
        )
    if kind == "response_commit":
        return GrpcConversationCommand(kind="command", message={"type": "response.commit"})
    if kind == "response_cancel":
        return GrpcConversationCommand(kind="command", message={"type": "response.cancel"})
    raise InvalidConfigError(f"{unknown_message_label}: {kind!r}")
