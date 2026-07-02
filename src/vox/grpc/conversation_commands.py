"""gRPC conversation-control command decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass

from vox.grpc import vox_pb2
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


def _response_command(
    kind: str | None,
    client_msg: vox_pb2.ConverseClientMessage | vox_pb2.RtcControlClientMessage,
    *,
    unknown_message_label: str,
) -> GrpcConversationCommand:
    if kind == "response_start":
        return GrpcConversationCommand(message={"type": "response.start"})
    if kind == "response_delta":
        return GrpcConversationCommand(
            message={
                "type": "response.delta",
                "delta": client_msg.response_delta.delta,
            },
        )
    if kind == "response_commit":
        return GrpcConversationCommand(message={"type": "response.commit"})
    if kind == "response_cancel":
        return GrpcConversationCommand(message={"type": "response.cancel"})
    raise InvalidConfigError(f"{unknown_message_label}: {kind!r}")
