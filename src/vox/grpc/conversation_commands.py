"""gRPC conversation-control command decoding."""

from __future__ import annotations

from google.protobuf.json_format import MessageToDict

from vox.conversation.profiles import DEFAULT_TURN_PROFILE
from vox.conversation.response_output import ResponseOutputOptions
from vox.grpc import vox_pb2
from vox.grpc.conversation_policy import conversation_turn_policy_overrides
from vox.operations.conversation import (
    build_conversation_session_config,
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
    client_event_command_from_payload_json,
)
from vox.operations.errors import InvalidConfigError
from vox.operations.rtc_runtime import (
    RtcCandidateCommand,
    RtcCloseCommand,
    RtcCommand,
    RtcOfferCommand,
)
from vox.streaming.eou import DEFAULT_TURN_DETECTOR
from vox.streaming.types import TARGET_SAMPLE_RATE


def conversation_session_update_to_command(
    update: vox_pb2.ConversationSessionUpdate,
) -> SessionUpdateCommand:
    policy_overrides = {}
    if update.HasField("policy"):
        policy_overrides = conversation_turn_policy_overrides(update.policy)
    return SessionUpdateCommand(
        config=build_conversation_session_config(
            stt_model=update.stt_model,
            tts_model=update.tts_model,
            voice=update.voice or None,
            language=update.language or "en",
            sample_rate=update.sample_rate or TARGET_SAMPLE_RATE,
            turn_profile=update.turn_profile or DEFAULT_TURN_PROFILE,
            vad_backend=update.vad_backend or "silero",
            turn_detector=update.turn_detector or DEFAULT_TURN_DETECTOR,
            policy_overrides=policy_overrides,
            include_word_timestamps=update.include_word_timestamps,
            speech_context=update.speech_context,
        )
    )


def converse_client_message_to_command(
    client_msg: vox_pb2.ConverseClientMessage,
) -> ConversationCommand:
    kind = client_msg.WhichOneof("msg")
    if kind == "session_update":
        return conversation_session_update_to_command(client_msg.session_update)
    if kind == "audio_append":
        return AudioAppendCommand(
            pcm16=client_msg.audio_append.pcm16,
            sample_rate=client_msg.audio_append.sample_rate or None,
        )
    return _response_command(kind, client_msg, unknown_message_label="unknown message kind")


def rtc_control_message_to_command(
    client_msg: vox_pb2.RtcControlClientMessage,
) -> RtcCommand:
    kind = client_msg.WhichOneof("msg")
    if kind == "offer":
        return RtcOfferCommand(
            offer_type=client_msg.offer.offer.type or "offer",
            sdp=client_msg.offer.offer.sdp,
            restart=client_msg.offer.restart,
        )
    if kind == "candidate":
        candidate = client_msg.candidate
        return RtcCandidateCommand(
            candidate=candidate.candidate,
            sdp_mid=candidate.sdp_mid if candidate.HasField("sdp_mid") else None,
            sdp_m_line_index=(candidate.sdp_m_line_index if candidate.HasField("sdp_m_line_index") else None),
            username_fragment=(candidate.username_fragment if candidate.HasField("username_fragment") else None),
        )
    if kind == "candidates_complete":
        return RtcCandidateCommand(candidate=None)
    if kind == "close":
        return RtcCloseCommand(reason=client_msg.close.reason or "client_closed")
    if kind == "session_update":
        return conversation_session_update_to_command(client_msg.session_update)
    if kind == "client_event":
        return client_event_command_from_payload_json(
            client_msg.client_event.event,
            client_msg.client_event.payload_json,
            invalid_json_message="client_event requires valid payload JSON",
            non_empty_message="client_event requires a non-empty event",
        )
    return _response_command(kind, client_msg, unknown_message_label="unknown control message kind")


def _response_command(
    kind: str | None,
    client_msg: vox_pb2.ConverseClientMessage | vox_pb2.RtcControlClientMessage,
    *,
    unknown_message_label: str,
) -> ConversationCommand:
    if kind == "response_start":
        output = None
        if client_msg.response_start.HasField("output"):
            raw_output = client_msg.response_start.output
            try:
                output = ResponseOutputOptions(
                    model=raw_output.model if raw_output.HasField("model") else None,
                    voice=raw_output.voice if raw_output.HasField("voice") else None,
                    language=raw_output.language if raw_output.HasField("language") else None,
                    speed=raw_output.speed if raw_output.HasField("speed") else None,
                    params=(
                        MessageToDict(raw_output.params)
                        if raw_output.HasField("params")
                        else None
                    ),
                )
            except ValueError as exc:
                raise InvalidConfigError(str(exc)) from exc
        return ResponseStartCommand(
            allow_interruptions=(
                client_msg.response_start.allow_interruptions
                if client_msg.response_start.HasField("allow_interruptions")
                else True
            ),
            generation_id=client_msg.response_start.generation_id or None,
            output=output,
        )
    if kind == "response_delta":
        return ResponseDeltaCommand(
            text=client_msg.response_delta.delta,
            allow_interruptions=(
                client_msg.response_delta.allow_interruptions
                if client_msg.response_delta.HasField("allow_interruptions")
                else True
            ),
            generation_id=client_msg.response_delta.generation_id or None,
        )
    if kind == "response_commit":
        return ResponseCommitCommand(generation_id=client_msg.response_commit.generation_id or None)
    if kind == "response_cancel":
        return ResponseCancelCommand(generation_id=client_msg.response_cancel.generation_id or None)
    if kind == "response_replace_text":
        return ResponseReplaceTextCommand(
            text=client_msg.response_replace_text.text,
            allow_interruptions=(
                client_msg.response_replace_text.allow_interruptions
                if client_msg.response_replace_text.HasField("allow_interruptions")
                else True
            ),
        )
    raise InvalidConfigError(f"{unknown_message_label}: {kind!r}")
