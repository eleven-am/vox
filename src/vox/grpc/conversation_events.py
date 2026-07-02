"""gRPC conversation event encoding."""

from __future__ import annotations

import base64

from vox.grpc import vox_pb2
from vox.grpc.conversation_policy import conversation_turn_policy_pb
from vox.grpc.transcript_messages import entity_messages, word_timestamp_messages
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvErrorEvent,
    ConvEvent,
    ConvInterruptionDetectedEvent,
    ConvInterruptionFalsePositiveEvent,
    ConvResponseCancelledEvent,
    ConvResponseCommittedEvent,
    ConvResponseCreatedEvent,
    ConvResponseDoneEvent,
    ConvSessionCreatedEvent,
    ConvSpeechStartedEvent,
    ConvSpeechStoppedEvent,
    ConvStateChangedEvent,
    ConvTranscriptDeltaEvent,
    ConvTranscriptDoneEvent,
    ConvTurnEouPredictedEvent,
    control_event_client_payload_json,
    control_event_as_client_event,
    rtc_session_attached_payload,
)


def conversation_error_pb(message: str) -> vox_pb2.ConverseServerMessage:
    return vox_pb2.ConverseServerMessage(error=vox_pb2.ConversationError(message=message))


def rtc_session_attached_pb(session_id: str) -> vox_pb2.ConverseServerMessage:
    return vox_pb2.ConverseServerMessage(
        rtc_session_attached=vox_pb2.RtcSessionAttached(
            **rtc_session_attached_payload(session_id),
            provider="webrtc",
        ),
    )


def rtc_client_event_pb(event_name: str, payload_json: str) -> vox_pb2.ConverseServerMessage:
    return vox_pb2.ConverseServerMessage(
        client_event=vox_pb2.RtcClientEvent(
            event=event_name,
            payload_json=payload_json,
        ),
    )


def rtc_client_event_from_control_event(event: dict) -> vox_pb2.ConverseServerMessage:
    event_name, payload = control_event_as_client_event(event)
    return rtc_client_event_pb(event_name, control_event_client_payload_json(payload))


def conversation_event_to_pb(event: ConvEvent) -> vox_pb2.ConverseServerMessage | None:
    if isinstance(event, ConvSessionCreatedEvent):
        session_created = vox_pb2.ConversationSessionCreated(
            turn_profile=event.config.turn_profile,
        )
        session_created.policy.CopyFrom(conversation_turn_policy_pb(event.config.policy))
        return vox_pb2.ConverseServerMessage(session_created=session_created)
    if isinstance(event, ConvSpeechStartedEvent):
        return vox_pb2.ConverseServerMessage(
            speech_started=vox_pb2.ConversationSpeechStarted(timestamp_ms=event.timestamp_ms),
        )
    if isinstance(event, ConvSpeechStoppedEvent):
        return vox_pb2.ConverseServerMessage(
            speech_stopped=vox_pb2.ConversationSpeechStopped(timestamp_ms=event.timestamp_ms),
        )
    if isinstance(event, ConvTranscriptDeltaEvent):
        return vox_pb2.ConverseServerMessage(
            transcript_delta=vox_pb2.ConversationTranscriptDelta(
                delta=event.delta,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            ),
        )
    if isinstance(event, ConvTranscriptDoneEvent):
        msg = vox_pb2.ConversationTranscriptDone(
            transcript=event.transcript,
            language=event.language,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
        )
        if event.eou_probability is not None:
            msg.eou_probability = event.eou_probability
        msg.entities.extend(entity_messages(event.entities))
        msg.topics.extend(str(topic) for topic in event.topics)
        msg.words.extend(word_timestamp_messages(event.words))
        return vox_pb2.ConverseServerMessage(transcript_done=msg)
    if isinstance(event, ConvResponseCreatedEvent):
        return vox_pb2.ConverseServerMessage(
            response_created=vox_pb2.ConversationResponseCreated(response_id=event.response_id),
        )
    if isinstance(event, ConvAudioDeltaEvent):
        pcm = base64.b64decode(event.audio_b64) if event.audio_b64 else b""
        return vox_pb2.ConverseServerMessage(
            audio_delta=vox_pb2.ConversationAudioDelta(
                audio=pcm,
                sample_rate=event.sample_rate,
                response_id=event.response_id,
                sequence=event.sequence,
            ),
        )
    if isinstance(event, ConvAudioClearEvent):
        return vox_pb2.ConverseServerMessage(
            audio_clear=vox_pb2.ConversationAudioClear(response_id=event.response_id),
        )
    if isinstance(event, ConvResponseDoneEvent):
        return vox_pb2.ConverseServerMessage(
            response_done=vox_pb2.ConversationResponseDone(response_id=event.response_id),
        )
    if isinstance(event, ConvResponseCancelledEvent):
        return vox_pb2.ConverseServerMessage(
            response_cancelled=vox_pb2.ConversationResponseCancelled(response_id=event.response_id),
        )
    if isinstance(event, ConvResponseCommittedEvent):
        return vox_pb2.ConverseServerMessage(
            response_committed=vox_pb2.ConversationResponseCommitted(response_id=event.response_id),
        )
    if isinstance(event, ConvInterruptionDetectedEvent):
        return vox_pb2.ConverseServerMessage(
            interruption_detected=vox_pb2.ConversationInterruptionDetected(
                response_id=event.response_id,
                vad_active_ms=event.vad_active_ms,
                partial_transcript=event.partial_transcript or "",
            ),
        )
    if isinstance(event, ConvInterruptionFalsePositiveEvent):
        return vox_pb2.ConverseServerMessage(
            interruption_false_positive=vox_pb2.ConversationInterruptionFalsePositive(
                response_id=event.response_id,
                vad_active_ms=event.vad_active_ms,
                partial_transcript=event.partial_transcript or "",
                reason=event.reason or "",
            ),
        )
    if isinstance(event, ConvTurnEouPredictedEvent):
        return vox_pb2.ConverseServerMessage(
            turn_eou_predicted=vox_pb2.ConversationTurnEouPredicted(
                probability=event.probability,
                threshold=event.threshold,
                decision=event.decision,
                action=event.action,
                delay_ms=event.delay_ms,
                turn_detector=event.turn_detector,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            ),
        )
    if isinstance(event, ConvStateChangedEvent):
        return vox_pb2.ConverseServerMessage(
            state_changed=vox_pb2.ConversationStateChanged(
                state=event.state,
                previous_state=event.previous_state,
            ),
        )
    if isinstance(event, ConvErrorEvent):
        return conversation_error_pb(event.message)
    return None


def conversation_wire_event_to_pb(event: dict) -> vox_pb2.ConverseServerMessage | None:
    from vox.operations.conversation import parse_conversation_wire_event

    mapped = parse_conversation_wire_event(event)
    if mapped is None:
        return None
    return conversation_event_to_pb(mapped)
