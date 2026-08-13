from __future__ import annotations

import base64

import pytest

from vox.conversation.response_output import ResponseOutputConfig
from vox.grpc import vox_pb2
from vox.grpc.conversation_commands import (
    converse_client_message_to_command,
    rtc_control_message_to_command,
)
from vox.grpc.conversation_events import conversation_event_to_pb
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvAudioResumeEvent,
    ConvAudioSuspendEvent,
    ConvErrorEvent,
    ConvInterruptionDetectedEvent,
    ConvInterruptionFalsePositiveEvent,
    ConvResponseCancelledEvent,
    ConvResponseCommittedEvent,
    ConvResponseCreatedEvent,
    ConvResponseDoneEvent,
    ConvResponseSpokenTextEvent,
    ConvSpeechStartedEvent,
    ConvSpeechStoppedEvent,
    ConvStateChangedEvent,
    ConvTranscriptDeltaEvent,
    ConvTranscriptDoneEvent,
    ConvTurnEouPredictedEvent,
    parse_conversation_wire_event,
    serialize_conversation_event,
)
from vox.operations.conversation_commands import ClientEventCommand
from vox.operations.errors import InvalidConfigError
from vox.server.pondsocket_events import decode_pondsocket_command


@pytest.mark.parametrize(
    ("event_name", "payload", "grpc_message"),
    [
        (
            "session.update",
            {
                "stt_model": "stt:1",
                "tts_model": "tts:1",
                "voice": "voice-1",
                "turn_profile": "headset",
                "turn_policy": {"speaking_interrupt_min_duration_ms": 300},
            },
            vox_pb2.ConverseClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="stt:1",
                    tts_model="tts:1",
                    voice="voice-1",
                    turn_profile="headset",
                    policy=vox_pb2.ConversationTurnPolicy(
                        speaking_interrupt_min_duration_ms=300,
                    ),
                )
            ),
        ),
        (
            "input_audio_buffer.append",
            {"audio": base64.b64encode(b"\x01\x02\x03\x04").decode(), "sample_rate": 16_000},
            vox_pb2.ConverseClientMessage(
                audio_append=vox_pb2.ConversationAudioAppend(
                    pcm16=b"\x01\x02\x03\x04",
                    sample_rate=16_000,
                )
            ),
        ),
        (
            "response.start",
            {"allow_interruptions": False},
            vox_pb2.ConverseClientMessage(response_start=vox_pb2.ConversationResponseStart(allow_interruptions=False)),
        ),
        (
            "response.delta",
            {"delta": "hello", "allow_interruptions": False},
            vox_pb2.ConverseClientMessage(
                response_delta=vox_pb2.ConversationResponseDelta(
                    delta="hello",
                    allow_interruptions=False,
                )
            ),
        ),
        (
            "response.replace_text",
            {"text": "replacement", "allow_interruptions": False},
            vox_pb2.ConverseClientMessage(
                response_replace_text=vox_pb2.ConversationResponseReplaceText(
                    text="replacement",
                    allow_interruptions=False,
                )
            ),
        ),
        (
            "response.commit",
            {},
            vox_pb2.ConverseClientMessage(response_commit=vox_pb2.ConversationResponseCommit()),
        ),
        (
            "response.cancel",
            {},
            vox_pb2.ConverseClientMessage(response_cancel=vox_pb2.ConversationResponseCancel()),
        ),
        (
            "response.start",
            {"generation_id": "gen-1"},
            vox_pb2.ConverseClientMessage(
                response_start=vox_pb2.ConversationResponseStart(generation_id="gen-1")
            ),
        ),
        (
            "response.start",
            {
                "generation_id": "gen-new",
                "supersedes_generation_id": "gen-old",
            },
            vox_pb2.ConverseClientMessage(
                response_start=vox_pb2.ConversationResponseStart(
                    generation_id="gen-new",
                    supersedes_generation_id="gen-old",
                )
            ),
        ),
        (
            "response.delta",
            {"delta": "hello", "generation_id": "gen-1"},
            vox_pb2.ConverseClientMessage(
                response_delta=vox_pb2.ConversationResponseDelta(
                    delta="hello",
                    generation_id="gen-1",
                )
            ),
        ),
        (
            "response.commit",
            {"generation_id": "gen-1"},
            vox_pb2.ConverseClientMessage(
                response_commit=vox_pb2.ConversationResponseCommit(generation_id="gen-1")
            ),
        ),
        (
            "response.cancel",
            {"generation_id": "gen-1"},
            vox_pb2.ConverseClientMessage(
                response_cancel=vox_pb2.ConversationResponseCancel(generation_id="gen-1")
            ),
        ),
    ],
)
def test_pondsocket_and_grpc_decode_to_identical_canonical_commands(
    event_name,
    payload,
    grpc_message,
):
    assert decode_pondsocket_command(event_name, payload) == converse_client_message_to_command(grpc_message)


def test_pondsocket_and_grpc_rtc_client_events_decode_identically():
    pondsocket_command = decode_pondsocket_command(
        "client.event",
        {"event": "ui.select", "payload": {"id": "a"}},
    )
    grpc_command = rtc_control_message_to_command(
        vox_pb2.RtcControlClientMessage(
            client_event=vox_pb2.RtcClientEvent(
                event="ui.select",
                payload_json='{"id": "a"}',
            )
        )
    )

    assert (
        pondsocket_command
        == grpc_command
        == ClientEventCommand(
            event="ui.select",
            payload={"id": "a"},
        )
    )


@pytest.mark.parametrize(
    ("event_name", "payload", "grpc_message"),
    [
        (
            "rtc.offer",
            {"offer": {"type": "offer", "sdp": "offer-sdp"}, "restart": True},
            vox_pb2.RtcControlClientMessage(
                offer=vox_pb2.RtcControlOffer(
                    offer=vox_pb2.RtcSessionDescription(type="offer", sdp="offer-sdp"),
                    restart=True,
                )
            ),
        ),
        (
            "rtc.ice_candidate",
            {
                "candidate": {
                    "candidate": "candidate:1",
                    "sdpMid": "audio",
                    "sdpMLineIndex": 0,
                    "usernameFragment": "ufrag",
                }
            },
            vox_pb2.RtcControlClientMessage(
                candidate=vox_pb2.RtcIceCandidate(
                    candidate="candidate:1",
                    sdp_mid="audio",
                    sdp_m_line_index=0,
                    username_fragment="ufrag",
                )
            ),
        ),
        (
            "rtc.ice_candidate",
            {"candidate": None},
            vox_pb2.RtcControlClientMessage(candidates_complete=vox_pb2.RtcIceCandidatesComplete()),
        ),
        (
            "rtc.close",
            {"reason": "client_closed"},
            vox_pb2.RtcControlClientMessage(close=vox_pb2.RtcControlClose(reason="client_closed")),
        ),
    ],
)
def test_pondsocket_and_grpc_rtc_signaling_decode_identically(
    event_name,
    payload,
    grpc_message,
):
    assert decode_pondsocket_command(event_name, payload) == rtc_control_message_to_command(grpc_message)


@pytest.mark.parametrize(
    "event",
    [
        ConvSpeechStartedEvent(timestamp_ms=10),
        ConvSpeechStoppedEvent(timestamp_ms=20),
        ConvTranscriptDeltaEvent(delta="hel", start_ms=10, end_ms=20),
        ConvTranscriptDoneEvent(
            transcript="hello",
            language="en",
            start_ms=10,
            end_ms=50,
            eou_probability=0.8,
            entities=({"text": "hello", "label": "greeting"},),
            topics=("greeting",),
            words=({"word": "hello", "start": 0.01, "end": 0.05},),
        ),
        ConvTurnEouPredictedEvent(
            probability=0.8,
            threshold=0.5,
            decision="complete",
            action="commit",
            delay_ms=0,
            turn_detector="livekit",
            start_ms=10,
            end_ms=50,
        ),
        ConvResponseCreatedEvent(response_id="resp_1"),
        ConvResponseCommittedEvent(response_id="resp_1"),
        ConvAudioDeltaEvent(
            audio=b"\x01\x02\x03\x04",
            sample_rate=24_000,
            audio_format="pcm16",
            response_id="resp_1",
            sequence=3,
        ),
        ConvAudioClearEvent(response_id="resp_1"),
        ConvAudioSuspendEvent(response_id="resp_1", candidate_id=5),
        ConvAudioResumeEvent(response_id="resp_1", candidate_id=5),
        ConvInterruptionDetectedEvent(
            response_id="resp_1",
            vad_active_ms=420,
            partial_transcript="wait",
            reason="speech_evidence",
        ),
        ConvInterruptionFalsePositiveEvent(
            response_id="resp_1",
            vad_active_ms=420,
            partial_transcript="anyway",
            reason="insufficient_evidence",
        ),
        ConvResponseCancelledEvent(response_id="resp_1"),
        ConvResponseSpokenTextEvent(
            response_id="resp_1",
            spoken_text="The spoken prefix",
            partial_status="matched",
            played_audio_ms=640,
        ),
        ConvResponseDoneEvent(response_id="resp_1"),
        ConvStateChangedEvent(state="listening", previous_state="speaking"),
        ConvErrorEvent(message="boom"),
        ConvErrorEvent(
            message="stale response generation",
            code="response_stale_generation",
            recoverable=True,
            generation_id="gen-1",
        ),
        ConvErrorEvent(
            message="action failed; response pipeline reset",
            code="session_failed",
            recoverable=False,
        ),
        ConvResponseCreatedEvent(
            response_id="resp_1",
            generation_id="gen-1",
            output=ResponseOutputConfig(
                model="chatterbox-tts-turbo:0.1.7",
                voice="samantha",
                language="fr",
                speed=0.9,
                params={"temperature": 0.7},
            ),
        ),
        ConvResponseCommittedEvent(response_id="resp_1", generation_id="gen-1"),
        ConvResponseDoneEvent(response_id="resp_1", generation_id="gen-1"),
        ConvResponseCancelledEvent(response_id="resp_1", generation_id="gen-1"),
        ConvResponseSpokenTextEvent(
            response_id="resp_1",
            generation_id="gen-1",
            spoken_text="The spoken prefix",
            partial_status="matched",
            played_audio_ms=640,
        ),
        ConvAudioClearEvent(response_id="resp_1", generation_id="gen-1"),
        ConvInterruptionDetectedEvent(
            response_id="resp_1",
            vad_active_ms=420,
            partial_transcript="wait",
            reason="speech_evidence",
            generation_id="gen-1",
        ),
        ConvInterruptionFalsePositiveEvent(
            response_id="resp_1",
            vad_active_ms=420,
            partial_transcript="anyway",
            reason="insufficient_evidence",
            generation_id="gen-1",
        ),
    ],
)
def test_json_and_protobuf_edges_encode_the_same_canonical_event(event):
    wire = serialize_conversation_event(event)
    assert wire is not None

    direct_pb = conversation_event_to_pb(event)
    reparsed = parse_conversation_wire_event(wire)
    assert reparsed is not None
    json_round_trip_pb = conversation_event_to_pb(reparsed)

    assert direct_pb is not None
    assert json_round_trip_pb == direct_pb


def test_audio_is_base64_only_at_json_edge_and_raw_in_protobuf():
    event = ConvAudioDeltaEvent(
        audio=b"\x00\x01\x02\x03",
        sample_rate=24_000,
        audio_format="pcm16",
        response_id="resp_1",
        sequence=1,
    )

    wire = serialize_conversation_event(event)
    protobuf = conversation_event_to_pb(event)

    assert wire is not None
    assert wire["audio"] == "AAECAw=="
    assert protobuf is not None
    assert protobuf.audio_delta.audio == event.audio


def test_pondsocket_rejects_malformed_audio_at_json_edge():
    with pytest.raises(InvalidConfigError, match="invalid base64 audio"):
        decode_pondsocket_command(
            "input_audio_buffer.append",
            {"audio": "not-base64!"},
        )
