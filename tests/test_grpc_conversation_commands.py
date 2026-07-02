from __future__ import annotations

import pytest

from vox.grpc import vox_pb2
from vox.grpc.conversation_commands import (
    conversation_session_update_to_message,
    converse_client_message_to_command,
    execute_converse_client_message,
    execute_rtc_control_message,
    rtc_control_message_to_command,
)
from vox.operations.conversation import (
    TURN_POLICY_OVERRIDE_FIELDS,
    ConversationOrchestrator,
    parse_session_update,
)
from vox.operations.errors import InvalidConfigError


def test_grpc_session_update_decodes_to_shared_command_shape():
    command = conversation_session_update_to_message(
        vox_pb2.ConversationSessionUpdate(
            stt_model="parakeet-stt-onnx:tdt-0.6b-v3",
            tts_model="kokoro-tts-onnx:v1.0",
            voice="af_heart",
            turn_profile="headset",
            policy=vox_pb2.ConversationTurnPolicy(
                speaking_interrupt_min_duration_ms=300,
            ),
        )
    )

    assert command == {
        "type": "session.update",
        "session": {
            "stt_model": "parakeet-stt-onnx:tdt-0.6b-v3",
            "tts_model": "kokoro-tts-onnx:v1.0",
            "voice": "af_heart",
            "turn_profile": "headset",
            "turn_policy": {
                "speaking_interrupt_min_duration_ms": 300,
            },
        },
    }
    config = parse_session_update(command)
    assert config.stt_model == "parakeet-stt-onnx:tdt-0.6b-v3"
    assert config.tts_model == "kokoro-tts-onnx:v1.0"
    assert config.voice == "af_heart"
    assert config.turn_profile == "headset"
    assert config.policy is not None
    assert config.policy.speaking_interrupt_min_duration_ms == 300
    assert config.policy.aec_warmup_ms == 250


def test_grpc_session_update_preserves_every_operation_owned_policy_override():
    command = conversation_session_update_to_message(
        vox_pb2.ConversationSessionUpdate(
            stt_model="parakeet-stt-onnx:tdt-0.6b-v3",
            tts_model="kokoro-tts-onnx:v1.0",
            policy=vox_pb2.ConversationTurnPolicy(
                allow_interrupt_while_speaking=False,
                min_interrupt_duration_ms=101,
                max_endpointing_delay_ms=102,
                stable_speaking_min_ms=103,
                false_interruption_timeout_ms=104,
                min_interrupt_words=105,
                partial_interrupts=True,
                dynamic_endpointing=False,
                min_endpointing_delay_ms=106,
                speaking_interrupt_min_duration_ms=107,
                speaking_interrupt_min_words=108,
                self_echo_min_words=109,
                self_echo_min_overlap=0.72,
                aec_warmup_ms=110,
                backchannel_end_cooldown_ms=111,
                vad_min_silence_ms=112,
            ),
        )
    )

    policy = command["session"]["turn_policy"]

    assert tuple(policy) == TURN_POLICY_OVERRIDE_FIELDS
    assert policy == {
        "allow_interrupt_while_speaking": False,
        "min_interrupt_duration_ms": 101,
        "max_endpointing_delay_ms": 102,
        "stable_speaking_min_ms": 103,
        "false_interruption_timeout_ms": 104,
        "min_interrupt_words": 105,
        "partial_interrupts": True,
        "dynamic_endpointing": False,
        "min_endpointing_delay_ms": 106,
        "speaking_interrupt_min_duration_ms": 107,
        "speaking_interrupt_min_words": 108,
        "self_echo_min_words": 109,
        "self_echo_min_overlap": pytest.approx(0.72),
        "aec_warmup_ms": 110,
        "backchannel_end_cooldown_ms": 111,
        "vad_min_silence_ms": 112,
    }

    config = parse_session_update(command)
    assert config.policy is not None
    assert config.policy.vad_min_silence_ms == 112
    assert config.policy.self_echo_min_overlap == pytest.approx(0.72)


def test_converse_audio_append_decodes_to_shared_command_shape():
    command = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(
            audio_append=vox_pb2.ConversationAudioAppend(pcm16=b"abc", sample_rate=16_000),
        )
    )

    assert command.message == {
        "type": "input_audio_buffer.append",
        "audio_pcm16": b"abc",
        "sample_rate": 16_000,
    }


def test_converse_and_rtc_response_delta_decode_to_same_command_shape():
    converse = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(
            response_delta=vox_pb2.ConversationResponseDelta(delta="hello"),
        )
    )
    rtc = rtc_control_message_to_command(
        vox_pb2.RtcControlClientMessage(
            response_delta=vox_pb2.ConversationResponseDelta(delta="hello"),
        )
    )

    assert converse.message == {"type": "response.delta", "delta": "hello"}
    assert rtc.message == converse.message


def test_converse_response_start_preserves_allow_interruptions():
    command = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(
            response_start=vox_pb2.ConversationResponseStart(allow_interruptions=False),
        )
    )

    assert command.message == {
        "type": "response.start",
        "allow_interruptions": False,
    }


def test_response_delta_preserves_allow_interruptions_for_both_grpc_transports():
    converse = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(
            response_delta=vox_pb2.ConversationResponseDelta(
                delta="hello",
                allow_interruptions=False,
            ),
        )
    )
    rtc = rtc_control_message_to_command(
        vox_pb2.RtcControlClientMessage(
            response_delta=vox_pb2.ConversationResponseDelta(
                delta="hello",
                allow_interruptions=False,
            ),
        )
    )

    assert converse.message == {
        "type": "response.delta",
        "delta": "hello",
        "allow_interruptions": False,
    }
    assert rtc.message == converse.message


def test_converse_and_rtc_response_replace_text_decode_to_same_command_shape():
    converse = converse_client_message_to_command(
        vox_pb2.ConverseClientMessage(
            response_replace_text=vox_pb2.ConversationResponseReplaceText(
                text="replacement",
                allow_interruptions=False,
            ),
        )
    )
    rtc = rtc_control_message_to_command(
        vox_pb2.RtcControlClientMessage(
            response_replace_text=vox_pb2.ConversationResponseReplaceText(
                text="replacement",
                allow_interruptions=False,
            ),
        )
    )

    assert converse.message == {
        "type": "response.replace_text",
        "text": "replacement",
        "allow_interruptions": False,
    }
    assert rtc.message == converse.message


def test_rtc_client_event_decodes_json_payload():
    command = rtc_control_message_to_command(
        vox_pb2.RtcControlClientMessage(
            client_event=vox_pb2.RtcClientEvent(
                event="app.marker",
                payload_json='{"n": 1}',
            ),
        )
    )

    assert command.message == {
        "type": "client.event",
        "event": "app.marker",
        "payload": {"n": 1},
    }


def test_rtc_client_event_rejects_invalid_json_payload():
    with pytest.raises(InvalidConfigError, match="valid payload JSON"):
        rtc_control_message_to_command(
            vox_pb2.RtcControlClientMessage(
                client_event=vox_pb2.RtcClientEvent(
                    event="app.marker",
                    payload_json="{",
                ),
            )
        )


def test_rtc_attach_is_not_a_conversation_command_after_attach_phase():
    with pytest.raises(InvalidConfigError, match="unknown control message kind: 'attach'"):
        rtc_control_message_to_command(
            vox_pb2.RtcControlClientMessage(
                attach=vox_pb2.RtcControlAttach(session_id="rtc_123"),
            )
        )


@pytest.mark.asyncio
async def test_execute_converse_client_message_owns_session_update_error_policy():
    orchestrator = ConversationOrchestrator(scheduler=object())

    with pytest.raises(InvalidConfigError, match="send session_update first"):
        await execute_converse_client_message(
            orchestrator,
            vox_pb2.ConverseClientMessage(
                audio_append=vox_pb2.ConversationAudioAppend(pcm16=b"\x00" * 100),
            ),
        )


@pytest.mark.asyncio
async def test_execute_rtc_control_message_owns_session_update_error_policy():
    orchestrator = ConversationOrchestrator(scheduler=object())

    with pytest.raises(InvalidConfigError, match="send session_update first"):
        await execute_rtc_control_message(
            orchestrator,
            vox_pb2.RtcControlClientMessage(
                response_delta=vox_pb2.ConversationResponseDelta(delta="hello"),
            ),
            client_event_handler=lambda _event, _payload: None,
        )


@pytest.mark.asyncio
async def test_execute_rtc_control_message_routes_client_events_before_session_update():
    orchestrator = ConversationOrchestrator(scheduler=object())
    received: list[tuple[str, object]] = []

    await execute_rtc_control_message(
        orchestrator,
        vox_pb2.RtcControlClientMessage(
            client_event=vox_pb2.RtcClientEvent(
                event="app.marker",
                payload_json='{"n": 1}',
            ),
        ),
        client_event_handler=lambda event, payload: received.append((event, payload)),
    )

    assert received == [("app.marker", {"n": 1})]
