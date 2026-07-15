from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)
from vox.operations.conversation import (
    RESPONSE_ALLOW_INTERRUPTION_FIELD,
    RESPONSE_COMMAND_ENVELOPE_FIELDS,
    RESPONSE_TEXT_COMPATIBILITY_FIELDS,
    SESSION_UPDATE_POLICY_FIELDS,
    SESSION_UPDATE_STT_MODEL_FIELDS,
    SESSION_UPDATE_TTS_MODEL_FIELDS,
    SESSION_UPDATE_TURN_DETECTOR_FIELDS,
    SESSION_UPDATE_TURN_PROFILE_FIELDS,
    SESSION_UPDATE_VAD_BACKEND_FIELDS,
    WIRE_BROWSER_EVENT,
    WIRE_CLIENT_EVENT,
    WIRE_RTC_CLIENT_DISCONNECTED,
    WIRE_RTC_SESSION_ATTACHED,
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationOrchestrator,
    ConvInterruptionDetectedEvent,
    ConvInterruptionFalsePositiveEvent,
    ConvResponseCreatedEvent,
    ConvSessionCreatedEvent,
    ConvTranscriptDeltaEvent,
    ConvTranscriptDoneEvent,
    ConvTurnEouPredictedEvent,
    browser_event_wire,
    client_disconnected_wire,
    client_event_command_from_parts,
    client_event_command_from_payload_json,
    client_event_payload,
    client_event_payload_json,
    control_event_as_client_event,
    control_event_client_payload_json,
    conversation_wire_event_payload,
    execute_conversation_command,
    execute_rtc_control_command,
    parse_conversation_wire_event,
    parse_session_update,
    pondsocket_event_to_conversation_command,
    response_command_payloads,
    response_text_fields,
    rtc_session_attached_payload,
    rtc_session_attached_wire,
    serialize_conversation_event,
    serialize_session_config,
    session_update_payload,
)
from vox.operations.errors import (
    InvalidConfigError,
    SessionAlreadyConfiguredError,
)


class ScriptedTTS(TTSAdapter):
    def __init__(self, chunks: int = 2) -> None:
        self._chunks = chunks
        self.texts: list[str] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted",
            type=ModelType.TTS,
            architectures=("scripted",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **kwargs):
        self.texts.append(text)
        for _ in range(self._chunks):
            yield SynthesizeChunk(
                audio=np.full(256, 0.02, dtype=np.float32).tobytes(),
                sample_rate=24_000,
                is_final=False,
            )
            await asyncio.sleep(0.005)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class DummyScheduler:
    def __init__(self, adapter: Any) -> None:
        self._a = adapter

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self._a


class CommandSpy:
    def __init__(self, *, configured: bool = True) -> None:
        self.config = object() if configured else None
        self.calls: list[tuple[str, tuple, dict]] = []

    async def start_session(self, config) -> None:
        self.config = config
        self.calls.append(("start_session", (config,), {}))

    async def ingest_pcm16(self, pcm16: bytes, sample_rate: int | None = None) -> None:
        self.calls.append(("ingest_pcm16", (pcm16,), {"sample_rate": sample_rate}))

    async def start_response(
        self,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
    ) -> None:
        kwargs: dict[str, object] = {"allow_interruptions": allow_interruptions}
        if generation_id is not None:
            kwargs["generation_id"] = generation_id
        self.calls.append(("start_response", (), kwargs))

    async def append_response_text(
        self,
        text: str,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
    ) -> None:
        kwargs: dict[str, object] = {"allow_interruptions": allow_interruptions}
        if generation_id is not None:
            kwargs["generation_id"] = generation_id
        self.calls.append(("append_response_text", (text,), kwargs))

    async def replace_response_text(self, text: str, *, allow_interruptions: bool = True) -> None:
        self.calls.append(("replace_response_text", (text,), {"allow_interruptions": allow_interruptions}))

    async def commit_response(self, *, generation_id: str | None = None) -> None:
        kwargs = {"generation_id": generation_id} if generation_id is not None else {}
        self.calls.append(("commit_response", (), kwargs))

    async def cancel_response(self, *, generation_id: str | None = None) -> None:
        kwargs = {"generation_id": generation_id} if generation_id is not None else {}
        self.calls.append(("cancel_response", (), kwargs))


def test_parse_session_update_requires_stt_model():
    with pytest.raises(InvalidConfigError):
        parse_session_update({"session": {"tts_model": "y:1"}})


def test_parse_session_update_requires_tts_model():
    with pytest.raises(InvalidConfigError):
        parse_session_update({"session": {"stt_model": "x:1"}})


def test_pondsocket_event_to_conversation_command_wraps_object_payload():
    assert pondsocket_event_to_conversation_command(
        "response.delta",
        {"delta": "hello", "allow_interruptions": False},
    ) == {
        "type": "response.delta",
        "delta": "hello",
        "allow_interruptions": False,
    }


def test_pondsocket_event_to_conversation_command_rejects_non_object_payload():
    with pytest.raises(InvalidConfigError, match="response.delta requires an object payload"):
        pondsocket_event_to_conversation_command("response.delta", "hello")


@pytest.mark.asyncio
async def test_execute_conversation_command_requires_message_type():
    spy = CommandSpy()

    with pytest.raises(InvalidConfigError, match="missing 'type' field"):
        await execute_conversation_command(spy, {})


@pytest.mark.asyncio
async def test_execute_conversation_command_requires_session_update_first():
    spy = CommandSpy(configured=False)

    with pytest.raises(InvalidConfigError, match="send session.update first"):
        await execute_conversation_command(spy, {"type": "response.start"})


@pytest.mark.asyncio
async def test_execute_conversation_command_starts_session_from_session_update():
    spy = CommandSpy(configured=False)

    await execute_conversation_command(
        spy,
        {"type": "session.update", "session": {"stt_model": "x:1", "tts_model": "y:1"}},
    )

    assert spy.calls[0][0] == "start_session"
    assert spy.config.stt_model == "x:1"
    assert spy.config.tts_model == "y:1"


@pytest.mark.asyncio
async def test_execute_conversation_command_appends_audio_and_response_text():
    spy = CommandSpy()

    await execute_conversation_command(
        spy,
        {
            "type": "input_audio_buffer.append",
            "audio": "AQIDBA==",
            "sample_rate": 16_000,
        },
    )
    await execute_conversation_command(
        spy,
        {
            "type": "response.delta",
            "response": {"delta": "hello", "allow_interruptions": False},
        },
    )

    assert spy.calls[0] == ("ingest_pcm16", (b"\x01\x02\x03\x04",), {"sample_rate": 16_000})
    assert spy.calls[1] == ("append_response_text", ("hello",), {"allow_interruptions": False})


@pytest.mark.asyncio
async def test_execute_conversation_command_forwards_response_generation_id():
    spy = CommandSpy()

    await execute_conversation_command(
        spy,
        {"type": "response.start", "generation_id": "generation-7"},
    )
    await execute_conversation_command(
        spy,
        {
            "type": "response.delta",
            "response": {"delta": "hello", "generationId": "generation-7"},
        },
    )
    await execute_conversation_command(
        spy,
        {"type": "response.commit", "generation_id": "generation-7"},
    )

    assert spy.calls == [
        (
            "start_response",
            (),
            {"allow_interruptions": True, "generation_id": "generation-7"},
        ),
        (
            "append_response_text",
            ("hello",),
            {"allow_interruptions": True, "generation_id": "generation-7"},
        ),
        ("commit_response", (), {"generation_id": "generation-7"}),
    ]


@pytest.mark.asyncio
async def test_execute_conversation_command_accepts_internal_raw_pcm_audio():
    spy = CommandSpy()

    await execute_conversation_command(
        spy,
        {
            "type": "input_audio_buffer.append",
            "audio_pcm16": b"\x01\x02\x03\x04",
            "sample_rate": 16_000,
        },
    )

    assert spy.calls == [("ingest_pcm16", (b"\x01\x02\x03\x04",), {"sample_rate": 16_000})]


@pytest.mark.asyncio
async def test_execute_conversation_command_rejects_empty_response_delta():
    spy = CommandSpy()

    with pytest.raises(InvalidConfigError, match="response.delta requires 'delta' text"):
        await execute_conversation_command(spy, {"type": "response.delta"})


@pytest.mark.asyncio
async def test_execute_conversation_command_replaces_response_text():
    spy = CommandSpy()

    await execute_conversation_command(
        spy,
        {"type": "response.replace_text", "text": "new text", "allow_interruptions": False},
    )

    assert spy.calls == [("replace_response_text", ("new text",), {"allow_interruptions": False})]


@pytest.mark.asyncio
async def test_response_command_compatibility_policy_is_named_and_ordered():
    spy = CommandSpy()

    message = {
        "type": "response.delta",
        "delta": "root fallback",
        "allow_interruptions": True,
        "response": {
            "text": "nested compatibility",
            "delta": "nested canonical",
            "allow_interruptions": False,
        },
    }

    await execute_conversation_command(spy, message)

    assert RESPONSE_COMMAND_ENVELOPE_FIELDS == ("response",)
    assert RESPONSE_TEXT_COMPATIBILITY_FIELDS == ("text", "delta")
    assert RESPONSE_ALLOW_INTERRUPTION_FIELD == "allow_interruptions"
    assert response_text_fields("delta") == ("delta", "text")
    assert response_command_payloads(message) == (message["response"], message)
    assert spy.calls == [("append_response_text", ("nested canonical",), {"allow_interruptions": False})]


@pytest.mark.asyncio
async def test_execute_conversation_command_rejects_unknown_type():
    spy = CommandSpy()

    with pytest.raises(InvalidConfigError, match="unknown message type: 'bogus'"):
        await execute_conversation_command(spy, {"type": "bogus"})


@pytest.mark.asyncio
async def test_execute_conversation_command_can_preserve_transport_unknown_label():
    spy = CommandSpy()

    with pytest.raises(InvalidConfigError, match="unknown conversation message type: 'bogus'"):
        await execute_conversation_command(
            spy,
            {"type": "bogus"},
            unknown_message_label="unknown conversation message type",
        )


@pytest.mark.asyncio
async def test_execute_conversation_command_dispatches_client_event_before_session_update():
    spy = CommandSpy(configured=False)
    received = []

    def on_client_event(event_name: str, payload) -> None:
        received.append((event_name, payload))

    await execute_conversation_command(
        spy,
        {"type": "client.event", "event": "render.url", "payload": {"url": "https://example.com"}},
        client_event_handler=on_client_event,
    )

    assert received == [("render.url", {"url": "https://example.com"})]
    assert spy.calls == []


@pytest.mark.asyncio
async def test_execute_conversation_command_rejects_invalid_client_event():
    spy = CommandSpy()

    with pytest.raises(InvalidConfigError, match="client.event requires a non-empty string 'event'"):
        await execute_conversation_command(
            spy,
            {"type": "client.event", "payload": {}},
            client_event_handler=lambda *_: None,
        )


@pytest.mark.asyncio
async def test_execute_conversation_command_can_disable_input_audio_for_control_only_transports():
    spy = CommandSpy()

    with pytest.raises(InvalidConfigError, match="unknown control message type"):
        await execute_conversation_command(
            spy,
            {"type": "input_audio_buffer.append", "audio": "AQIDBA=="},
            allow_input_audio=False,
            unknown_message_label="unknown control message type",
        )


@pytest.mark.asyncio
async def test_execute_rtc_control_command_disables_input_audio_by_policy():
    spy = CommandSpy()

    with pytest.raises(InvalidConfigError, match="unknown control message type"):
        await execute_rtc_control_command(
            spy,
            {"type": "input_audio_buffer.append", "audio": "AQIDBA=="},
            client_event_handler=lambda *_: None,
        )

    assert spy.calls == []


@pytest.mark.asyncio
async def test_execute_rtc_control_command_dispatches_client_event():
    spy = CommandSpy(configured=False)
    received = []

    def on_client_event(event_name: str, payload) -> None:
        received.append((event_name, payload))

    await execute_rtc_control_command(
        spy,
        {"type": "client.event", "event": "ui.toast", "payload": {"message": "hi"}},
        client_event_handler=on_client_event,
    )

    assert received == [("ui.toast", {"message": "hi"})]
    assert spy.calls == []


def test_client_event_payloads_are_operation_owned_contracts():
    assert client_event_command_from_parts(" ui.toast ", {"message": "hi"}) == {
        "type": "client.event",
        "event": "ui.toast",
        "payload": {"message": "hi"},
    }
    assert client_event_payload("ui.toast", {"message": "hi"}) == {
        "event": "ui.toast",
        "payload": {"message": "hi"},
    }
    assert client_event_payload_json("ui.toast", {"message": "hi"}) == (
        '{"event": "ui.toast", "payload": {"message": "hi"}}'
    )
    assert control_event_client_payload_json({"session_id": "rtc_1"}) == '{"session_id": "rtc_1"}'
    assert browser_event_wire("rtc_1", "mic.level", {"value": 0.5}) == {
        "type": WIRE_BROWSER_EVENT,
        "session_id": "rtc_1",
        "event": "mic.level",
        "payload": {"value": 0.5},
    }
    assert client_disconnected_wire(
        "rtc_1",
        reason="peer_connection_closed",
        connection_state="closed",
        ice_connection_state="closed",
        data_channel_state="closed",
    ) == {
        "type": WIRE_RTC_CLIENT_DISCONNECTED,
        "session_id": "rtc_1",
        "reason": "peer_connection_closed",
        "connection_state": "closed",
        "ice_connection_state": "closed",
        "data_channel_state": "closed",
    }


def test_client_event_command_rejects_empty_event_name():
    with pytest.raises(InvalidConfigError, match="non-empty string 'event'"):
        client_event_command_from_parts(" ", {"message": "hi"})


def test_client_event_command_from_payload_json_decodes_payload_at_operation_boundary():
    assert client_event_command_from_payload_json("app.marker", '{"n": 1}') == {
        "type": "client.event",
        "event": "app.marker",
        "payload": {"n": 1},
    }
    assert client_event_command_from_payload_json("app.marker", "") == {
        "type": "client.event",
        "event": "app.marker",
        "payload": None,
    }

    with pytest.raises(InvalidConfigError, match="requires valid payload JSON"):
        client_event_command_from_payload_json("app.marker", "{")


def test_control_event_as_client_event_preserves_explicit_client_event_contract():
    assert control_event_as_client_event(
        {"type": WIRE_CLIENT_EVENT, "event": "render.url", "payload": {"url": "https://example.com"}}
    ) == ("render.url", {"url": "https://example.com"})

    assert control_event_as_client_event(
        {"type": WIRE_RTC_CLIENT_DISCONNECTED, "session_id": "rtc_1", "reason": "closed"}
    ) == (
        WIRE_RTC_CLIENT_DISCONNECTED,
        {"session_id": "rtc_1", "reason": "closed"},
    )

    with pytest.raises(InvalidConfigError, match="control event requires a non-empty string 'type'"):
        control_event_as_client_event({"event": "missing.type"})


def test_conversation_wire_event_payload_is_operation_owned_transport_contract():
    assert conversation_wire_event_payload(
        {"type": "response.created", "response_id": "resp_1", "session_id": "rtc_1"}
    ) == (
        "response.created",
        {"response_id": "resp_1", "session_id": "rtc_1"},
    )

    with pytest.raises(
        InvalidConfigError,
        match="conversation wire event requires a non-empty string 'type'",
    ):
        conversation_wire_event_payload({"response_id": "resp_1"})


def test_parse_session_update_accepts_turn_policy_overrides():
    config = parse_session_update(
        {
            "session": {
                "stt_model": "x:1",
                "tts_model": "y:1",
                "turn_policy": {
                    "min_interrupt_duration_ms": 150,
                    "stable_speaking_min_ms": 100,
                    "speaking_interrupt_min_duration_ms": 650,
                    "speaking_interrupt_min_words": 3,
                    "self_echo_min_words": 4,
                    "self_echo_min_overlap": 0.8,
                },
            },
        }
    )
    assert config.policy is not None
    assert config.policy.min_interrupt_duration_ms == 150
    assert config.policy.stable_speaking_min_ms == 100
    assert config.policy.speaking_interrupt_min_duration_ms == 650
    assert config.policy.speaking_interrupt_min_words == 3
    assert config.policy.self_echo_min_words == 4
    assert config.policy.self_echo_min_overlap == pytest.approx(0.8)


def test_parse_session_update_applies_turn_profile_defaults():
    config = parse_session_update(
        {
            "session": {
                "stt_model": "x:1",
                "tts_model": "y:1",
                "turn_profile": "headset",
            },
        }
    )
    assert config.turn_profile == "headset"
    assert config.policy is not None
    assert config.policy.min_interrupt_duration_ms == 180
    assert config.policy.speaking_interrupt_min_words == 1
    assert config.policy.aec_warmup_ms == 250
    assert config.policy.vad_min_silence_ms == 600


def test_parse_session_update_allows_vad_min_silence_override():
    config = parse_session_update(
        {
            "session": {
                "stt_model": "x:1",
                "tts_model": "y:1",
                "turn_policy": {"vad_min_silence_ms": 550},
            },
        }
    )
    assert config.policy is not None
    assert config.policy.vad_min_silence_ms == 550


def test_parse_session_update_allows_profile_with_explicit_overrides():
    config = parse_session_update(
        {
            "session": {
                "stt_model": "x:1",
                "tts_model": "y:1",
                "turn_profile": "speakerphone",
                "turn_policy": {
                    "speaking_interrupt_min_words": 4,
                    "aec_warmup_ms": 600,
                },
            },
        }
    )
    assert config.turn_profile == "speakerphone"
    assert config.policy is not None
    assert config.policy.speaking_interrupt_min_words == 4
    assert config.policy.aec_warmup_ms == 600
    assert config.policy.backchannel_end_cooldown_ms == 1800


def test_parse_session_update_accepts_noisy_alias():
    config = parse_session_update(
        {
            "session": {
                "stt_model": "x:1",
                "tts_model": "y:1",
                "turn_profile": "noisy",
            },
        }
    )
    assert config.turn_profile == "noisy_room"
    assert config.policy.vad_min_silence_ms == 1200


def test_parse_session_update_rejects_unknown_turn_profile():
    with pytest.raises(InvalidConfigError):
        parse_session_update(
            {
                "session": {
                    "stt_model": "x:1",
                    "tts_model": "y:1",
                    "turn_profile": "spaceship",
                },
            }
        )


def test_parse_session_update_compatibility_fields_are_explicit_policy():
    assert SESSION_UPDATE_STT_MODEL_FIELDS == (
        "stt_model",
        "input_audio_transcription.model",
    )
    assert SESSION_UPDATE_TTS_MODEL_FIELDS == (
        "tts_model",
        "output_audio_generation.model",
    )
    assert SESSION_UPDATE_TURN_PROFILE_FIELDS == ("turn_profile", "profile")
    assert SESSION_UPDATE_VAD_BACKEND_FIELDS == ("vad_backend", "vad")
    assert SESSION_UPDATE_TURN_DETECTOR_FIELDS == ("turn_detector", "eou_model")
    assert SESSION_UPDATE_POLICY_FIELDS == ("turn_policy", "policy")

    config = parse_session_update(
        {
            "session": {
                "input_audio_transcription": {"model": "legacy-stt:1"},
                "output_audio_generation": {"model": "legacy-tts:1"},
                "profile": "browser",
                "vad": "silero",
                "eou_model": "livekit",
            },
        }
    )

    assert config.stt_model == "legacy-stt:1"
    assert config.tts_model == "legacy-tts:1"
    assert config.turn_profile == "browser_default"
    assert config.vad_backend == "silero"
    assert config.turn_detector == "livekit"


def test_session_update_payload_accepts_root_or_session_envelope_explicitly():
    root = {"stt_model": "x:1", "tts_model": "y:1"}
    wrapped = {"session": root}

    assert session_update_payload(root) is root
    assert session_update_payload(wrapped) is root
    assert session_update_payload({"session": None, **root})["stt_model"] == "x:1"


def test_parse_session_update_accepts_policy_alias_with_canonical_precedence():
    canonical = parse_session_update(
        {
            "session": {
                "stt_model": "x:1",
                "tts_model": "y:1",
                "turn_policy": {"vad_min_silence_ms": 550},
                "policy": {"vad_min_silence_ms": 1200},
            },
        }
    )
    alias = parse_session_update(
        {
            "session": {
                "stt_model": "x:1",
                "tts_model": "y:1",
                "policy": {"vad_min_silence_ms": 1200},
            },
        }
    )

    assert canonical.policy is not None
    assert canonical.policy.vad_min_silence_ms == 550
    assert alias.policy is not None
    assert alias.policy.vad_min_silence_ms == 1200


def test_parse_session_update_prefers_canonical_fields_over_compatibility_aliases():
    config = parse_session_update(
        {
            "session": {
                "stt_model": "canonical-stt:1",
                "input_audio_transcription": {"model": "legacy-stt:1"},
                "tts_model": "canonical-tts:1",
                "output_audio_generation": {"model": "legacy-tts:1"},
                "turn_profile": "headset",
                "profile": "browser",
                "vad_backend": "silero",
                "vad": "other-vad",
                "turn_detector": "livekit",
                "eou_model": "other-eou",
            },
        }
    )

    assert config.stt_model == "canonical-stt:1"
    assert config.tts_model == "canonical-tts:1"
    assert config.turn_profile == "headset"
    assert config.vad_backend == "silero"
    assert config.turn_detector == "livekit"


def test_serialize_session_config_round_trip_includes_policy_and_audio_format():
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1", "sample_rate": 48_000},
        }
    )
    payload = serialize_session_config(config)
    assert payload["stt_model"] == "x:1"
    assert payload["tts_model"] == "y:1"
    assert payload["output_audio_format"] == "pcm16"
    assert payload["output_sample_rate"] == 48_000
    assert payload["turn_profile"] == "default"
    assert payload["turn_policy"]["min_interrupt_duration_ms"] > 0
    assert payload["turn_policy"]["speaking_interrupt_min_duration_ms"] == 500
    assert payload["turn_policy"]["speaking_interrupt_min_words"] == 2
    assert payload["turn_policy"]["self_echo_min_words"] == 3
    assert payload["turn_policy"]["self_echo_min_overlap"] == pytest.approx(0.7)
    assert payload["turn_policy"]["aec_warmup_ms"] == 750
    assert payload["turn_policy"]["backchannel_end_cooldown_ms"] == 1500


def test_serialize_conversation_event_preserves_transcript_metadata():
    event = ConvTranscriptDoneEvent(
        transcript="hello",
        language="en",
        start_ms=100,
        end_ms=900,
        eou_probability=0.73,
        entities=({"text": "hello", "label": "greeting"},),
        topics=("greeting",),
        words=({"word": "hello", "start": 0.1, "end": 0.9},),
    )

    assert serialize_conversation_event(event) == {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "hello",
        "language": "en",
        "start_ms": 100,
        "end_ms": 900,
        "eou_probability": 0.73,
        "entities": [{"text": "hello", "label": "greeting"}],
        "topics": ["greeting"],
        "words": [{"word": "hello", "start": 0.1, "end": 0.9}],
    }


def test_serialize_conversation_event_preserves_response_audio_contract():
    assert serialize_conversation_event(ConvResponseCreatedEvent(response_id="resp_1")) == {
        "type": "response.created",
        "response_id": "resp_1",
    }
    assert serialize_conversation_event(
        ConvAudioDeltaEvent(
            audio_b64="AAAA",
            sample_rate=24_000,
            audio_format="pcm16",
            response_id="resp_1",
            sequence=3,
        )
    ) == {
        "type": "response.audio.delta",
        "audio": "AAAA",
        "sample_rate": 24_000,
        "audio_format": "pcm16",
        "response_id": "resp_1",
        "sequence": 3,
    }


def test_serialize_conversation_event_uses_operation_wire_error_constant():
    assert serialize_conversation_event(ConvErrorEvent(message="boom")) == {
        "type": "error",
        "message": "boom",
    }


def test_rtc_session_attached_contract_is_operation_owned():
    assert rtc_session_attached_payload("rtc_123") == {"session_id": "rtc_123"}
    assert rtc_session_attached_wire("rtc_123") == {
        "type": WIRE_RTC_SESSION_ATTACHED,
        "session_id": "rtc_123",
    }


def test_audio_clear_wire_event_maps_to_operation_event():
    event = parse_conversation_wire_event({"type": "response.audio.clear", "response_id": "resp_1"})
    assert isinstance(event, ConvAudioClearEvent)
    assert event.response_id == "resp_1"


def test_interruption_detected_wire_event_maps_to_operation_event():
    event = parse_conversation_wire_event(
        {
            "type": "interruption.detected",
            "response_id": "resp_1",
            "vad_active_ms": 320,
            "partial_transcript": "wait",
            "reason": "partial_keyword",
        }
    )
    assert isinstance(event, ConvInterruptionDetectedEvent)
    assert event.response_id == "resp_1"
    assert event.reason == "partial_keyword"


def test_transcript_delta_wire_event_maps_to_operation_event():
    event = parse_conversation_wire_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": "hello there",
            "start_ms": 100,
            "end_ms": 700,
        }
    )
    assert isinstance(event, ConvTranscriptDeltaEvent)
    assert event.delta == "hello there"
    assert event.start_ms == 100
    assert event.end_ms == 700


def test_interruption_false_positive_preserves_reason():
    event = parse_conversation_wire_event(
        {
            "type": "interruption.false_positive",
            "response_id": "resp_1",
            "vad_active_ms": 120,
            "partial_transcript": "mhmm",
            "reason": "backchannel",
        }
    )
    assert isinstance(event, ConvInterruptionFalsePositiveEvent)
    assert event.reason == "backchannel"


def test_turn_eou_predicted_wire_event_maps_to_operation_event():
    event = parse_conversation_wire_event(
        {
            "type": "turn.eou.predicted",
            "probability": 0.82,
            "threshold": 0.5,
            "decision": "complete",
            "action": "commit",
            "delay_ms": 0,
            "turn_detector": "livekit",
            "start_ms": 10,
            "end_ms": 420,
        }
    )

    assert isinstance(event, ConvTurnEouPredictedEvent)
    assert event.probability == pytest.approx(0.82)
    assert event.decision == "complete"
    assert event.action == "commit"
    assert event.turn_detector == "livekit"
    assert event.delay_ms == 0
    assert event.start_ms == 10
    assert event.end_ms == 420


@pytest.mark.asyncio
async def test_start_session_emits_session_created_event():
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(ScriptedTTS()))
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
        }
    )
    await orchestrator.start_session(config)
    await orchestrator.end_of_stream()
    events: list = []
    async for event in orchestrator.events():
        events.append(event)
    assert any(isinstance(e, ConvSessionCreatedEvent) for e in events)
    assert isinstance(events[-1], ConvDoneEvent)
    await orchestrator.close()


@pytest.mark.asyncio
async def test_double_start_raises_session_already_configured():
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(ScriptedTTS()))
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1"},
        }
    )
    await orchestrator.start_session(config)
    with pytest.raises(SessionAlreadyConfiguredError):
        await orchestrator.start_session(config)
    await orchestrator.close()


@pytest.mark.asyncio
async def test_streaming_response_emits_audio_and_done_events():
    adapter = ScriptedTTS(chunks=2)
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(adapter))
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
        }
    )
    await orchestrator.start_session(config)
    await orchestrator.start_response()
    await orchestrator.append_response_text("hi there")
    await orchestrator.commit_response()
    await orchestrator.end_of_stream()

    events: list = []
    async for event in orchestrator.events():
        events.append(event)
    types = {type(e).__name__ for e in events}
    assert "ConvSessionCreatedEvent" in types
    assert "ConvResponseCreatedEvent" in types
    assert "ConvResponseCommittedEvent" in types
    assert "ConvAudioDeltaEvent" in types
    assert "ConvResponseDoneEvent" in types
    audio_deltas = [e for e in events if isinstance(e, ConvAudioDeltaEvent)]
    assert audio_deltas
    assert audio_deltas[0].audio_format == "pcm16"
    assert audio_deltas[0].response_id
    assert audio_deltas[0].sequence > 0
    await orchestrator.close()


@pytest.mark.asyncio
async def test_response_delta_requires_an_explicit_active_response_generation():
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(ScriptedTTS()))
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
        }
    )
    await orchestrator.start_session(config)

    with pytest.raises(InvalidConfigError, match="response.start required"):
        await orchestrator.append_response_text("stale delta")

    await orchestrator.close()


@pytest.mark.asyncio
async def test_cancelled_response_generation_rejects_late_delta_and_commit():
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(ScriptedTTS(chunks=5)))
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
        }
    )
    await orchestrator.start_session(config)
    await orchestrator.start_response()
    await orchestrator.cancel_response()

    with pytest.raises(InvalidConfigError, match="response.start required"):
        await orchestrator.append_response_text("late delta")
    with pytest.raises(InvalidConfigError, match="response.start required"):
        await orchestrator.commit_response()

    await orchestrator.close()


@pytest.mark.asyncio
async def test_stale_wire_generation_cannot_write_into_a_new_response():
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(ScriptedTTS(chunks=5)))
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
        }
    )
    await orchestrator.start_session(config)
    await orchestrator.start_response(generation_id="generation-a")
    await orchestrator.cancel_response(generation_id="generation-a")
    await orchestrator.start_response(generation_id="generation-b")

    with pytest.raises(InvalidConfigError, match="stale response generation"):
        await orchestrator.append_response_text(
            "late delta",
            generation_id="generation-a",
        )
    await orchestrator.append_response_text(
        "current delta",
        generation_id="generation-b",
    )
    await orchestrator.commit_response(generation_id="generation-b")

    await orchestrator.close()


@pytest.mark.asyncio
async def test_end_of_stream_can_skip_pending_response_flush_for_rtc_shutdown():
    adapter = ScriptedTTS(chunks=1)
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(adapter))
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
        }
    )
    await orchestrator.start_session(config)
    await orchestrator.start_response()
    await orchestrator.append_response_text("do not synthesize yet")

    await orchestrator.end_of_stream(flush_response=False)

    events: list = []
    async for event in orchestrator.events():
        events.append(event)
    assert isinstance(events[-1], ConvDoneEvent)
    assert adapter.texts == []
    await orchestrator.close()


@pytest.mark.asyncio
async def test_report_error_emits_error_event():
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(ScriptedTTS()))
    await orchestrator.report_error("boom")
    await orchestrator.end_of_stream()
    events: list = []
    async for event in orchestrator.events():
        events.append(event)
    assert any(isinstance(e, ConvErrorEvent) and e.message == "boom" for e in events)
    await orchestrator.close()
