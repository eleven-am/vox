from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest

from vox.conversation.session import (
    ERROR_CODE_RESPONSE_ALREADY_ACTIVE,
    ERROR_CODE_RESPONSE_REJECTED_TURN_STATE,
    ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH,
    ERROR_CODE_RESPONSE_STALE_GENERATION,
    ERROR_CODE_SESSION_FAILED,
)
from vox.conversation.types import TurnEvent, TurnEventType
from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)
from vox.operations.conversation import (
    SESSION_UPDATE_POLICY_FIELDS,
    SESSION_UPDATE_STT_MODEL_FIELDS,
    SESSION_UPDATE_TTS_MODEL_FIELDS,
    SESSION_UPDATE_TURN_DETECTOR_FIELDS,
    SESSION_UPDATE_TURN_PROFILE_FIELDS,
    SESSION_UPDATE_VAD_BACKEND_FIELDS,
    WIRE_BROWSER_EVENT,
    WIRE_CLIENT_EVENT,
    WIRE_RTC_CLIENT_DISCONNECTED,
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationOrchestrator,
    ConvInterruptionDetectedEvent,
    ConvInterruptionFalsePositiveEvent,
    ConvResponseCancelledEvent,
    ConvResponseCommittedEvent,
    ConvResponseCreatedEvent,
    ConvResponseDoneEvent,
    ConvSessionCreatedEvent,
    ConvTranscriptDeltaEvent,
    ConvTranscriptDoneEvent,
    ConvTurnEouPredictedEvent,
    browser_event_wire,
    client_disconnected_wire,
    client_event_payload,
    client_event_payload_json,
    control_event_as_client_event,
    control_event_client_payload_json,
    conversation_wire_event_payload,
    parse_conversation_wire_event,
    parse_session_update,
    serialize_conversation_event,
    serialize_session_config,
    session_update_payload,
)
from vox.operations.errors import (
    ConversationCommandError,
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


def test_parse_session_update_requires_stt_model():
    with pytest.raises(InvalidConfigError):
        parse_session_update({"session": {"tts_model": "y:1"}})


def test_parse_session_update_requires_tts_model():
    with pytest.raises(InvalidConfigError):
        parse_session_update({"session": {"stt_model": "x:1"}})


def test_client_event_payloads_are_operation_owned_contracts():
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


def test_control_event_as_client_event_preserves_explicit_client_event_contract():
    assert control_event_as_client_event(
        {"type": WIRE_CLIENT_EVENT, "event": "render.url", "payload": {"url": "https://example.com"}}
    ) == ("render.url", {"url": "https://example.com"})
    assert control_event_as_client_event(
        {"type": WIRE_BROWSER_EVENT, "event": "ui.select", "payload": {"id": "choice-a"}}
    ) == ("ui.select", {"id": "choice-a"})

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
            audio=b"\x00\x00\x00",
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
        "code": "",
        "recoverable": True,
    }


def test_serialize_error_event_always_emits_code_and_recoverable():
    assert serialize_conversation_event(
        ConvErrorEvent(
            message="pipeline reset",
            code=ERROR_CODE_SESSION_FAILED,
            recoverable=False,
        )
    ) == {
        "type": "error",
        "message": "pipeline reset",
        "code": ERROR_CODE_SESSION_FAILED,
        "recoverable": False,
    }


def test_serialize_error_event_includes_generation_id_when_known():
    assert serialize_conversation_event(
        ConvErrorEvent(
            message="response rejected: user speech is active",
            code=ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH,
            recoverable=True,
            generation_id="gen-1",
        )
    ) == {
        "type": "error",
        "message": "response rejected: user speech is active",
        "code": ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH,
        "recoverable": True,
        "generation_id": "gen-1",
    }


def test_parse_error_wire_event_defaults_missing_code_to_recoverable():
    event = parse_conversation_wire_event({"type": "error", "message": "legacy boom"})
    assert isinstance(event, ConvErrorEvent)
    assert event.code == ""
    assert event.recoverable is True
    assert event.generation_id is None


def test_parse_error_wire_event_round_trips_typed_fields():
    event = parse_conversation_wire_event(
        {
            "type": "error",
            "message": "stale",
            "code": ERROR_CODE_RESPONSE_STALE_GENERATION,
            "recoverable": True,
            "generation_id": "gen-9",
        }
    )
    assert isinstance(event, ConvErrorEvent)
    assert event.code == ERROR_CODE_RESPONSE_STALE_GENERATION
    assert event.recoverable is True
    assert event.generation_id == "gen-9"


def test_serialize_lifecycle_events_carry_generation_id_only_when_known():
    assert serialize_conversation_event(
        ConvResponseCreatedEvent(response_id="resp_1", generation_id="gen-1")
    ) == {
        "type": "response.created",
        "response_id": "resp_1",
        "generation_id": "gen-1",
    }
    assert serialize_conversation_event(ConvResponseCreatedEvent(response_id="resp_1")) == {
        "type": "response.created",
        "response_id": "resp_1",
    }
    assert serialize_conversation_event(
        ConvAudioClearEvent(response_id="resp_1", generation_id="gen-1")
    ) == {
        "type": "response.audio.clear",
        "response_id": "resp_1",
        "generation_id": "gen-1",
    }
    detected = serialize_conversation_event(
        ConvInterruptionDetectedEvent(
            response_id="resp_1",
            vad_active_ms=200,
            partial_transcript="wait",
            reason="stable_partial",
            generation_id="gen-1",
        )
    )
    assert detected is not None
    assert detected["generation_id"] == "gen-1"


def test_parse_lifecycle_wire_events_round_trip_generation_id():
    created = parse_conversation_wire_event(
        {"type": "response.created", "response_id": "resp_1", "generation_id": "gen-1"}
    )
    assert isinstance(created, ConvResponseCreatedEvent)
    assert created.generation_id == "gen-1"

    legacy = parse_conversation_wire_event({"type": "response.created", "response_id": "resp_1"})
    assert isinstance(legacy, ConvResponseCreatedEvent)
    assert legacy.generation_id is None


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


async def _collect_orchestrator_events(orchestrator: ConversationOrchestrator) -> list:
    events: list = []
    async for event in orchestrator.events():
        events.append(event)
    return events


async def _started_orchestrator(adapter: Any) -> ConversationOrchestrator:
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(adapter))
    config = parse_session_update(
        {
            "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
        }
    )
    await orchestrator.start_session(config)
    return orchestrator


@pytest.mark.asyncio
async def test_response_lifecycle_events_echo_callers_generation_id():
    orchestrator = await _started_orchestrator(ScriptedTTS(chunks=2))
    await orchestrator.start_response(generation_id="gen-1")
    await orchestrator.append_response_text("hi there", generation_id="gen-1")
    await orchestrator.commit_response(generation_id="gen-1")
    await orchestrator.end_of_stream()

    events = await _collect_orchestrator_events(orchestrator)
    created = next(e for e in events if isinstance(e, ConvResponseCreatedEvent))
    committed = next(e for e in events if isinstance(e, ConvResponseCommittedEvent))
    done = next(e for e in events if isinstance(e, ConvResponseDoneEvent))
    assert created.generation_id == "gen-1"
    assert committed.generation_id == "gen-1"
    assert done.generation_id == "gen-1"
    assert created.response_id == committed.response_id == done.response_id
    await orchestrator.close()


@pytest.mark.asyncio
async def test_lifecycle_events_omit_generation_id_without_client_correlation():
    orchestrator = await _started_orchestrator(ScriptedTTS(chunks=2))
    await orchestrator.start_response()
    await orchestrator.append_response_text("hi there")
    await orchestrator.commit_response()
    await orchestrator.end_of_stream()

    events = await _collect_orchestrator_events(orchestrator)
    created = next(e for e in events if isinstance(e, ConvResponseCreatedEvent))
    done = next(e for e in events if isinstance(e, ConvResponseDoneEvent))
    assert created.generation_id is None
    assert done.generation_id is None
    wire = serialize_conversation_event(created)
    assert wire is not None
    assert "generation_id" not in wire
    await orchestrator.close()


@pytest.mark.asyncio
async def test_cancelled_response_event_carries_generation_id():
    orchestrator = await _started_orchestrator(ScriptedTTS(chunks=5))
    await orchestrator.start_response(generation_id="gen-1")
    await orchestrator.append_response_text("a long reply", generation_id="gen-1")
    await orchestrator.cancel_response(generation_id="gen-1")
    await orchestrator.end_of_stream(flush_response=False)

    events = await _collect_orchestrator_events(orchestrator)
    cancelled = [e for e in events if isinstance(e, ConvResponseCancelledEvent)]
    assert cancelled
    assert cancelled[0].generation_id == "gen-1"
    await orchestrator.close()


@pytest.mark.asyncio
async def test_rejected_start_error_event_carries_callers_generation_id():
    orchestrator = await _started_orchestrator(ScriptedTTS())
    session = orchestrator._session
    assert session is not None
    await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))

    with pytest.raises(ConversationCommandError) as exc_info:
        await orchestrator.start_response(generation_id="gen-rejected")

    assert exc_info.value.code == ERROR_CODE_RESPONSE_REJECTED_TURN_STATE
    assert exc_info.value.recoverable is True
    assert exc_info.value.generation_id == "gen-rejected"
    await orchestrator.close()


@pytest.mark.asyncio
async def test_start_while_active_raises_already_active_with_generation_id():
    orchestrator = await _started_orchestrator(ScriptedTTS(chunks=5))
    await orchestrator.start_response(generation_id="gen-1")

    with pytest.raises(ConversationCommandError, match="response already active") as exc_info:
        await orchestrator.start_response(generation_id="gen-2")

    assert exc_info.value.code == ERROR_CODE_RESPONSE_ALREADY_ACTIVE
    assert exc_info.value.recoverable is True
    assert exc_info.value.generation_id == "gen-2"
    await orchestrator.close()


@pytest.mark.asyncio
async def test_delta_and_commit_without_start_raise_stale_generation():
    orchestrator = await _started_orchestrator(ScriptedTTS())

    with pytest.raises(ConversationCommandError, match="response.start required") as delta_exc:
        await orchestrator.append_response_text("stale delta", generation_id="gen-a")
    with pytest.raises(ConversationCommandError, match="response.start required") as commit_exc:
        await orchestrator.commit_response(generation_id="gen-a")

    assert delta_exc.value.code == ERROR_CODE_RESPONSE_STALE_GENERATION
    assert delta_exc.value.generation_id == "gen-a"
    assert commit_exc.value.code == ERROR_CODE_RESPONSE_STALE_GENERATION
    assert commit_exc.value.generation_id == "gen-a"
    await orchestrator.close()


@pytest.mark.asyncio
async def test_stale_generation_delta_and_commit_raise_typed_errors():
    orchestrator = await _started_orchestrator(ScriptedTTS(chunks=5))
    await orchestrator.start_response(generation_id="gen-a")
    await orchestrator.cancel_response(generation_id="gen-a")
    await orchestrator.start_response(generation_id="gen-b")

    with pytest.raises(ConversationCommandError, match="stale response generation") as delta_exc:
        await orchestrator.append_response_text("late delta", generation_id="gen-a")
    with pytest.raises(ConversationCommandError, match="stale response generation") as commit_exc:
        await orchestrator.commit_response(generation_id="gen-a")

    assert delta_exc.value.code == ERROR_CODE_RESPONSE_STALE_GENERATION
    assert delta_exc.value.recoverable is True
    assert delta_exc.value.generation_id == "gen-a"
    assert commit_exc.value.code == ERROR_CODE_RESPONSE_STALE_GENERATION
    assert commit_exc.value.generation_id == "gen-a"
    await orchestrator.close()
