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
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationOrchestrator,
    ConvInterruptionDetectedEvent,
    ConvSessionCreatedEvent,
    ConvTurnEouPredictedEvent,
    _wire_event_to_session_event,
    parse_session_update,
    serialize_session_config,
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
            name="scripted", type=ModelType.TTS,
            architectures=("scripted",), default_sample_rate=24_000,
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
                sample_rate=24_000, is_final=False,
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


def test_parse_session_update_accepts_turn_policy_overrides():
    config = parse_session_update({
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
    })
    assert config.policy is not None
    assert config.policy.min_interrupt_duration_ms == 150
    assert config.policy.stable_speaking_min_ms == 100
    assert config.policy.speaking_interrupt_min_duration_ms == 650
    assert config.policy.speaking_interrupt_min_words == 3
    assert config.policy.self_echo_min_words == 4
    assert config.policy.self_echo_min_overlap == pytest.approx(0.8)


def test_parse_session_update_applies_turn_profile_defaults():
    config = parse_session_update({
        "session": {
            "stt_model": "x:1",
            "tts_model": "y:1",
            "turn_profile": "headset",
        },
    })
    assert config.turn_profile == "headset"
    assert config.policy is not None
    assert config.policy.min_interrupt_duration_ms == 180
    assert config.policy.speaking_interrupt_min_words == 1
    assert config.policy.aec_warmup_ms == 250


def test_parse_session_update_allows_profile_with_explicit_overrides():
    config = parse_session_update({
        "session": {
            "stt_model": "x:1",
            "tts_model": "y:1",
            "turn_profile": "speakerphone",
            "turn_policy": {
                "speaking_interrupt_min_words": 4,
                "aec_warmup_ms": 600,
            },
        },
    })
    assert config.turn_profile == "speakerphone"
    assert config.policy is not None
    assert config.policy.speaking_interrupt_min_words == 4
    assert config.policy.aec_warmup_ms == 600
    assert config.policy.backchannel_end_cooldown_ms == 1800


def test_parse_session_update_rejects_unknown_turn_profile():
    with pytest.raises(InvalidConfigError):
        parse_session_update({
            "session": {
                "stt_model": "x:1",
                "tts_model": "y:1",
                "turn_profile": "spaceship",
            },
        })


def test_serialize_session_config_round_trip_includes_policy_and_audio_format():
    config = parse_session_update({
        "session": {"stt_model": "x:1", "tts_model": "y:1", "sample_rate": 48_000},
    })
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


def test_audio_clear_wire_event_maps_to_operation_event():
    event = _wire_event_to_session_event({"type": "response.audio.clear", "response_id": "resp_1"})
    assert isinstance(event, ConvAudioClearEvent)
    assert event.response_id == "resp_1"


def test_interruption_detected_wire_event_maps_to_operation_event():
    event = _wire_event_to_session_event({
        "type": "interruption.detected",
        "response_id": "resp_1",
        "vad_active_ms": 320,
        "partial_transcript": "wait",
    })
    assert isinstance(event, ConvInterruptionDetectedEvent)
    assert event.response_id == "resp_1"


def test_turn_eou_predicted_wire_event_maps_to_operation_event():
    event = _wire_event_to_session_event({
        "type": "turn.eou.predicted",
        "probability": 0.82,
        "threshold": 0.5,
        "decision": "complete",
        "action": "commit",
        "delay_ms": 0,
        "turn_detector": "livekit",
        "start_ms": 10,
        "end_ms": 420,
    })

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
    config = parse_session_update({
        "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
    })
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
    config = parse_session_update({
        "session": {"stt_model": "x:1", "tts_model": "y:1"},
    })
    await orchestrator.start_session(config)
    with pytest.raises(SessionAlreadyConfiguredError):
        await orchestrator.start_session(config)
    await orchestrator.close()


@pytest.mark.asyncio
async def test_streaming_response_emits_audio_and_done_events():
    adapter = ScriptedTTS(chunks=2)
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(adapter))
    config = parse_session_update({
        "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
    })
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
async def test_end_of_stream_can_skip_pending_response_flush_for_rtc_shutdown():
    adapter = ScriptedTTS(chunks=1)
    orchestrator = ConversationOrchestrator(scheduler=DummyScheduler(adapter))
    config = parse_session_update({
        "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
    })
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
