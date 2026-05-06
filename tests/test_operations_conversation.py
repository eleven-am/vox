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
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationOrchestrator,
    ConvSessionCreatedEvent,
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
            },
        },
    })
    assert config.policy is not None
    assert config.policy.min_interrupt_duration_ms == 150
    assert config.policy.stable_speaking_min_ms == 100


def test_serialize_session_config_round_trip_includes_policy_and_audio_format():
    config = parse_session_update({
        "session": {"stt_model": "x:1", "tts_model": "y:1", "sample_rate": 48_000},
    })
    payload = serialize_session_config(config)
    assert payload["stt_model"] == "x:1"
    assert payload["tts_model"] == "y:1"
    assert payload["output_audio_format"] == "pcm16"
    assert payload["output_sample_rate"] == 48_000
    assert payload["turn_policy"]["min_interrupt_duration_ms"] > 0


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
