from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from vox.core.adapter import TTSAdapter
from vox.core.errors import ModelNotFoundError
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesisParameterInfo,
    SynthesizeChunk,
    VoiceInfo,
)
from vox.operations.errors import (
    InvalidConfigError,
    NoDefaultModelError,
    StoredModelNotFoundError,
    UnsupportedFormatError,
    VoiceCloningUnsupportedOperationError,
    VoiceReferenceNotFoundError,
    WrongModelTypeError,
)
from vox.operations.streaming_synthesis_longform import (
    LongformSynthesisSession,
    TtsAudioChunkEvent,
    TtsAudioStartEvent,
    TtsDoneEvent,
    TtsErrorEvent,
    TtsProgressEvent,
    TtsReadyEvent,
    longform_tts_event_payload,
    normalize_longform_tts_config,
)


class FakeStreamingTTSAdapter(TTSAdapter):
    def __init__(self, *, supports_voice_cloning: bool = False) -> None:
        self.supports_voice_cloning = supports_voice_cloning
        self.last_kwargs: dict[str, Any] | None = None

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts", type=ModelType.TTS,
            architectures=("fake",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            supports_streaming=True,
            supports_voice_cloning=self.supports_voice_cloning,
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default", language="en")]

    async def synthesize(self, text, **kwargs):
        self.last_kwargs = kwargs
        yield SynthesizeChunk(audio=np.full(2400, 0.1, dtype=np.float32).tobytes(), sample_rate=24_000, is_final=False)
        yield SynthesizeChunk(audio=np.full(2400, 0.2, dtype=np.float32).tobytes(), sample_rate=24_000, is_final=True)


class ParamStreamingTTSAdapter(FakeStreamingTTSAdapter):
    def synthesis_parameters(self):
        return (
            SynthesisParameterInfo(
                name="temperature",
                type="number",
                min_value=0.0,
                max_value=2.0,
            ),
        )


class FakeScheduler:
    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self._adapter


class MissingScheduler:
    def acquire(self, model: str):
        raise ModelNotFoundError(model)


def _make_registry() -> Any:
    registry = MagicMock()
    registry.available_models.return_value = {}
    return registry


class _StoreModel:
    def __init__(self, full_name: str, mtype: str) -> None:
        self.full_name = full_name
        self.type = type("T", (), {"value": mtype})()


def _make_store(tts: str | None = None) -> Any:
    store = MagicMock()
    store.list_models.return_value = [_StoreModel(tts, "tts")] if tts else []
    return store


def _make_voice_store(tmp_path: Path, *, tts: str = "t:1", voice_id: str = "clone") -> Any:
    store = _make_store(tts=tts)
    store.voices_dir = tmp_path
    voice_dir = tmp_path / voice_id
    voice_dir.mkdir(parents=True)
    (voice_dir / "metadata.json").write_text(
        json.dumps({
            "id": voice_id,
            "name": "Clone",
            "language": "en",
            "created_at": 1,
        }),
        encoding="utf-8",
    )
    return store


async def _drain_events(session: LongformSynthesisSession, *, timeout: float = 3.0) -> list:
    events: list = []

    async def collect() -> None:
        async for event in session.events():
            events.append(event)

    await asyncio.wait_for(collect(), timeout=timeout)
    return events


def test_normalize_rejects_unsupported_response_format():
    with pytest.raises(UnsupportedFormatError):
        normalize_longform_tts_config(
            model="t:1", voice=None, speed=1.0, language=None,
            response_format="wav", chunk_chars=None,
            registry=_make_registry(), store=_make_store(),
        )


def test_normalize_requires_model_or_default():
    with pytest.raises(NoDefaultModelError):
        normalize_longform_tts_config(
            model="", voice=None, speed=1.0, language=None,
            response_format="pcm16", chunk_chars=None,
            registry=_make_registry(), store=_make_store(),
        )


def test_normalize_rejects_invalid_chunk_chars():
    with pytest.raises(InvalidConfigError):
        normalize_longform_tts_config(
            model="t:1", voice=None, speed=1.0, language=None,
            response_format="pcm16", chunk_chars="not-an-int",
            registry=_make_registry(), store=_make_store(),
        )


def test_normalize_clamps_non_positive_speed():
    for bad_speed in (-2.0, 0.0):
        config = normalize_longform_tts_config(
            model="t:1", voice=None, speed=bad_speed, language=None,
            response_format="pcm16", chunk_chars=None,
            registry=_make_registry(), store=_make_store(),
        )
        assert config.speed == 1.0


def test_normalize_preserves_params():
    config = normalize_longform_tts_config(
        model="t:1", voice=None, speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        params={"temperature": 0.4},
        registry=_make_registry(), store=_make_store(),
    )

    assert config.params == {"temperature": 0.4}


def test_normalize_rejects_non_object_params():
    with pytest.raises(InvalidConfigError, match="params must be a JSON object"):
        normalize_longform_tts_config(
            model="t:1", voice=None, speed=1.0, language=None,
            response_format="pcm16", chunk_chars=None,
            params=["temperature"],
            registry=_make_registry(), store=_make_store(),
        )


def test_longform_tts_event_payloads_preserve_wire_contract():
    assert longform_tts_event_payload(
        TtsReadyEvent(
            model="t:1",
            voice="default",
            response_format="pcm16",
            chunk_chars=500,
        )
    ) == {
        "type": "ready",
        "model": "t:1",
        "voice": "default",
        "response_format": "pcm16",
        "chunk_chars": 500,
    }
    assert longform_tts_event_payload(
        TtsAudioStartEvent(sample_rate=24_000, response_format="pcm16")
    ) == {
        "type": "audio_start",
        "sample_rate": 24_000,
        "response_format": "pcm16",
    }
    assert longform_tts_event_payload(
        TtsProgressEvent(
            completed_chars=10,
            total_chars=20,
            chunks_completed=1,
            chunks_total=2,
        )
    ) == {
        "type": "progress",
        "completed_chars": 10,
        "total_chars": 20,
        "chunks_completed": 1,
        "chunks_total": 2,
    }
    assert longform_tts_event_payload(
        TtsDoneEvent(
            response_format="pcm16",
            audio_duration_ms=1000,
            processing_ms=50,
            text_length=20,
        )
    ) == {
        "type": "done",
        "response_format": "pcm16",
        "audio_duration_ms": 1000,
        "processing_ms": 50,
        "text_length": 20,
    }
    assert longform_tts_event_payload(TtsErrorEvent(message="boom")) == {
        "type": "error",
        "message": "boom",
    }
    assert longform_tts_event_payload(TtsAudioChunkEvent(data=b"audio")) is None


@pytest.mark.asyncio
async def test_text_synthesis_emits_ready_audio_done():
    session = LongformSynthesisSession(
        scheduler=FakeScheduler(FakeStreamingTTSAdapter()),
        registry=_make_registry(), store=_make_store(tts="t:1"),
    )
    config = normalize_longform_tts_config(
        model="t:1", voice="default", speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        registry=_make_registry(), store=_make_store(tts="t:1"),
    )
    await session.configure(config)
    session.append_text("Hello world. " * 4)
    await session.end_of_stream()
    events = await _drain_events(session)

    assert any(isinstance(e, TtsReadyEvent) for e in events)
    assert any(isinstance(e, TtsAudioStartEvent) for e in events)
    assert any(isinstance(e, TtsAudioChunkEvent) for e in events)
    assert any(isinstance(e, TtsProgressEvent) for e in events)
    assert any(isinstance(e, TtsDoneEvent) for e in events)
    await session.close()


@pytest.mark.asyncio
async def test_text_synthesis_passes_params_to_adapter():
    adapter = ParamStreamingTTSAdapter()
    session = LongformSynthesisSession(
        scheduler=FakeScheduler(adapter),
        registry=_make_registry(), store=_make_store(tts="t:1"),
    )
    config = normalize_longform_tts_config(
        model="t:1", voice="default", speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        registry=_make_registry(), store=_make_store(tts="t:1"),
        params={"temperature": 0.4},
    )
    await session.configure(config)
    session.append_text("Hello world.")
    await session.end_of_stream()
    await _drain_events(session)

    assert adapter.last_kwargs is not None
    assert adapter.last_kwargs["params"] == {"temperature": 0.4}
    await session.close()


@pytest.mark.asyncio
async def test_empty_text_emits_error_event():
    session = LongformSynthesisSession(
        scheduler=FakeScheduler(FakeStreamingTTSAdapter()),
        registry=_make_registry(), store=_make_store(tts="t:1"),
    )
    config = normalize_longform_tts_config(
        model="t:1", voice="default", speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        registry=_make_registry(), store=_make_store(tts="t:1"),
    )
    await session.configure(config)
    await session.end_of_stream()
    events = await _drain_events(session)

    assert any(isinstance(e, TtsErrorEvent) and "input" in e.message.lower() for e in events)
    await session.close()


@pytest.mark.asyncio
async def test_configure_rejects_non_tts_model():
    class NotTTS:
        @asynccontextmanager
        async def acquire(self, _model: str):
            yield object()

    session = LongformSynthesisSession(
        scheduler=NotTTS(),
        registry=_make_registry(), store=_make_store(tts="t:1"),
    )
    config = normalize_longform_tts_config(
        model="t:1", voice="default", speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        registry=_make_registry(), store=_make_store(tts="t:1"),
    )
    with pytest.raises(WrongModelTypeError):
        await session.configure(config)
    await session.close()


@pytest.mark.asyncio
async def test_configure_maps_missing_stored_model_to_operation_error():
    session = LongformSynthesisSession(
        scheduler=MissingScheduler(),
        registry=_make_registry(), store=_make_store(tts="missing:latest"),
    )
    config = normalize_longform_tts_config(
        model="missing:latest", voice="default", speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        registry=_make_registry(), store=_make_store(tts="missing:latest"),
    )

    with pytest.raises(StoredModelNotFoundError, match="missing:latest"):
        await session.configure(config)
    await session.close()


@pytest.mark.asyncio
async def test_configure_maps_cloned_voice_unsupported_to_operation_error(tmp_path: Path):
    store = _make_voice_store(tmp_path)
    session = LongformSynthesisSession(
        scheduler=FakeScheduler(FakeStreamingTTSAdapter()),
        registry=_make_registry(), store=store,
    )
    config = normalize_longform_tts_config(
        model="t:1", voice="clone", speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        registry=_make_registry(), store=store,
    )

    with pytest.raises(VoiceCloningUnsupportedOperationError):
        await session.configure(config)
    await session.close()


@pytest.mark.asyncio
async def test_configure_maps_missing_reference_audio_to_operation_error(tmp_path: Path):
    store = _make_voice_store(tmp_path)
    session = LongformSynthesisSession(
        scheduler=FakeScheduler(FakeStreamingTTSAdapter(supports_voice_cloning=True)),
        registry=_make_registry(), store=store,
    )
    config = normalize_longform_tts_config(
        model="t:1", voice="clone", speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        registry=_make_registry(), store=store,
    )

    with pytest.raises(VoiceReferenceNotFoundError):
        await session.configure(config)
    await session.close()


@pytest.mark.asyncio
async def test_configure_or_report_emits_terminal_error_event():
    session = LongformSynthesisSession(
        scheduler=MissingScheduler(),
        registry=_make_registry(), store=_make_store(tts="missing:latest"),
    )
    config = normalize_longform_tts_config(
        model="missing:latest", voice="default", speed=1.0, language=None,
        response_format="pcm16", chunk_chars=None,
        registry=_make_registry(), store=_make_store(tts="missing:latest"),
    )

    ok = await session.configure_or_report(config)
    events = await _drain_events(session)

    assert ok is False
    assert len(events) == 1
    assert isinstance(events[0], TtsErrorEvent)
    assert "missing:latest" in events[0].message
    await session.close()


@pytest.mark.asyncio
async def test_end_of_stream_or_report_emits_not_configured_error():
    session = LongformSynthesisSession(
        scheduler=FakeScheduler(FakeStreamingTTSAdapter()),
        registry=_make_registry(), store=_make_store(tts="t:1"),
    )

    ok = await session.end_of_stream_or_report()
    events = await _drain_events(session)

    assert ok is False
    assert len(events) == 1
    assert isinstance(events[0], TtsErrorEvent)
    assert events[0].message == "Session not configured"
    await session.close()
