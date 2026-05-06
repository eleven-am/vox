from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

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
from vox.operations.errors import (
    InvalidConfigError,
    NoDefaultModelError,
    UnsupportedFormatError,
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
    normalize_longform_tts_config,
)


class FakeStreamingTTSAdapter(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts", type=ModelType.TTS,
            architectures=("fake",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            supports_streaming=True,
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default", language="en")]

    async def synthesize(self, text, **kwargs):
        yield SynthesizeChunk(audio=np.full(2400, 0.1, dtype=np.float32).tobytes(), sample_rate=24_000, is_final=False)
        yield SynthesizeChunk(audio=np.full(2400, 0.2, dtype=np.float32).tobytes(), sample_rate=24_000, is_final=True)


class FakeScheduler:
    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self._adapter


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
