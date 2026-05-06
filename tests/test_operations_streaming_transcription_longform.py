from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from vox.core.adapter import STTAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    TranscribeResult,
    TranscriptSegment,
)
from vox.operations.errors import (
    InvalidConfigError,
    NoDefaultModelError,
    UnsupportedFormatError,
    WrongModelTypeError,
)
from vox.operations.streaming_transcription_longform import (
    LongformDoneEvent,
    LongformErrorEvent,
    LongformProgressEvent,
    LongformReadyEvent,
    LongformTranscriptionSession,
    normalize_longform_config,
)


class FakeChunkingSTTAdapter(STTAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt", type=ModelType.STT,
            architectures=("fake",), default_sample_rate=16_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        self.calls += 1
        return TranscribeResult(
            text=f"chunk {self.calls}", language="en",
            duration_ms=int(len(audio) / 16_000 * 1000),
            segments=(TranscriptSegment(
                text=f"chunk {self.calls}", start_ms=0,
                end_ms=int(len(audio) / 16_000 * 1000),
            ),),
        )


class FakeScheduler:
    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self._adapter


class _StoreModel:
    def __init__(self, full_name: str, mtype: str) -> None:
        self.full_name = full_name
        self.type = type("T", (), {"value": mtype})()


def _make_registry() -> Any:
    registry = MagicMock()
    registry.available_models.return_value = {}
    return registry


def _make_store(stt: str | None = None) -> Any:
    store = MagicMock()
    store.list_models.return_value = [_StoreModel(stt, "stt")] if stt else []
    return store


async def _drain_events(session: LongformTranscriptionSession, *, timeout: float = 2.0) -> list:
    events: list = []

    async def collect() -> None:
        async for event in session.events():
            events.append(event)

    await asyncio.wait_for(collect(), timeout=timeout)
    return events


def test_normalize_rejects_unsupported_input_format():
    with pytest.raises(UnsupportedFormatError):
        normalize_longform_config(
            model="m:1", sample_rate=16_000, input_format="m4a",
            language=None, word_timestamps=False, temperature=0.0,
            chunk_ms=None, overlap_ms=None,
            registry=_make_registry(), store=_make_store(),
        )


def test_normalize_rejects_overlap_larger_than_chunk():
    with pytest.raises(InvalidConfigError):
        normalize_longform_config(
            model="m:1", sample_rate=16_000, input_format="pcm16",
            language=None, word_timestamps=False, temperature=0.0,
            chunk_ms=2000, overlap_ms=2000,
            registry=_make_registry(), store=_make_store(),
        )


def test_normalize_requires_model_or_default():
    with pytest.raises(NoDefaultModelError):
        normalize_longform_config(
            model="", sample_rate=16_000, input_format="pcm16",
            language=None, word_timestamps=False, temperature=0.0,
            chunk_ms=None, overlap_ms=None,
            registry=_make_registry(), store=_make_store(),
        )


@pytest.mark.asyncio
async def test_pcm16_chunks_emit_progress_and_done():
    adapter = FakeChunkingSTTAdapter()
    session = LongformTranscriptionSession(
        scheduler=FakeScheduler(adapter),
        registry=_make_registry(), store=_make_store(stt="m:1"),
    )
    config = normalize_longform_config(
        model="m:1", sample_rate=16_000, input_format="pcm16",
        language=None, word_timestamps=False, temperature=0.0,
        chunk_ms=1_000, overlap_ms=200,
        registry=_make_registry(), store=_make_store(stt="m:1"),
    )

    await session.configure(config)
    await session.submit_chunk(np.zeros(16_000, dtype=np.int16).tobytes())
    await session.end_of_stream()
    events = await _drain_events(session)

    assert any(isinstance(e, LongformReadyEvent) for e in events)
    assert any(isinstance(e, LongformProgressEvent) for e in events)
    done = next(e for e in events if isinstance(e, LongformDoneEvent))
    assert done.text == "chunk 1"
    assert done.language == "en"
    assert len(done.segments) == 1
    await session.close()


@pytest.mark.asyncio
async def test_empty_stream_emits_error_event():
    adapter = FakeChunkingSTTAdapter()
    session = LongformTranscriptionSession(
        scheduler=FakeScheduler(adapter),
        registry=_make_registry(), store=_make_store(stt="m:1"),
    )
    config = normalize_longform_config(
        model="m:1", sample_rate=16_000, input_format="pcm16",
        language=None, word_timestamps=False, temperature=0.0,
        chunk_ms=1_000, overlap_ms=0,
        registry=_make_registry(), store=_make_store(stt="m:1"),
    )

    await session.configure(config)
    await session.end_of_stream()
    events = await _drain_events(session)

    assert any(isinstance(e, LongformErrorEvent) and "audio" in e.message.lower() for e in events)
    await session.close()


@pytest.mark.asyncio
async def test_configure_rejects_non_stt_model():
    class NotSTT:
        @asynccontextmanager
        async def acquire(self, _model: str):
            yield object()

    session = LongformTranscriptionSession(
        scheduler=NotSTT(),
        registry=_make_registry(), store=_make_store(stt="m:1"),
    )
    config = normalize_longform_config(
        model="m:1", sample_rate=16_000, input_format="pcm16",
        language=None, word_timestamps=False, temperature=0.0,
        chunk_ms=1_000, overlap_ms=0,
        registry=_make_registry(), store=_make_store(stt="m:1"),
    )
    with pytest.raises(WrongModelTypeError):
        await session.configure(config)
    await session.close()
