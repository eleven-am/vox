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
    NoDefaultModelError,
    SessionAlreadyConfiguredError,
    SessionNotConfiguredError,
)
from vox.operations.streaming_transcription import (
    DoneEvent,
    SessionReadyEvent,
    StreamingTranscriptionConfig,
    StreamingTranscriptionSession,
    TranscriptEvent,
)


class FakeSTTAdapter(STTAdapter):
    def __init__(self, text: str = "hello world") -> None:
        self._text = text
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
            text=self._text, language=kwargs.get("language") or "en",
            duration_ms=int(len(audio) / 16_000 * 1000),
            segments=(
                TranscriptSegment(text=self._text, start_ms=0, end_ms=200),
            ),
        )


class FakeScheduler:
    def __init__(self, adapter: STTAdapter) -> None:
        self._adapter = adapter

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self._adapter


class _StoreModel:
    def __init__(self, full_name: str, mtype: str) -> None:
        self.full_name = full_name
        self.type = type("T", (), {"value": mtype})()


def _make_registry(default_stt: str | None = None) -> Any:
    registry = MagicMock()
    if default_stt:
        name, tag = default_stt.split(":")
        registry.available_models.return_value = {name: {tag: {"type": "stt"}}}
    else:
        registry.available_models.return_value = {}
    return registry


def _make_store(stt: str | None = None) -> Any:
    store = MagicMock()
    store.list_models.return_value = [_StoreModel(stt, "stt")] if stt else []
    return store


async def _collect_events(
    session: StreamingTranscriptionSession,
    *,
    max_events: int = 30,
    timeout: float = 2.0,
) -> list:
    events = []

    async def collect() -> None:
        async for event in session.events():
            events.append(event)
            if len(events) >= max_events:
                return

    await asyncio.wait_for(collect(), timeout=timeout)
    return events


@pytest.mark.asyncio
async def test_configure_with_explicit_model_emits_session_ready():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )
    config = StreamingTranscriptionConfig(model="whisper:large-v3", language="fr")
    await session.configure(config)
    await session.end_of_stream()
    events = await _collect_events(session)

    ready = [e for e in events if isinstance(e, SessionReadyEvent)]
    assert len(ready) == 1
    assert ready[0].model == "whisper:large-v3"
    assert ready[0].language == "fr"
    assert any(isinstance(e, DoneEvent) for e in events)
    await session.close()


@pytest.mark.asyncio
async def test_configure_falls_back_to_default_model():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(default_stt="whisper:large-v3"),
        store=_make_store(),
    )
    await session.configure(StreamingTranscriptionConfig())
    await session.end_of_stream()
    events = await _collect_events(session)

    ready = next(e for e in events if isinstance(e, SessionReadyEvent))
    assert ready.model == "whisper:large-v3"
    await session.close()


@pytest.mark.asyncio
async def test_configure_without_default_raises_no_default_model():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )
    with pytest.raises(NoDefaultModelError):
        await session.configure(StreamingTranscriptionConfig())
    await session.close()


@pytest.mark.asyncio
async def test_double_configure_raises_session_already_configured():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )
    await session.configure(StreamingTranscriptionConfig(model="m:1"))
    with pytest.raises(SessionAlreadyConfiguredError):
        await session.configure(StreamingTranscriptionConfig(model="m:2"))
    await session.close()


@pytest.mark.asyncio
async def test_submit_pcm16_before_configure_raises():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )
    with pytest.raises(SessionNotConfiguredError):
        await session.submit_pcm16(b"\x00" * 100)
    await session.close()


@pytest.mark.asyncio
async def test_end_of_stream_flushes_remaining_audio_through_transcribe():
    adapter = FakeSTTAdapter(text="final transcript")
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(adapter),
        registry=_make_registry(default_stt="m:1"),
        store=_make_store(),
    )
    await session.configure(StreamingTranscriptionConfig(language="fr", include_word_timestamps=True))
    pcm16 = np.zeros(16_000, dtype=np.int16).tobytes()
    await session.submit_pcm16(pcm16)
    session._session.start_speech()
    session._session.append_audio(np.zeros(16_000, dtype=np.float32))
    await session.end_of_stream()
    events = await _collect_events(session)

    transcripts = [e for e in events if isinstance(e, TranscriptEvent)]
    assert any(t.transcript.text == "final transcript" for t in transcripts)
    assert adapter.calls >= 1
    await session.close()


@pytest.mark.asyncio
async def test_report_error_emits_error_event_in_order():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )
    await session.configure(StreamingTranscriptionConfig(model="m:1"))
    await session.report_error("boom")
    await session.end_of_stream()
    events = await _collect_events(session)

    types = [type(e).__name__ for e in events]
    assert types.index("SessionReadyEvent") < types.index("ErrorEvent") < types.index("DoneEvent")
    await session.close()


@pytest.mark.asyncio
async def test_events_iterator_terminates_on_done():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )
    await session.configure(StreamingTranscriptionConfig(model="m:1"))
    await session.end_of_stream()

    seen = []
    async for event in session.events():
        seen.append(event)
    assert isinstance(seen[-1], DoneEvent)
    await session.close()
