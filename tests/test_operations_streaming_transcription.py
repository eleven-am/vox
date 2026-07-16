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
from vox.operations.streaming_reporting import streaming_operation_error_message
from vox.operations.streaming_transcription import (
    DoneEvent,
    ErrorEvent,
    SessionReadyEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    StreamingTranscriptionConfig,
    StreamingTranscriptionSession,
    TranscriptEvent,
    streaming_transcription_config_from_fields,
    streaming_transcription_event_payload,
)
from vox.streaming.types import SpeechStarted, StreamTranscript


class FakeSTTAdapter(STTAdapter):
    def __init__(self, text: str = "hello world") -> None:
        self._text = text
        self.calls = 0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt",
            type=ModelType.STT,
            architectures=("fake",),
            default_sample_rate=16_000,
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
            text=self._text,
            language=kwargs.get("language") or "en",
            duration_ms=int(len(audio) / 16_000 * 1000),
            segments=(TranscriptSegment(text=self._text, start_ms=0, end_ms=200),),
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


def test_streaming_operation_error_message_special_cases_missing_default_stt():
    assert (
        streaming_operation_error_message(NoDefaultModelError("stt"))
        == "No STT model specified and no default STT model available"
    )


def test_streaming_config_from_fields_preserves_transport_defaults():
    config = streaming_transcription_config_from_fields(
        model=None,
        language="",
        sample_rate=0,
        partials=True,
        partial_window_ms=0,
        partial_stride_ms=0,
        include_word_timestamps=True,
        temperature=None,
    )

    assert config == StreamingTranscriptionConfig(
        model="",
        language="en",
        sample_rate=16_000,
        partials=True,
        partial_window_ms=1500,
        partial_stride_ms=700,
        include_word_timestamps=True,
        temperature=0.0,
    )


def test_streaming_event_payload_preserves_realtime_wire_contract():
    transcript = StreamTranscript(
        text="hello",
        is_partial=False,
        start_ms=100,
        end_ms=400,
        audio_duration_ms=300,
        processing_duration_ms=20,
        model="m:1",
        eou_probability=0.8,
        entities=[{"type": "PERSON", "text": "Roy"}],
        topics=["topic"],
        words=[{"word": "hello", "start_ms": 100, "end_ms": 400}],
        segments=[{"text": "hello", "start_ms": 100, "end_ms": 400}],
    )

    assert streaming_transcription_event_payload(SessionReadyEvent("m:1", "en", 16_000)) == {
        "type": "ready",
    }
    assert streaming_transcription_event_payload(SpeechStartedEvent(timestamp_ms=10)) == {
        "type": "speech_started",
        "timestamp_ms": 10,
    }
    assert streaming_transcription_event_payload(SpeechStoppedEvent(timestamp_ms=20)) == {
        "type": "speech_stopped",
        "timestamp_ms": 20,
    }
    assert streaming_transcription_event_payload(TranscriptEvent(transcript=transcript)) == {
        "type": "transcript",
        "text": "hello",
        "is_partial": False,
        "start_ms": 100,
        "end_ms": 400,
        "audio_duration_ms": 300,
        "processing_duration_ms": 20,
        "model": "m:1",
        "eou_probability": 0.8,
        "entities": [{"type": "PERSON", "text": "Roy"}],
        "topics": ["topic"],
        "words": [{"word": "hello", "start_ms": 100, "end_ms": 400}],
        "segments": [{"text": "hello", "start_ms": 100, "end_ms": 400}],
    }
    assert streaming_transcription_event_payload(ErrorEvent(message="boom")) == {
        "type": "error",
        "message": "boom",
    }
    assert streaming_transcription_event_payload(DoneEvent()) == {"type": "done"}


@pytest.mark.asyncio
async def test_report_operation_error_uses_streaming_message_policy():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )

    await session.report_operation_error(NoDefaultModelError("stt"))
    await session.end_of_stream()
    events = await _collect_events(session)

    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert errors[0].message == "No STT model specified and no default STT model available"
    await session.close()


@pytest.mark.asyncio
async def test_configure_or_report_emits_in_band_operation_error():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )

    ok = await session.configure_or_report(StreamingTranscriptionConfig())
    await session.end_of_stream()
    events = await _collect_events(session)

    assert ok is False
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert errors[0].message == "No STT model specified and no default STT model available"
    assert any(isinstance(event, DoneEvent) for event in events)
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("submit_pcm16_or_report", (b"\x00" * 100,), {}),
        ("submit_opus_or_report", (b"\x00" * 12,), {"sample_rate": 48_000, "channels": 1}),
        ("submit_encoded_or_report", (b"not-audio",), {"format_hint": "wav"}),
    ],
)
async def test_audio_submit_or_report_emits_not_configured_errors(method_name, args, kwargs):
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(),
        store=_make_store(),
    )

    method = getattr(session, method_name)
    ok = await method(*args, **kwargs)
    await session.end_of_stream()
    events = await _collect_events(session)

    assert ok is False
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert errors[0].message == "Session not configured"
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
async def test_end_of_stream_does_not_retranscribe_completed_utterance():
    adapter = FakeSTTAdapter(text="already emitted")
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(adapter),
        registry=_make_registry(default_stt="m:1"),
        store=_make_store(),
    )
    await session.configure(StreamingTranscriptionConfig())
    session._session.start_speech()
    session._session.append_audio(np.zeros(16_000, dtype=np.float32))
    session._session.stop_speech()
    await session.end_of_stream()
    events = await _collect_events(session)

    transcripts = [e for e in events if isinstance(e, TranscriptEvent)]
    assert transcripts == []
    await session.close()


@pytest.mark.asyncio
async def test_speech_start_chunk_is_kept_in_active_buffer():
    session = StreamingTranscriptionSession(
        scheduler=FakeScheduler(FakeSTTAdapter()),
        registry=_make_registry(default_stt="m:1"),
        store=_make_store(),
    )
    await session.configure(StreamingTranscriptionConfig())

    async def fake_process_audio(audio):
        yield SpeechStarted(timestamp_ms=0)

    assert session._pipeline is not None
    session._pipeline.process_audio = fake_process_audio  # type: ignore[method-assign]

    pcm16 = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
    await session.submit_pcm16(pcm16, sample_rate=16_000)

    assert session._session.get_buffer_length() == 1600
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
