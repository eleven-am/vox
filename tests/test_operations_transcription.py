from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from vox.audio.codecs import encode_wav
from vox.operations.errors import (
    EmptyAudioError,
    NoDefaultModelError,
    WrongModelTypeError,
)
from vox.operations.transcription import (
    AnnotateRequest,
    TranscriptionRequest,
    annotate_text,
    transcribe,
)

from tests.fakes import FakeSTTAdapter as FakeSTT, FakeTTSAdapter as FakeTTS, FakeScheduler as DummyScheduler


def _wav_bytes(dur_s: float = 1.0, sr: int = 16_000) -> bytes:
    audio = np.zeros(int(dur_s * sr), dtype=np.float32)
    return encode_wav(audio, sr)


@pytest.mark.asyncio
async def test_transcribe_returns_bundle_with_processing_ms():
    adapter = FakeSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(audio=_wav_bytes(), model="fake-stt:latest"),
    )
    assert bundle.result.text == "hello world"
    assert bundle.result.model == "fake-stt:latest"
    assert bundle.processing_ms >= 0


@pytest.mark.asyncio
async def test_transcribe_passes_kwargs_to_adapter():
    adapter = FakeSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(
            audio=_wav_bytes(),
            model="fake-stt:latest",
            language="fr",
            word_timestamps=True,
            temperature=0.7,
        ),
    )
    assert adapter.last_kwargs == {"language": "fr", "word_timestamps": True, "temperature": 0.7}


@pytest.mark.asyncio
async def test_transcribe_raises_on_empty_audio():
    sched = DummyScheduler(FakeSTT())
    registry = MagicMock()
    with pytest.raises(EmptyAudioError):
        await transcribe(
            scheduler=sched, registry=registry, store=None,
            request=TranscriptionRequest(audio=b"", model="fake-stt:latest"),
        )


@pytest.mark.asyncio
async def test_transcribe_raises_when_no_default_model():
    sched = DummyScheduler(FakeSTT())
    registry = MagicMock()
    registry.available_models.return_value = {}
    with pytest.raises(NoDefaultModelError):
        await transcribe(
            scheduler=sched, registry=registry, store=None,
            request=TranscriptionRequest(audio=_wav_bytes(), model=""),
        )


@pytest.mark.asyncio
async def test_transcribe_raises_when_adapter_is_tts():
    sched = DummyScheduler(FakeTTS())
    registry = MagicMock()
    with pytest.raises(WrongModelTypeError):
        await transcribe(
            scheduler=sched, registry=registry, store=None,
            request=TranscriptionRequest(audio=_wav_bytes(), model="fake-tts:latest"),
        )


@pytest.mark.asyncio
async def test_transcribe_resolves_default_from_registry():
    adapter = FakeSTT()
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    registry.available_models.return_value = {
        "fake-stt": {"latest": {"type": "stt"}},
    }
    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(audio=_wav_bytes()),
    )
    assert bundle.result.model == "fake-stt:latest"


@pytest.mark.asyncio
async def test_transcribe_annotate_text_when_requested():
    adapter = FakeSTT(text="Alice visited Paris", language="en")
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    bundle = await transcribe(
        scheduler=sched, registry=registry, store=None,
        request=TranscriptionRequest(
            audio=_wav_bytes(), model="fake-stt:latest", annotate_text=True,
        ),
    )
    assert isinstance(bundle.entities, tuple)
    assert isinstance(bundle.topics, tuple)


def test_annotate_text_returns_dataclass():
    result = annotate_text(AnnotateRequest(text="Alice visited Paris", language="en"))
    assert hasattr(result, "entities")
    assert hasattr(result, "topics")
