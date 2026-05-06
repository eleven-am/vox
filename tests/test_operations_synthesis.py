from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vox.audio.codecs import encode_wav
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import create_stored_voice
from vox.core.store import BlobStore
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)
from vox.operations.errors import (
    EmptyInputError,
    NoAudioGeneratedError,
    NoDefaultModelError,
    WrongModelTypeError,
)
from vox.operations.synthesis import (
    SynthesisRequest,
    synthesize_full,
    synthesize_raw,
    synthesize_stream,
)

from tests.fakes import FakeSTTAdapter as FakeSTT, FakeScheduler as DummyScheduler


class FakeTTS(TTSAdapter):
    def __init__(self, max_input_chars: int = 0, supports_voice_cloning: bool = False):
        self._max = max_input_chars
        self._cloning = supports_voice_cloning
        self.last_kwargs: dict | None = None
        self.calls: list[str] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts", type=ModelType.TTS,
            architectures=("fake",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            supports_voice_cloning=self._cloning,
            max_input_chars=self._max,
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **kwargs):
        self.last_kwargs = kwargs
        self.calls.append(text)
        yield SynthesizeChunk(
            audio=np.full(2048, 0.0, dtype=np.float32).tobytes(),
            sample_rate=24_000, is_final=False,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


@pytest.mark.asyncio
async def test_synthesize_full_returns_wav_bytes(tmp_path: Path):
    sched = DummyScheduler(FakeTTS())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()
    bundle = await synthesize_full(
        scheduler=sched, registry=registry, store=store,
        request=SynthesisRequest(input="hello", model="fake-tts:latest", response_format="wav"),
    )
    assert bundle.audio[:4] == b"RIFF"
    assert bundle.content_type == "audio/wav"
    assert bundle.sample_rate == 24_000
    assert bundle.audio_ms > 0


@pytest.mark.asyncio
async def test_synthesize_full_raises_on_empty_input(tmp_path: Path):
    sched = DummyScheduler(FakeTTS())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()
    with pytest.raises(EmptyInputError):
        await synthesize_full(
            scheduler=sched, registry=registry, store=store,
            request=SynthesisRequest(input="", model="fake-tts:latest"),
        )


@pytest.mark.asyncio
async def test_synthesize_full_raises_when_no_default_model(tmp_path: Path):
    sched = DummyScheduler(FakeTTS())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()
    registry.available_models.return_value = {}
    with pytest.raises(NoDefaultModelError):
        await synthesize_full(
            scheduler=sched, registry=registry, store=store,
            request=SynthesisRequest(input="hello"),
        )


@pytest.mark.asyncio
async def test_synthesize_full_raises_on_wrong_adapter_type(tmp_path: Path):
    sched = DummyScheduler(FakeSTT())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()
    with pytest.raises(WrongModelTypeError):
        await synthesize_full(
            scheduler=sched, registry=registry, store=store,
            request=SynthesisRequest(input="hello", model="fake-stt:latest"),
        )


@pytest.mark.asyncio
async def test_synthesize_full_no_audio_generated_raises(tmp_path: Path):
    class _Empty(FakeTTS):
        async def synthesize(self, text, **kw):
            yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)

    sched = DummyScheduler(_Empty())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()
    with pytest.raises(NoAudioGeneratedError):
        await synthesize_full(
            scheduler=sched, registry=registry, store=store,
            request=SynthesisRequest(input="hello", model="fake-tts:latest"),
        )


@pytest.mark.asyncio
async def test_synthesize_stream_yields_encoded_chunks(tmp_path: Path):
    sched = DummyScheduler(FakeTTS())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()
    iterator = await synthesize_stream(
        scheduler=sched, registry=registry, store=store,
        request=SynthesisRequest(input="hello", model="fake-tts:latest", response_format="wav"),
    )
    chunks = [chunk async for chunk in iterator]
    assert len(chunks) >= 1
    assert chunks[0][:4] == b"RIFF"


@pytest.mark.asyncio
async def test_synthesize_raw_yields_pcm_chunks_with_final_marker(tmp_path: Path):
    adapter = FakeTTS(max_input_chars=8)
    sched = DummyScheduler(adapter)
    store = BlobStore(root=tmp_path)
    registry = MagicMock()
    iterator = await synthesize_raw(
        scheduler=sched, registry=registry, store=store,
        request=SynthesisRequest(input="One. Two. Three.", model="fake-tts:latest"),
    )
    chunks = [chunk async for chunk in iterator]
    assert len(adapter.calls) == 3
    finals = [c.is_final for c in chunks]
    assert finals[-1] is True
    assert all(not f for f in finals[:-1])


@pytest.mark.asyncio
async def test_synthesize_full_uses_stored_clone_reference(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    create_stored_voice(
        store, voice_id="voice1234", name="Roy",
        audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
        content_type="audio/wav", reference_text="hi there",
    )
    adapter = FakeTTS(supports_voice_cloning=True)
    sched = DummyScheduler(adapter)
    registry = MagicMock()
    await synthesize_full(
        scheduler=sched, registry=registry, store=store,
        request=SynthesisRequest(input="hello", model="fake-tts:latest", voice="voice1234"),
    )
    assert adapter.last_kwargs["voice"] is None
    assert adapter.last_kwargs["reference_audio"] is not None
    assert adapter.last_kwargs["reference_text"] == "hi there"
