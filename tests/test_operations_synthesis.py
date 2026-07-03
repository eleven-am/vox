from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.fakes import FakeScheduler as DummyScheduler
from tests.fakes import FakeSTTAdapter as FakeSTT
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
    InvalidConfigError,
    NoAudioGeneratedError,
    NoDefaultModelError,
    StoredModelNotFoundError,
    VoiceCloningUnsupportedOperationError,
    VoiceReferenceNotFoundError,
    WrongModelTypeError,
)
from vox.operations.synthesis import (
    SynthesisRequest,
    synthesis_request_from_fields,
    synthesize_audio_response,
    synthesize_full,
    synthesize_incremental,
    synthesize_raw,
    synthesize_stream,
)


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


class MultiChunkTTS(FakeTTS):
    def __init__(self, chunks: int = 3):
        super().__init__()
        self.chunks = chunks

    async def synthesize(self, text, **kwargs):
        self.last_kwargs = kwargs
        self.calls.append(text)
        for idx in range(self.chunks):
            yield SynthesizeChunk(
                audio=np.full(256, idx / 10, dtype=np.float32).tobytes(),
                sample_rate=24_000,
                is_final=False,
            )
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class ValidatingTTS(FakeTTS):
    def validate_synthesis_request(self, **kwargs):
        raise InvalidConfigError("invalid synthesis request")


def test_synthesis_request_from_fields_normalizes_transport_input():
    request = synthesis_request_from_fields(
        input="hello",
        model="fake-tts:latest",
        voice="",
        speed=0,
        language="",
        response_format="WAV",
    )

    assert request == SynthesisRequest(
        input="hello",
        model="fake-tts:latest",
        voice=None,
        speed=1.0,
        language=None,
        response_format="wav",
    )


def test_synthesis_request_from_fields_defaults_missing_response_format():
    request = synthesis_request_from_fields(
        input="hello",
        model="fake-tts:latest",
        response_format=None,
    )

    assert request.response_format == "wav"


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
async def test_synthesize_stream_multichunk_wav_has_single_container_header(tmp_path: Path):
    sched = DummyScheduler(MultiChunkTTS(chunks=3))
    store = BlobStore(root=tmp_path)
    registry = MagicMock()
    iterator = await synthesize_stream(
        scheduler=sched, registry=registry, store=store,
        request=SynthesisRequest(input="hello", model="fake-tts:latest", response_format="wav"),
    )
    body = b"".join([chunk async for chunk in iterator])
    assert body[:4] == b"RIFF"
    assert body.count(b"RIFF") == 1
    assert body.count(b"WAVE") == 1


@pytest.mark.asyncio
async def test_synthesize_audio_response_streamed_preflights_wrong_model_type(tmp_path: Path):
    sched = DummyScheduler(FakeSTT())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    with pytest.raises(WrongModelTypeError):
        await synthesize_audio_response(
            scheduler=sched,
            registry=registry,
            store=store,
            request=SynthesisRequest(input="hello", model="fake-stt:latest", response_format="wav"),
            stream=True,
        )


@pytest.mark.asyncio
async def test_synthesize_audio_response_non_streamed_wav_uses_incremental_metadata(tmp_path: Path):
    sched = DummyScheduler(MultiChunkTTS(chunks=2))
    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    response = await synthesize_audio_response(
        scheduler=sched,
        registry=registry,
        store=store,
        request=SynthesisRequest(input="hello", model="fake-tts:latest", response_format="wav"),
        stream=False,
    )
    body = b"".join([chunk async for chunk in response.chunks])

    assert response.content_type == "audio/wav"
    assert response.response_format == "wav"
    assert response.filename == "speech.wav"
    assert body[:4] == b"RIFF"
    assert body.count(b"RIFF") == 1


@pytest.mark.asyncio
async def test_synthesize_audio_response_non_incremental_format_returns_encoded_bundle(tmp_path: Path):
    sched = DummyScheduler(MultiChunkTTS(chunks=1))
    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    response = await synthesize_audio_response(
        scheduler=sched,
        registry=registry,
        store=store,
        request=SynthesisRequest(input="hello", model="fake-tts:latest", response_format="flac"),
        stream=False,
    )
    body = b"".join([chunk async for chunk in response.chunks])

    assert response.content_type == "audio/flac"
    assert response.filename == "speech.flac"
    assert body.startswith(b"fLaC")


@pytest.mark.asyncio
async def test_synthesize_audio_response_non_streamed_wav_preflights_before_returning(tmp_path: Path):
    sched = DummyScheduler(FakeSTT())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    with pytest.raises(WrongModelTypeError):
        await synthesize_audio_response(
            scheduler=sched,
            registry=registry,
            store=store,
            request=SynthesisRequest(input="hello", model="fake-stt:latest", response_format="wav"),
            stream=False,
        )


@pytest.mark.asyncio
async def test_synthesize_audio_response_streamed_runs_adapter_validation_before_returning(tmp_path: Path):
    sched = DummyScheduler(ValidatingTTS())
    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    with pytest.raises(InvalidConfigError, match="invalid synthesis request"):
        await synthesize_audio_response(
            scheduler=sched,
            registry=registry,
            store=store,
            request=SynthesisRequest(input="hello", model="fake-tts:latest", response_format="wav"),
            stream=True,
        )


@pytest.mark.asyncio
async def test_synthesize_incremental_wav_streams_single_header_and_pcm_chunks(tmp_path: Path):
    adapter = MultiChunkTTS(chunks=3)
    sched = DummyScheduler(adapter)
    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    iterator = await synthesize_incremental(
        scheduler=sched,
        registry=registry,
        store=store,
        request=SynthesisRequest(input="hello", model="fake-tts:latest", response_format="wav"),
    )
    chunks = [chunk async for chunk in iterator]

    assert len(chunks) == 4
    assert chunks[0][:4] == b"RIFF"
    assert chunks[0][8:12] == b"WAVE"
    assert chunks[0].count(b"RIFF") == 1
    assert all(chunk[:4] != b"RIFF" for chunk in chunks[1:])
    assert sum(len(chunk) for chunk in chunks[1:]) == 3 * 256 * 2


@pytest.mark.asyncio
async def test_synthesize_incremental_does_not_concatenate_chunks(tmp_path: Path, monkeypatch):
    adapter = MultiChunkTTS(chunks=2)
    sched = DummyScheduler(adapter)
    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    def _explode(*_args, **_kwargs):
        raise AssertionError("incremental synthesis must not concatenate all audio")

    monkeypatch.setattr(np, "concatenate", _explode)
    iterator = await synthesize_incremental(
        scheduler=sched,
        registry=registry,
        store=store,
        request=SynthesisRequest(input="hello", model="fake-tts:latest", response_format="pcm"),
    )
    chunks = [chunk async for chunk in iterator]

    assert len(chunks) == 2
    assert all(chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_synthesize_incremental_uses_adapter_text_chunking(tmp_path: Path):
    adapter = FakeTTS(max_input_chars=8)
    sched = DummyScheduler(adapter)
    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    iterator = await synthesize_incremental(
        scheduler=sched,
        registry=registry,
        store=store,
        request=SynthesisRequest(
            input="One. Two. Three.",
            model="fake-tts:latest",
            response_format="pcm",
        ),
    )
    chunks = [chunk async for chunk in iterator]

    assert chunks
    assert adapter.calls == ["One.", "Two.", "Three."]


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


@pytest.mark.asyncio
async def test_synthesize_full_translates_missing_stored_voice_reference_to_operation_error(tmp_path: Path):
    sched = DummyScheduler(FakeTTS(supports_voice_cloning=True))
    store = BlobStore(root=tmp_path)
    create_stored_voice(
        store,
        voice_id="voice1234",
        name="Roy",
        audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
        content_type="audio/wav",
    )
    (store.voices_dir / "voice1234" / "reference.wav").unlink()
    registry = MagicMock()

    with pytest.raises(VoiceReferenceNotFoundError):
        await synthesize_full(
            scheduler=sched,
            registry=registry,
            store=store,
            request=SynthesisRequest(input="hello", model="fake-tts:latest", voice="voice1234"),
        )


@pytest.mark.asyncio
async def test_synthesize_full_translates_clone_unsupported_to_operation_error(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    create_stored_voice(
        store,
        voice_id="voice1234",
        name="Roy",
        audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
        content_type="audio/wav",
    )
    sched = DummyScheduler(FakeTTS(supports_voice_cloning=False))
    registry = MagicMock()

    with pytest.raises(VoiceCloningUnsupportedOperationError):
        await synthesize_full(
            scheduler=sched,
            registry=registry,
            store=store,
            request=SynthesisRequest(input="hello", model="fake-tts:latest", voice="voice1234"),
        )


@pytest.mark.asyncio
async def test_synthesize_preflight_translates_model_not_found_to_operation_error(tmp_path: Path):
    from vox.core.errors import ModelNotFoundError

    class MissingModelScheduler:
        def acquire(self, model):
            raise ModelNotFoundError(model)

    store = BlobStore(root=tmp_path)
    registry = MagicMock()

    with pytest.raises(StoredModelNotFoundError):
        await synthesize_audio_response(
            scheduler=MissingModelScheduler(),
            registry=registry,
            store=store,
            request=SynthesisRequest(input="hello", model="missing:latest", response_format="wav"),
            stream=False,
        )
