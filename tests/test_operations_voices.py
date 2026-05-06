from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

from vox.audio.codecs import encode_wav
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import create_stored_voice
from vox.core.store import BlobStore
from vox.core.types import (
    AdapterInfo,
    LoadedModelInfo,
    ModelFormat,
    ModelType,
    VoiceInfo,
)
from vox.operations.errors import (
    VoiceAudioRequiredError,
    VoiceIdRequiredError,
    VoiceNameRequiredError,
    VoiceNotFoundOperationError,
    WrongModelTypeError,
)
from vox.operations.voices import (
    CreateVoiceRequest,
    create_voice,
    delete_voice,
    get_voice_reference,
    list_voices,
)

from tests.fakes import FakeSTTAdapter as FakeSTT, FakeScheduler


def _wav() -> bytes:
    return encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000)


class FakeTTS(TTSAdapter):
    def __init__(self, cloning: bool = False):
        self._cloning = cloning

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts", type=ModelType.TTS,
            architectures=("fake",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            supports_voice_cloning=self._cloning,
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self):
        return [VoiceInfo(id="default", name="Default", language="en")]
    async def synthesize(self, text, **kw):
        if False:
            yield


class DummyScheduler(FakeScheduler):
    def __init__(self, adapter, loaded=None):
        super().__init__(adapter)
        self._loaded_list = loaded or []

    def list_loaded(self):
        return self._loaded_list


@pytest.mark.asyncio
async def test_list_voices_for_model(tmp_path: Path):
    sched = DummyScheduler(FakeTTS())
    store = BlobStore(root=tmp_path)
    listed = await list_voices(scheduler=sched, store=store, model="fake-tts:latest")
    assert len(listed) == 1
    assert listed[0].voice.id == "default"
    assert listed[0].model is None


@pytest.mark.asyncio
async def test_list_voices_for_loaded_models_no_filter(tmp_path: Path):
    loaded = LoadedModelInfo(name="fake-tts", tag="latest", type=ModelType.TTS, device="cpu")
    sched = DummyScheduler(FakeTTS(), loaded=[loaded])
    store = BlobStore(root=tmp_path)
    listed = await list_voices(scheduler=sched, store=store, model=None)
    assert len(listed) == 1
    assert listed[0].model == "fake-tts:latest"


@pytest.mark.asyncio
async def test_list_voices_for_model_includes_stored_clones_when_supported(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    create_stored_voice(store, voice_id="voice1234", name="Roy", audio_bytes=_wav(), content_type="audio/wav")
    sched = DummyScheduler(FakeTTS(cloning=True))
    listed = await list_voices(scheduler=sched, store=store, model="fake-tts:latest")
    assert any(v.voice.id == "voice1234" and v.voice.is_cloned for v in listed)


@pytest.mark.asyncio
async def test_list_voices_for_stt_raises_wrong_type(tmp_path: Path):
    sched = DummyScheduler(FakeSTT())
    store = BlobStore(root=tmp_path)
    with pytest.raises(WrongModelTypeError):
        await list_voices(scheduler=sched, store=store, model="fake-stt:latest")


def test_create_voice_persists_metadata_and_audio(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    voice = create_voice(
        store=store,
        request=CreateVoiceRequest(
            name="Roy", audio=_wav(), content_type="audio/wav",
            language="en", gender="male", reference_text="hello",
        ),
    )
    assert voice.name == "Roy"
    assert (store.voices_dir / voice.id / "reference.wav").is_file()


def test_create_voice_requires_name(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    with pytest.raises(VoiceNameRequiredError):
        create_voice(store=store, request=CreateVoiceRequest(name="", audio=_wav()))


def test_create_voice_requires_audio(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    with pytest.raises(VoiceAudioRequiredError):
        create_voice(store=store, request=CreateVoiceRequest(name="Roy", audio=b""))


def test_delete_voice_removes_directory(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    create_stored_voice(store, voice_id="v1", name="Roy", audio_bytes=_wav(), content_type="audio/wav")
    delete_voice(store=store, voice_id="v1")
    assert not (store.voices_dir / "v1").exists()


def test_delete_voice_unknown_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    with pytest.raises(VoiceNotFoundOperationError):
        delete_voice(store=store, voice_id="missing")


def test_delete_voice_empty_id_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    with pytest.raises(VoiceIdRequiredError):
        delete_voice(store=store, voice_id="")


def test_get_voice_reference_returns_bytes(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    create_stored_voice(store, voice_id="v1", name="Roy", audio_bytes=_wav(), content_type="audio/wav")
    data = get_voice_reference(store=store, voice_id="v1")
    assert data[:4] == b"RIFF"


def test_get_voice_reference_unknown_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    with pytest.raises(VoiceNotFoundOperationError):
        get_voice_reference(store=store, voice_id="missing")
