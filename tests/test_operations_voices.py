from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fakes import FakeScheduler
from tests.fakes import FakeSTTAdapter as FakeSTT
from vox.audio.codecs import encode_wav
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import create_stored_voice
from vox.core.errors import VoxError
from vox.core.store import BlobStore
from vox.core.types import (
    AdapterInfo,
    LoadedModelInfo,
    ModelFormat,
    ModelType,
    VoiceInfo,
)
from vox.operations.errors import (
    InternalOperationError,
    InvalidConfigError,
    StoredModelNotFoundError,
    VoiceAudioRequiredError,
    VoiceIdRequiredError,
    VoiceNameRequiredError,
    VoiceNotFoundOperationError,
    VoiceReferenceInvalidError,
    WrongModelTypeError,
)
from vox.operations.voices import (
    CreateVoiceRequest,
    ListedVoice,
    create_voice,
    created_voice_payload,
    delete_voice,
    deleted_voice_payload,
    get_voice_reference,
    list_voices,
    list_voices_payload,
    voice_payload,
)


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


class BrokenVoiceTTS(FakeTTS):
    def list_voices(self):
        raise VoxError("voice inventory failed")


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


@pytest.mark.asyncio
async def test_list_voices_translates_missing_model_to_operation_error(tmp_path: Path):
    sched = FakeScheduler()
    store = BlobStore(root=tmp_path)

    with pytest.raises(StoredModelNotFoundError):
        await list_voices(scheduler=sched, store=store, model="missing:latest")


@pytest.mark.asyncio
async def test_list_voices_translates_core_voice_failure_to_operation_error(tmp_path: Path):
    sched = DummyScheduler(BrokenVoiceTTS())
    store = BlobStore(root=tmp_path)

    with pytest.raises(InternalOperationError, match="voice inventory failed"):
        await list_voices(scheduler=sched, store=store, model="fake-tts:latest")


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


def test_create_voice_translates_invalid_reference_audio(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    short_wav = encode_wav(np.full(4_000, 0.1, dtype=np.float32), 16_000)

    with pytest.raises(VoiceReferenceInvalidError, match="too short"):
        create_voice(
            store=store,
            request=CreateVoiceRequest(
                name="Roy",
                audio=short_wav,
                content_type="audio/wav",
            ),
        )


def test_create_voice_translates_decode_failure(tmp_path: Path):
    store = BlobStore(root=tmp_path)

    with pytest.raises(InvalidConfigError):
        create_voice(
            store=store,
            request=CreateVoiceRequest(
                name="Roy",
                audio=b"not an audio file",
                content_type="audio/wav",
            ),
        )


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


def test_voice_payload_preserves_http_contract_shape():
    voice = VoiceInfo(
        id="af_heart",
        name="Heart",
        language="en",
        gender="female",
        description="Default voice",
        is_cloned=False,
    )

    assert voice_payload(voice) == {
        "id": "af_heart",
        "name": "Heart",
        "language": "en",
        "gender": "female",
        "description": "Default voice",
        "is_cloned": False,
    }


def test_list_voices_payload_can_include_model_for_unfiltered_route():
    listed = [
        ListedVoice(
            voice=VoiceInfo(id="default", name="Default", language="en"),
            model="kokoro-tts-onnx:v1.0",
        )
    ]

    assert list_voices_payload(listed, include_model=True) == {
        "voices": [
            {
                "model": "kokoro-tts-onnx:v1.0",
                "id": "default",
                "name": "Default",
                "language": "en",
                "gender": None,
                "description": None,
                "is_cloned": False,
            }
        ]
    }


def test_list_voices_payload_omits_model_for_model_filtered_route():
    listed = [
        ListedVoice(
            voice=VoiceInfo(id="default", name="Default", language="en"),
            model=None,
        )
    ]

    assert list_voices_payload(listed, include_model=False) == {
        "voices": [
            {
                "id": "default",
                "name": "Default",
                "language": "en",
                "gender": None,
                "description": None,
                "is_cloned": False,
            }
        ]
    }


def test_created_and_deleted_voice_payloads_preserve_http_contract_shape():
    voice = SimpleNamespace(
        id="voice1234",
        name="Roy",
        language="en",
        gender="male",
        created_at="2026-07-02T10:00:00Z",
    )

    assert created_voice_payload(voice) == {
        "id": "voice1234",
        "name": "Roy",
        "language": "en",
        "gender": "male",
        "created_at": "2026-07-02T10:00:00Z",
    }
    assert deleted_voice_payload("voice1234") == {"id": "voice1234", "deleted": True}
