from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from vox.audio.codecs import encode_wav
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.cloned_voices import create_stored_voice
from vox.core.store import BlobStore
from vox.core.types import (
    AdapterInfo,
    LoadedModelInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    TranscribeResult,
    TranscriptSegment,
    VoiceInfo,
)
from vox.grpc import vox_pb2


def _make_store(tmp_path: Path) -> BlobStore:
    return BlobStore(root=tmp_path)


def _make_registry_mock() -> MagicMock:
    registry = MagicMock()
    registry.resolve_model_ref.side_effect = lambda name, tag, explicit_tag=False: (name, tag)
    return registry


class FakeContext:
    def __init__(self):
        self._code = None
        self._details = None

    async def abort(self, code, details):
        self._code = code
        self._details = details
        raise Exception(f"gRPC abort: {code} {details}")


class DummyScheduler:
    def __init__(self, adapter, loaded_models=None):
        self._adapter = adapter
        self._loaded_models = loaded_models or []

    @asynccontextmanager
    async def acquire(self, _model_name: str):
        yield self._adapter

    def list_loaded(self):
        return self._loaded_models


class FakeCloneableTTSAdapter(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts-cloneable", type=ModelType.TTS,
            architectures=("fake",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            supports_voice_cloning=True,
        )
    def load(self, *a, **k): pass
    def unload(self): pass
    @property
    def is_loaded(self): return True
    def list_voices(self):
        return [VoiceInfo(id="default", name="Default", language="en")]
    async def synthesize(self, text, **kwargs):
        yield SynthesizeChunk(audio=np.zeros(64, dtype=np.float32).tobytes(), sample_rate=24_000, is_final=False)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class FakeSTTAdapter(STTAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt", type=ModelType.STT,
            architectures=("fake",), default_sample_rate=16_000,
            supported_formats=(ModelFormat.ONNX,),
        )
    def load(self, *a, **k): pass
    def unload(self): pass
    @property
    def is_loaded(self): return True

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        return TranscribeResult(
            text="hello grpc", language="en", duration_ms=1000, model="fake-stt:latest",
            segments=(TranscriptSegment(text="hello grpc", start_ms=0, end_ms=1000),),
        )


class TestHealthServicer:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        from vox.grpc.health_servicer import HealthServicer

        servicer = HealthServicer(MagicMock())
        resp = await servicer.Health(vox_pb2.HealthRequest(), FakeContext())
        assert resp.status == "ok"

    @pytest.mark.asyncio
    async def test_list_loaded_returns_models(self):
        from vox.grpc.health_servicer import HealthServicer

        loaded_model = LoadedModelInfo(
            name="whisper", tag="large-v3", type=ModelType.STT, device="cpu",
            vram_bytes=1000, loaded_at=1.0, last_used=2.0, ref_count=1,
        )
        scheduler = MagicMock()
        scheduler.list_loaded.return_value = [loaded_model]
        servicer = HealthServicer(scheduler)
        resp = await servicer.ListLoaded(vox_pb2.ListLoadedRequest(), FakeContext())
        assert len(resp.models) == 1
        assert resp.models[0].name == "whisper"
        assert resp.models[0].type == "stt"


class TestModelServicerMapping:
    @pytest.mark.asyncio
    async def test_list_returns_models(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        store = _make_store(tmp_path)
        registry = _make_registry_mock()
        scheduler = MagicMock()

        model_info = MagicMock()
        model_info.full_name = "whisper:large-v3"
        model_info.type = MagicMock(value="stt")
        model_info.format = MagicMock(value="ct2")
        model_info.architecture = "whisper"
        model_info.size_bytes = 5000
        model_info.description = "test model"
        store.list_models = MagicMock(return_value=[model_info])

        servicer = ModelServicer(store, registry, scheduler)
        resp = await servicer.List(vox_pb2.ListModelsRequest(), FakeContext())
        assert len(resp.models) == 1
        assert resp.models[0].name == "whisper:large-v3"
        assert resp.models[0].type == "stt"

    @pytest.mark.asyncio
    async def test_show_not_found_aborts_with_not_found(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        servicer = ModelServicer(_make_store(tmp_path), _make_registry_mock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Show(vox_pb2.ShowRequest(name="nonexistent:v1"), FakeContext())

    @pytest.mark.asyncio
    async def test_delete_not_found_aborts_with_not_found(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        scheduler = MagicMock()
        scheduler.unload = AsyncMock(return_value=True)
        servicer = ModelServicer(_make_store(tmp_path), _make_registry_mock(), scheduler)
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Delete(vox_pb2.DeleteRequest(name="nonexistent:v1"), FakeContext())

    @pytest.mark.asyncio
    async def test_pull_unknown_yields_error_event(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        registry = _make_registry_mock()
        registry.lookup.return_value = None
        servicer = ModelServicer(_make_store(tmp_path), registry, MagicMock())

        messages = []
        async for msg in servicer.Pull(vox_pb2.PullRequest(name="nonexistent:v1"), FakeContext()):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].status == "error"
        assert "not found" in messages[0].error


class TestTranscriptionServicerMapping:
    @pytest.mark.asyncio
    async def test_transcribe_returns_text_and_segments(self, tmp_path):
        from vox.grpc.transcription_servicer import TranscriptionServicer

        store = _make_store(tmp_path)
        registry = MagicMock()
        scheduler = DummyScheduler(FakeSTTAdapter())
        servicer = TranscriptionServicer(store, registry, scheduler)

        response = await servicer.Transcribe(
            vox_pb2.TranscribeRequest(
                model="fake-stt:latest",
                audio=encode_wav(np.zeros(16_000, dtype=np.float32), 16_000),
                format_hint="wav",
            ),
            FakeContext(),
        )
        assert response.text == "hello grpc"
        assert response.model == "fake-stt:latest"

    @pytest.mark.asyncio
    async def test_transcribe_no_audio_aborts_with_invalid_argument(self, tmp_path):
        from vox.grpc.transcription_servicer import TranscriptionServicer

        servicer = TranscriptionServicer(_make_store(tmp_path), MagicMock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Transcribe(
                vox_pb2.TranscribeRequest(model="whisper:large-v3", audio=b""),
                FakeContext(),
            )

    @pytest.mark.asyncio
    async def test_transcribe_no_default_model_aborts_with_invalid_argument(self, tmp_path):
        from vox.grpc.transcription_servicer import TranscriptionServicer

        registry = MagicMock()
        registry.available_models.return_value = {}
        servicer = TranscriptionServicer(_make_store(tmp_path), registry, MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Transcribe(
                vox_pb2.TranscribeRequest(audio=b"\x00\x00"),
                FakeContext(),
            )


class TestSynthesisServicerMapping:
    @pytest.mark.asyncio
    async def test_synthesize_no_input_aborts_with_invalid_argument(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        servicer = SynthesisServicer(_make_store(tmp_path), MagicMock(), MagicMock())
        with pytest.raises(Exception, match="gRPC abort"):
            async for _ in servicer.Synthesize(
                vox_pb2.SynthesizeRequest(model="kokoro:v1.0", input=""), FakeContext(),
            ):
                pass

    @pytest.mark.asyncio
    async def test_list_voices_empty(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        scheduler = MagicMock()
        scheduler.list_loaded.return_value = []
        servicer = SynthesisServicer(_make_store(tmp_path), MagicMock(), scheduler)
        resp = await servicer.ListVoices(vox_pb2.ListVoicesRequest(), FakeContext())
        assert len(resp.voices) == 0

    @pytest.mark.asyncio
    async def test_create_voice_returns_proto_with_id_and_name(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        servicer = SynthesisServicer(store, MagicMock(), MagicMock())
        resp = await servicer.CreateVoice(
            vox_pb2.CreateVoiceRequest(
                name="Roy",
                audio=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
                format_hint="wav", language="en", gender="male",
            ),
            FakeContext(),
        )
        assert resp.voice.name == "Roy"
        assert resp.voice.is_cloned is True

    @pytest.mark.asyncio
    async def test_delete_voice_returns_deleted_proto(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        create_stored_voice(
            store, voice_id="voice1234", name="Roy",
            audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
            content_type="audio/wav",
        )
        servicer = SynthesisServicer(store, MagicMock(), MagicMock())
        resp = await servicer.DeleteVoice(
            vox_pb2.DeleteVoiceRequest(id="voice1234"), FakeContext(),
        )
        assert resp.id == "voice1234"
        assert resp.deleted is True

    @pytest.mark.asyncio
    async def test_list_voices_includes_loaded_model_field(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        create_stored_voice(
            store, voice_id="voice1234", name="Roy",
            audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
            content_type="audio/wav",
        )
        loaded = LoadedModelInfo(name="test-tts", tag="latest", type=ModelType.TTS, device="cpu")
        scheduler = DummyScheduler(FakeCloneableTTSAdapter(), loaded_models=[loaded])
        servicer = SynthesisServicer(store, MagicMock(), scheduler)
        resp = await servicer.ListVoices(vox_pb2.ListVoicesRequest(), FakeContext())
        assert any(v.id == "voice1234" and v.is_cloned for v in resp.voices)


class TestProtoMessages:
    def test_pull_progress_fields(self):
        msg = vox_pb2.PullProgress(status="downloading model.onnx", completed=1, total=3)
        assert msg.status == "downloading model.onnx"
        assert msg.completed == 1
        assert msg.total == 3

    def test_audio_chunk_fields(self):
        audio_bytes = np.zeros(100, dtype=np.float32).tobytes()
        msg = vox_pb2.AudioChunk(audio=audio_bytes, sample_rate=24000, is_final=False)
        assert len(msg.audio) == 400
        assert msg.sample_rate == 24000
        assert msg.is_final is False

    def test_transcribe_request_fields(self):
        msg = vox_pb2.TranscribeRequest(
            audio=b"\x00\x00", model="whisper:large-v3", language="en",
            word_timestamps=True, temperature=0.2,
        )
        assert msg.model == "whisper:large-v3"
        assert msg.word_timestamps is True

    def test_synthesize_request_fields(self):
        msg = vox_pb2.SynthesizeRequest(
            model="kokoro:v1.0", input="Hello world", voice="af_heart", speed=1.5,
        )
        assert msg.input == "Hello world"
        assert msg.speed == pytest.approx(1.5)

    def test_voice_info_fields(self):
        msg = vox_pb2.VoiceInfo(id="af_heart", name="Heart", language="en-us", gender="female")
        assert msg.id == "af_heart"
        assert msg.gender == "female"

    def test_create_voice_request_fields(self):
        msg = vox_pb2.CreateVoiceRequest(name="Roy", audio=b"123", format_hint="wav")
        assert msg.name == "Roy"
        assert msg.audio == b"123"
        assert msg.format_hint == "wav"

    def test_delete_voice_response_fields(self):
        msg = vox_pb2.DeleteVoiceResponse(id="voice1234", deleted=True)
        assert msg.id == "voice1234"
        assert msg.deleted is True
