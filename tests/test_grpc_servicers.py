from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from vox.core.store import BlobStore
from vox.core.types import LoadedModelInfo, ModelType
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


class TestHealthServicer:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        from vox.grpc.health_servicer import HealthServicer

        scheduler = MagicMock()
        servicer = HealthServicer(scheduler)

        resp = await servicer.Health(vox_pb2.HealthRequest(), FakeContext())
        assert resp.status == "ok"

    @pytest.mark.asyncio
    async def test_list_loaded_returns_models(self):
        from vox.grpc.health_servicer import HealthServicer

        loaded_model = LoadedModelInfo(
            name="whisper",
            tag="large-v3",
            type=ModelType.STT,
            device="cpu",
            vram_bytes=1000,
            loaded_at=1.0,
            last_used=2.0,
            ref_count=1,
        )
        scheduler = MagicMock()
        scheduler.list_loaded.return_value = [loaded_model]

        servicer = HealthServicer(scheduler)
        resp = await servicer.ListLoaded(vox_pb2.ListLoadedRequest(), FakeContext())

        assert len(resp.models) == 1
        assert resp.models[0].name == "whisper"
        assert resp.models[0].tag == "large-v3"
        assert resp.models[0].type == "stt"
        assert resp.models[0].device == "cpu"

    @pytest.mark.asyncio
    async def test_list_loaded_empty(self):
        from vox.grpc.health_servicer import HealthServicer

        scheduler = MagicMock()
        scheduler.list_loaded.return_value = []

        servicer = HealthServicer(scheduler)
        resp = await servicer.ListLoaded(vox_pb2.ListLoadedRequest(), FakeContext())
        assert len(resp.models) == 0


class TestModelServicer:
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
    async def test_show_not_found(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        store = _make_store(tmp_path)
        registry = _make_registry_mock()
        scheduler = MagicMock()

        servicer = ModelServicer(store, registry, scheduler)
        ctx = FakeContext()

        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Show(vox_pb2.ShowRequest(name="nonexistent:v1"), ctx)

    @pytest.mark.asyncio
    async def test_delete_not_found(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        store = _make_store(tmp_path)
        registry = _make_registry_mock()
        scheduler = MagicMock()
        scheduler.unload = AsyncMock(return_value=True)

        servicer = ModelServicer(store, registry, scheduler)
        ctx = FakeContext()

        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Delete(vox_pb2.DeleteRequest(name="nonexistent:v1"), ctx)

    @pytest.mark.asyncio
    async def test_pull_unknown_model(self, tmp_path):
        from vox.grpc.model_servicer import ModelServicer

        store = _make_store(tmp_path)
        registry = _make_registry_mock()
        registry.lookup.return_value = None
        scheduler = MagicMock()

        servicer = ModelServicer(store, registry, scheduler)

        messages = []
        async for msg in servicer.Pull(vox_pb2.PullRequest(name="nonexistent:v1"), FakeContext()):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].status == "error"
        assert "not found" in messages[0].error


class TestTranscriptionServicer:
    @pytest.mark.asyncio
    async def test_transcribe_no_audio(self, tmp_path):
        from vox.grpc.transcription_servicer import TranscriptionServicer

        store = _make_store(tmp_path)
        registry = MagicMock()
        scheduler = MagicMock()

        servicer = TranscriptionServicer(store, registry, scheduler)
        ctx = FakeContext()

        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Transcribe(
                vox_pb2.TranscribeRequest(model="whisper:large-v3", audio=b""),
                ctx,
            )

    @pytest.mark.asyncio
    async def test_transcribe_no_model_no_default(self, tmp_path):
        from vox.grpc.transcription_servicer import TranscriptionServicer

        store = _make_store(tmp_path)
        registry = MagicMock()
        registry.available_models.return_value = {}
        scheduler = MagicMock()

        servicer = TranscriptionServicer(store, registry, scheduler)
        ctx = FakeContext()

        with pytest.raises(Exception, match="gRPC abort"):
            await servicer.Transcribe(
                vox_pb2.TranscribeRequest(audio=b"\x00\x00"),
                ctx,
            )


class TestSynthesisServicer:
    @pytest.mark.asyncio
    async def test_synthesize_no_input(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        registry = MagicMock()
        scheduler = MagicMock()

        servicer = SynthesisServicer(store, registry, scheduler)
        ctx = FakeContext()

        with pytest.raises(Exception, match="gRPC abort"):
            async for _ in servicer.Synthesize(
                vox_pb2.SynthesizeRequest(model="kokoro:v1.0", input=""),
                ctx,
            ):
                pass

    @pytest.mark.asyncio
    async def test_list_voices_empty(self, tmp_path):
        from vox.grpc.synthesis_servicer import SynthesisServicer

        store = _make_store(tmp_path)
        registry = MagicMock()
        scheduler = MagicMock()
        scheduler.list_loaded.return_value = []

        servicer = SynthesisServicer(store, registry, scheduler)
        resp = await servicer.ListVoices(vox_pb2.ListVoicesRequest(), FakeContext())
        assert len(resp.voices) == 0


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
            audio=b"\x00\x00",
            model="whisper:large-v3",
            language="en",
            word_timestamps=True,
            temperature=0.2,
        )
        assert msg.model == "whisper:large-v3"
        assert msg.word_timestamps is True

    def test_synthesize_request_fields(self):
        msg = vox_pb2.SynthesizeRequest(
            model="kokoro:v1.0",
            input="Hello world",
            voice="af_heart",
            speed=1.5,
        )
        assert msg.input == "Hello world"
        assert msg.speed == pytest.approx(1.5)

    def test_voice_info_fields(self):
        msg = vox_pb2.VoiceInfo(
            id="af_heart",
            name="Heart",
            language="en-us",
            gender="female",
        )
        assert msg.id == "af_heart"
        assert msg.gender == "female"
