from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.audio.codecs import encode_wav
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.store import BlobStore, Manifest, ManifestLayer
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelInfo,
    ModelType,
    SynthesizeChunk,
    TranscribeResult,
    TranscriptSegment,
    VoiceInfo,
    WordTimestamp,
)
from vox.server.routes import get_default_model
from vox.server.routes.transcribe import _mime_to_format

from tests.fakes import FakeScheduler


def _wav_bytes(dur_s: float = 1.0, sr: int = 16_000) -> bytes:
    audio = np.zeros(int(dur_s * sr), dtype=np.float32)
    return encode_wav(audio, sr)


class FakeSTTAdapter(STTAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt", type=ModelType.STT,
            architectures=("fake",), default_sample_rate=16000,
            supported_formats=(ModelFormat.ONNX,),
            supports_word_timestamps=True,
        )
    def load(self, *a, **k): pass
    def unload(self): pass
    @property
    def is_loaded(self): return True
    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        return TranscribeResult(
            text="hello world", language="en", duration_ms=1000, model="test",
            segments=(
                TranscriptSegment(
                    text="hello world", start_ms=0, end_ms=1000,
                    words=(
                        WordTimestamp(word="hello", start_ms=0, end_ms=500, confidence=0.99),
                        WordTimestamp(word="world", start_ms=500, end_ms=1000, confidence=0.98),
                    ),
                ),
            ),
        )


class FakeTTSAdapter(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts", type=ModelType.TTS,
            architectures=("fake",), default_sample_rate=24000,
            supported_formats=(ModelFormat.ONNX,),
        )
    def load(self, *a, **k): pass
    def unload(self): pass
    @property
    def is_loaded(self): return True
    def list_voices(self):
        return [VoiceInfo(id="default", name="Default", language="en")]
    async def synthesize(self, text, **kw):
        yield SynthesizeChunk(audio=np.zeros(24000, dtype=np.float32).tobytes(), sample_rate=24000, is_final=True)


class MockScheduler(FakeScheduler):
    def __init__(self):
        super().__init__()
        self._loaded = []
        self._unload = True
        self.preloaded: list[str] = []

    def list_loaded(self): return self._loaded
    def set_loaded(self, ms): self._loaded = ms
    def set_unload_result(self, v: bool): self._unload = v

    async def unload(self, name: str) -> bool: return self._unload
    async def preload(self, name: str) -> None: self.preloaded.append(name)


def _build_app(scheduler: MockScheduler | None = None, registry: Any = None, store: Any = None) -> FastAPI:
    app = FastAPI()
    app.state.scheduler = scheduler or MockScheduler()
    app.state.registry = registry or MagicMock()
    app.state.store = store or MagicMock(list_models=MagicMock(return_value=[]))

    resolver = getattr(app.state.registry, "resolve_model_ref", None)
    if (
        isinstance(resolver, MagicMock)
        and resolver.side_effect is None
        and not isinstance(resolver.return_value, tuple)
    ):
        resolver.side_effect = lambda name, tag, explicit_tag=False: (name, tag)

    from vox.server.routes import health, models, synthesize, transcribe, voices

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(transcribe.router)
    app.include_router(synthesize.router)
    app.include_router(voices.router)
    return app


@pytest.fixture
def client():
    return TestClient(_build_app())


class TestHealth:
    def test_health_endpoint(self, client: TestClient):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ps_no_models_loaded(self, client: TestClient):
        resp = client.get("/v1/models/loaded")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


class TestListModels:
    def test_list_models_empty(self, client: TestClient):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


class TestPullModels:
    def test_pull_emits_ndjson_with_status_lines(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        registry = MagicMock()
        registry.lookup.return_value = {
            "architecture": "fake", "type": "stt", "adapter": "fake", "format": "onnx",
            "source": "owner/repo", "parameters": {}, "adapter_package": "",
        }
        registry.resolve_model_ref.side_effect = lambda n, t, explicit_tag=False: (n, t)
        scheduler = MockScheduler()
        app = _build_app(registry=registry, store=store, scheduler=scheduler)
        client = TestClient(app)

        downloaded = tmp_path / "model.bin"
        downloaded.write_bytes(b"x")
        with (
            patch("huggingface_hub.HfApi") as mock_api_cls,
            patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        ):
            mock_api_cls.return_value.repo_info.return_value = MagicMock(
                siblings=[MagicMock(rfilename="model.bin")]
            )
            resp = client.post("/v1/models/pull", json={"name": "foo:latest"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        assert '"status": "success"' in resp.text

    def test_pull_unknown_returns_404(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        registry = MagicMock()
        registry.lookup.return_value = None
        registry.resolve_model_ref.side_effect = lambda n, t, explicit_tag=False: (n, t)
        scheduler = MockScheduler()
        app = _build_app(registry=registry, store=store, scheduler=scheduler)
        client = TestClient(app)
        resp = client.post("/v1/models/pull", json={"name": "missing:latest"})
        assert resp.status_code == 404


class TestTranscribeMapping:
    def _client(self) -> TestClient:
        scheduler = MockScheduler()
        scheduler.register("test-stt:latest", FakeSTTAdapter())
        return TestClient(_build_app(scheduler=scheduler))

    def test_default_response_format_is_json_text_only(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "test-stt:latest"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"text"}
        assert body["text"] == "hello world"

    def test_text_format_returns_plain(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "test-stt:latest", "response_format": "text"},
        )
        assert resp.status_code == 200
        assert resp.text == "hello world"
        assert "text/plain" in resp.headers["content-type"]

    def test_verbose_json_includes_segments(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "test-stt:latest", "response_format": "verbose_json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "segments" in body and "duration_ms" in body and "processing_ms" in body

    def test_model_not_found_maps_to_404(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "missing:latest"},
        )
        assert resp.status_code == 404

    def test_wrong_model_type_maps_to_400(self):
        scheduler = MockScheduler()
        scheduler.register("test-tts:latest", FakeTTSAdapter())
        client = TestClient(_build_app(scheduler=scheduler))
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "test-tts:latest"},
        )
        assert resp.status_code == 400
        assert "not an STT model" in resp.json()["detail"]


class TestSynthesizeMapping:
    def _client(self) -> TestClient:
        scheduler = MockScheduler()
        scheduler.register("test-tts:latest", FakeTTSAdapter())
        return TestClient(_build_app(scheduler=scheduler))

    def test_returns_audio_with_riff_header(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "test-tts:latest", "input": "hello", "response_format": "wav"},
        )
        assert resp.status_code == 200
        assert "audio/wav" in resp.headers["content-type"]
        assert resp.content[:4] == b"RIFF"

    def test_model_not_found_maps_to_404(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "missing:latest", "input": "hello"},
        )
        assert resp.status_code == 404

    def test_wrong_model_type_maps_to_400(self):
        scheduler = MockScheduler()
        scheduler.register("test-stt:latest", FakeSTTAdapter())
        client = TestClient(_build_app(scheduler=scheduler))
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "test-stt:latest", "input": "hello"},
        )
        assert resp.status_code == 400
        assert "not a TTS model" in resp.json()["detail"]


class TestVoicesMapping:
    def test_voices_empty(self, client: TestClient):
        resp = client.get("/v1/audio/voices")
        assert resp.status_code == 200
        assert resp.json()["voices"] == []

    def test_voices_model_not_found_maps_to_404(self):
        app = _build_app()
        client = TestClient(app)
        resp = client.get("/v1/audio/voices", params={"model": "missing:latest"})
        assert resp.status_code == 404

    def test_create_voice_persists_via_route(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        client = TestClient(_build_app(store=store))
        wav = encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000)
        resp = client.post(
            "/v1/audio/voices",
            files={"audio_sample": ("sample.wav", io.BytesIO(wav), "audio/wav")},
            data={"name": "Roy"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Roy"
        assert (store.voices_dir / body["id"] / "reference.wav").is_file()

    def test_delete_voice_route_removes_directory(self, tmp_path: Path):
        from vox.core.cloned_voices import create_stored_voice

        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store, voice_id="voice1234", name="Roy",
            audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
            content_type="audio/wav",
        )
        client = TestClient(_build_app(store=store))
        resp = client.delete("/v1/audio/voices/voice1234")
        assert resp.status_code == 200
        assert resp.json() == {"id": "voice1234", "deleted": True}


class TestGetDefaultModel:
    def test_get_default_model_prefers_pulled(self):
        pulled = ModelInfo(
            name="whisper", tag="large-v3", type=ModelType.STT,
            format=ModelFormat.ONNX, architecture="whisper", adapter="whisper",
            size_bytes=100,
        )
        store = MagicMock()
        store.list_models.return_value = [pulled]
        registry = MagicMock()
        assert get_default_model("stt", registry, store) == "whisper:large-v3"

    def test_get_default_model_falls_back_to_catalog(self):
        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {
            "whisper": {"large-v3": {"type": "stt", "source": "test"}},
        }
        assert get_default_model("stt", registry, store) == "whisper:large-v3"

    def test_get_default_model_raises_when_none(self):
        from fastapi import HTTPException

        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {}
        with pytest.raises(HTTPException) as exc:
            get_default_model("stt", registry, store)
        assert exc.value.status_code == 400


class TestMimeToFormat:
    def test_mime_to_format_conversions(self):
        assert _mime_to_format("audio/wav") == "wav"
        assert _mime_to_format("audio/mpeg") == "mp3"
        assert _mime_to_format("audio/x-wav") == "wav"
        assert _mime_to_format("audio/x-flac") == "flac"
        assert _mime_to_format("audio/ogg") == "ogg"
        assert _mime_to_format("audio/webm") == "webm"
        assert _mime_to_format("audio/flac") == "flac"
        assert _mime_to_format(None) is None
        assert _mime_to_format("") is None


def _make_manifest():
    return Manifest(
        layers=[ManifestLayer(
            media_type="application/vox.model.onnx",
            digest="sha256-abc123", size=1024, filename="model.onnx",
        )],
        config={
            "architecture": "whisper", "type": "stt", "adapter": "whisper",
            "format": "onnx", "description": "Test model",
        },
    )


def _make_store_mock(**overrides):
    store = MagicMock()
    store.list_models.return_value = overrides.get("list_models", [])
    store.resolve_model.return_value = overrides.get("resolve_model")
    store.delete_model.return_value = None
    store.gc_blobs.return_value = None
    return store


class TestShowModelMapping:
    def test_show_model_returns_details(self):
        manifest = _make_manifest()
        store = _make_store_mock(resolve_model=manifest)
        client = TestClient(_build_app(store=store))
        resp = client.get("/v1/models/whisper:large-v3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "whisper:large-v3"
        assert body["config"]["architecture"] == "whisper"
        assert body["layers"][0]["digest"] == "sha256-abc123"

    def test_show_model_not_found(self):
        store = _make_store_mock(resolve_model=None)
        client = TestClient(_build_app(store=store))
        resp = client.get("/v1/models/no-such-model:latest")
        assert resp.status_code == 404


class TestDeleteModelMapping:
    def test_delete_model_success(self):
        manifest = _make_manifest()
        store = _make_store_mock(resolve_model=manifest)
        scheduler = MockScheduler()
        scheduler.set_unload_result(True)
        client = TestClient(_build_app(scheduler=scheduler, store=store))
        resp = client.delete("/v1/models/whisper:large-v3")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_delete_model_in_use_409(self):
        store = _make_store_mock()
        scheduler = MockScheduler()
        scheduler.set_unload_result(False)
        client = TestClient(_build_app(scheduler=scheduler, store=store))
        resp = client.delete("/v1/models/whisper:large-v3")
        assert resp.status_code == 409
        assert "in use" in resp.json()["detail"].lower()

    def test_delete_model_not_found_404(self):
        store = _make_store_mock(resolve_model=None)
        scheduler = MockScheduler()
        scheduler.set_unload_result(True)
        client = TestClient(_build_app(scheduler=scheduler, store=store))
        resp = client.delete("/v1/models/no-such-model:latest")
        assert resp.status_code == 404


class TestCreateApp:
    def test_create_app_returns_fastapi_instance(self, tmp_path):
        from vox.server.app import create_app

        app = create_app(vox_home=tmp_path)
        assert isinstance(app, FastAPI)
        assert hasattr(app.state, "scheduler")
        assert hasattr(app.state, "registry")
        assert hasattr(app.state, "store")
        assert app.title == "Vox"
