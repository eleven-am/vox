"""Tests for Vox HTTP routes (health, models, transcribe, synthesize, voices)."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.errors import ModelNotFoundError
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

# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------

class FakeSTTAdapter(STTAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt",
            type=ModelType.STT,
            architectures=("fake",),
            default_sample_rate=16000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def transcribe(self, audio, **kwargs) -> TranscribeResult:
        segments = (
            TranscriptSegment(
                text="hello world",
                start_ms=0,
                end_ms=1000,
                words=(
                    WordTimestamp(word="hello", start_ms=0, end_ms=500, confidence=0.99),
                    WordTimestamp(word="world", start_ms=500, end_ms=1000, confidence=0.98),
                ),
            ),
        )
        return TranscribeResult(
            text="hello world", language="en", duration_ms=1000, model="test",
            segments=segments,
        )


class FakeTTSAdapter(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts",
            type=ModelType.TTS,
            architectures=("fake",),
            default_sample_rate=24000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self):
        return [
            VoiceInfo(id="default", name="Default", language="en"),
            VoiceInfo(id="narrator", name="Narrator", language="en", gender="male"),
        ]

    async def synthesize(self, text, **kwargs):
        audio = np.zeros(24000, dtype=np.float32)
        yield SynthesizeChunk(
            audio=audio.tobytes(), sample_rate=24000, is_final=True
        )


# ---------------------------------------------------------------------------
# Mock scheduler
# ---------------------------------------------------------------------------

class MockScheduler:
    """Minimal scheduler mock that returns fake adapters via acquire()."""

    def __init__(self):
        self._adapters: dict[str, STTAdapter | TTSAdapter] = {}
        self._loaded_models: list = []
        self._unload_response: bool = True

    def register(self, model_name: str, adapter: STTAdapter | TTSAdapter) -> None:
        self._adapters[model_name] = adapter

    @asynccontextmanager
    async def acquire(self, model_name: str):
        # Normalise "name:tag" -> lookup
        adapter = self._adapters.get(model_name)
        if adapter is None:
            raise ModelNotFoundError(model_name)
        yield adapter

    def list_loaded(self):
        return self._loaded_models

    def set_loaded(self, models):
        self._loaded_models = models

    def set_unload_result(self, value: bool):
        self._unload_response = value

    async def unload(self, model_name: str) -> bool:
        """Return True if unloaded successfully, False if in use."""
        return self._unload_response

    async def start(self):
        pass

    async def stop(self):
        pass


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

def _build_app(
    scheduler: MockScheduler | None = None,
    registry: Any = None,
    store: Any = None,
) -> FastAPI:
    """Build a FastAPI app with injected mocks (no lifespan)."""
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
    """Default TestClient with empty scheduler/registry/store."""
    app = _build_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health & PS
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_endpoint(self, client: TestClient):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ps_no_models_loaded(self, client: TestClient):
        resp = client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------

class TestListModels:
    def test_list_models_empty(self, client: TestClient):
        resp = client.get("/api/list")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


class TestPullModels:
    def test_pull_persists_catalog_source_in_manifest(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        registry = MagicMock()
        registry.lookup.return_value = {
            "architecture": "qwen3-tts",
            "type": "tts",
            "adapter": "qwen3-tts",
            "format": "pytorch",
            "source": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "parameters": {"default_voice": "Ryan"},
            "adapter_package": "",
        }
        app = _build_app(registry=registry, store=store)
        client = TestClient(app)

        downloaded = tmp_path / "model.bin"
        downloaded.write_bytes(b"fake-qwen-model")

        with (
            patch("huggingface_hub.HfApi") as mock_api_cls,
            patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        ):
            mock_api = mock_api_cls.return_value
            mock_api.repo_info.return_value = MagicMock(
                siblings=[MagicMock(rfilename="model.bin")]
            )

            resp = client.post("/api/pull", json={"name": "qwen3-tts:0.6b"})

        assert resp.status_code == 200
        assert '"status": "success"' in resp.text

        manifest = store.resolve_model("qwen3-tts", "0.6b")
        assert manifest is not None
        assert manifest.config["source"] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


# ---------------------------------------------------------------------------
# Transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    def _make_client_with_stt(self) -> TestClient:
        scheduler = MockScheduler()
        scheduler.register("test-stt:latest", FakeSTTAdapter())
        app = _build_app(scheduler=scheduler)
        return TestClient(app)

    @patch("vox.server.routes.transcribe.prepare_for_stt", return_value=np.zeros(16000, dtype=np.float32))
    def test_transcribe_returns_json(self, _mock_prep):
        client = self._make_client_with_stt()
        audio_bytes = b"\x00" * 1000
        resp = client.post(
            "/api/transcribe",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "test-stt:latest", "response_format": "json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "hello world"
        assert body["language"] == "en"
        assert body["duration_ms"] == 1000
        assert "processing_ms" in body

    @patch("vox.server.routes.transcribe.prepare_for_stt", return_value=np.zeros(16000, dtype=np.float32))
    def test_transcribe_text_format(self, _mock_prep):
        client = self._make_client_with_stt()
        audio_bytes = b"\x00" * 1000
        resp = client.post(
            "/api/transcribe",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "test-stt:latest", "response_format": "text"},
        )
        assert resp.status_code == 200
        assert resp.text == "hello world"
        assert "text/plain" in resp.headers["content-type"]

    @patch("vox.server.routes.transcribe.prepare_for_stt", return_value=np.zeros(16000, dtype=np.float32))
    def test_transcribe_model_not_found_404(self, _mock_prep):
        client = self._make_client_with_stt()
        audio_bytes = b"\x00" * 1000
        resp = client.post(
            "/api/transcribe",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "no-such-model:latest"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------

class TestSynthesize:
    def _make_client_with_tts(self) -> TestClient:
        scheduler = MockScheduler()
        scheduler.register("test-tts:latest", FakeTTSAdapter())
        app = _build_app(scheduler=scheduler)
        return TestClient(app)

    def test_synthesize_returns_wav(self):
        client = self._make_client_with_tts()
        resp = client.post(
            "/api/synthesize",
            json={
                "model": "test-tts:latest",
                "input": "hello",
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200
        assert "audio/wav" in resp.headers["content-type"]
        # WAV files start with RIFF header
        assert resp.content[:4] == b"RIFF"

    def test_synthesize_model_not_found_404(self):
        client = self._make_client_with_tts()
        resp = client.post(
            "/api/synthesize",
            json={
                "model": "no-such-model:latest",
                "input": "hello",
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

class TestVoices:
    def test_voices_empty(self, client: TestClient):
        resp = client.get("/api/voices")
        assert resp.status_code == 200
        assert resp.json()["voices"] == []


# ---------------------------------------------------------------------------
# get_default_model
# ---------------------------------------------------------------------------

class TestGetDefaultModel:
    def test_get_default_model_prefers_pulled(self):
        """When the store has a pulled model of the requested type, use it."""
        pulled = ModelInfo(
            name="whisper",
            tag="large-v3",
            type=ModelType.STT,
            format=ModelFormat.ONNX,
            architecture="whisper",
            adapter="whisper",
            size_bytes=100,
        )
        store = MagicMock()
        store.list_models.return_value = [pulled]
        registry = MagicMock()

        result = get_default_model("stt", registry, store)
        assert result == "whisper:large-v3"

    def test_get_default_model_falls_back_to_catalog(self):
        """When no pulled models match, fall back to the catalog."""
        store = MagicMock()
        store.list_models.return_value = []

        registry = MagicMock()
        registry.available_models.return_value = {
            "whisper": {
                "large-v3": {"type": "stt", "source": "test"},
            },
        }

        result = get_default_model("stt", registry, store)
        assert result == "whisper:large-v3"

    def test_get_default_model_raises_when_none(self):
        """When no models of the requested type exist anywhere, raise 400."""
        from fastapi import HTTPException

        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {}

        with pytest.raises(HTTPException) as exc_info:
            get_default_model("stt", registry, store)
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _mime_to_format
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers for new tests
# ---------------------------------------------------------------------------

def _make_manifest():
    """Create a minimal Manifest for testing."""
    return Manifest(
        layers=[
            ManifestLayer(
                media_type="application/vox.model.onnx",
                digest="sha256-abc123",
                size=1024,
                filename="model.onnx",
            ),
        ],
        config={
            "architecture": "whisper",
            "type": "stt",
            "adapter": "whisper",
            "format": "onnx",
            "description": "Test model",
        },
    )


def _make_store_mock(**overrides):
    """Build a store MagicMock with sensible defaults."""
    store = MagicMock()
    store.list_models.return_value = overrides.get("list_models", [])
    store.resolve_model.return_value = overrides.get("resolve_model")
    store.delete_model.return_value = None
    store.gc_blobs.return_value = None
    return store


# ---------------------------------------------------------------------------
# Show model
# ---------------------------------------------------------------------------

class TestShowModel:
    def test_show_model_returns_details(self):
        manifest = _make_manifest()
        store = _make_store_mock(resolve_model=manifest)
        app = _build_app(store=store)
        client = TestClient(app)

        resp = client.post("/api/show", json={"name": "whisper:large-v3"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "whisper:large-v3"
        assert body["config"]["architecture"] == "whisper"
        assert len(body["layers"]) == 1
        assert body["layers"][0]["digest"] == "sha256-abc123"
        assert body["layers"][0]["filename"] == "model.onnx"

    def test_show_model_not_found(self):
        store = _make_store_mock(resolve_model=None)
        app = _build_app(store=store)
        client = TestClient(app)

        resp = client.post("/api/show", json={"name": "no-such-model:latest"})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Delete model
# ---------------------------------------------------------------------------

class TestDeleteModel:
    def test_delete_model_success(self):
        manifest = _make_manifest()
        store = _make_store_mock(resolve_model=manifest)
        scheduler = MockScheduler()
        scheduler.set_unload_result(True)
        app = _build_app(scheduler=scheduler, store=store)
        client = TestClient(app)

        resp = client.request("DELETE", "/api/delete", json={"name": "whisper:large-v3"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        store.delete_model.assert_called_once_with("whisper", "large-v3")
        store.gc_blobs.assert_called_once()

    def test_delete_model_in_use_409(self):
        store = _make_store_mock()
        scheduler = MockScheduler()
        scheduler.set_unload_result(False)
        app = _build_app(scheduler=scheduler, store=store)
        client = TestClient(app)

        resp = client.request("DELETE", "/api/delete", json={"name": "whisper:large-v3"})
        assert resp.status_code == 409
        assert "in use" in resp.json()["detail"].lower()

    def test_delete_model_not_found_404(self):
        store = _make_store_mock(resolve_model=None)
        scheduler = MockScheduler()
        scheduler.set_unload_result(True)
        app = _build_app(scheduler=scheduler, store=store)
        client = TestClient(app)

        resp = client.request("DELETE", "/api/delete", json={"name": "no-such-model:latest"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Transcribe - verbose_json, OpenAI endpoint, wrong model type
# ---------------------------------------------------------------------------

class TestTranscribeExtended:
    def _make_client_with_stt(self) -> TestClient:
        scheduler = MockScheduler()
        scheduler.register("test-stt:latest", FakeSTTAdapter())
        app = _build_app(scheduler=scheduler)
        return TestClient(app)

    @patch("vox.server.routes.transcribe.prepare_for_stt", return_value=np.zeros(16000, dtype=np.float32))
    def test_transcribe_verbose_json(self, _mock_prep):
        client = self._make_client_with_stt()
        audio_bytes = b"\x00" * 1000
        resp = client.post(
            "/api/transcribe",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "test-stt:latest", "response_format": "verbose_json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "hello world"
        assert "segments" in body
        assert len(body["segments"]) == 1
        seg = body["segments"][0]
        assert seg["text"] == "hello world"
        assert seg["start_ms"] == 0
        assert seg["end_ms"] == 1000
        assert len(seg["words"]) == 2
        assert seg["words"][0]["word"] == "hello"
        assert seg["words"][0]["confidence"] == 0.99
        assert seg["words"][1]["word"] == "world"

    @patch("vox.server.routes.transcribe.prepare_for_stt", return_value=np.zeros(16000, dtype=np.float32))
    def test_openai_transcribe_endpoint(self, _mock_prep):
        client = self._make_client_with_stt()
        audio_bytes = b"\x00" * 1000
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "test-stt:latest", "response_format": "json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "hello world"

    @patch("vox.server.routes.transcribe.prepare_for_stt", return_value=np.zeros(16000, dtype=np.float32))
    def test_openai_transcribe_text_format(self, _mock_prep):
        client = self._make_client_with_stt()
        audio_bytes = b"\x00" * 1000
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "test-stt:latest", "response_format": "text"},
        )
        assert resp.status_code == 200
        assert resp.text == "hello world"

    @patch("vox.server.routes.transcribe.prepare_for_stt", return_value=np.zeros(16000, dtype=np.float32))
    def test_transcribe_wrong_model_type_400(self, _mock_prep):
        """Sending a TTS model name to /api/transcribe should return 400."""
        scheduler = MockScheduler()
        scheduler.register("test-tts:latest", FakeTTSAdapter())
        app = _build_app(scheduler=scheduler)
        client = TestClient(app)

        audio_bytes = b"\x00" * 1000
        resp = client.post(
            "/api/transcribe",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "test-tts:latest"},
        )
        assert resp.status_code == 400
        assert "not an STT model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Synthesize - OpenAI endpoint, streaming, wrong model type
# ---------------------------------------------------------------------------

class TestSynthesizeExtended:
    def _make_client_with_tts(self) -> TestClient:
        scheduler = MockScheduler()
        scheduler.register("test-tts:latest", FakeTTSAdapter())
        app = _build_app(scheduler=scheduler)
        return TestClient(app)

    def test_openai_speech_endpoint(self):
        client = self._make_client_with_tts()
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts:latest",
                "input": "hello",
                "voice": "default",
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200
        assert "audio/" in resp.headers["content-type"]
        assert resp.content[:4] == b"RIFF"

    def test_synthesize_streaming(self):
        client = self._make_client_with_tts()
        resp = client.post(
            "/api/synthesize",
            json={
                "model": "test-tts:latest",
                "input": "hello",
                "response_format": "wav",
                "stream": True,
            },
        )
        assert resp.status_code == 200
        # Streaming should still return audio content
        assert len(resp.content) > 0

    def test_synthesize_wrong_model_type_400(self):
        """Sending an STT model name to /api/synthesize should return 400."""
        scheduler = MockScheduler()
        scheduler.register("test-stt:latest", FakeSTTAdapter())
        app = _build_app(scheduler=scheduler)
        client = TestClient(app)

        resp = client.post(
            "/api/synthesize",
            json={
                "model": "test-stt:latest",
                "input": "hello",
            },
        )
        assert resp.status_code == 400
        assert "not a TTS model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Voices - with model query
# ---------------------------------------------------------------------------

class TestVoicesExtended:
    def test_voices_with_model(self):
        """Query voices for a specific loaded TTS model."""
        scheduler = MockScheduler()
        scheduler.register("test-tts:latest", FakeTTSAdapter())
        app = _build_app(scheduler=scheduler)
        client = TestClient(app)

        resp = client.get("/api/voices", params={"model": "test-tts:latest"})
        assert resp.status_code == 200
        voices = resp.json()["voices"]
        assert len(voices) == 2
        ids = {v["id"] for v in voices}
        assert "default" in ids
        assert "narrator" in ids
        assert voices[0]["language"] == "en"

    def test_voices_with_loaded_tts_models(self):
        """When no model is specified, list voices from all loaded TTS models."""
        scheduler = MockScheduler()
        scheduler.register("test-tts:latest", FakeTTSAdapter())
        # Simulate list_loaded returning a model info object
        loaded_info = MagicMock()
        loaded_info.type.value = "tts"
        loaded_info.name = "test-tts"
        loaded_info.tag = "latest"
        scheduler.set_loaded([loaded_info])
        app = _build_app(scheduler=scheduler)
        client = TestClient(app)

        resp = client.get("/api/voices")
        assert resp.status_code == 200
        voices = resp.json()["voices"]
        assert len(voices) == 2
        # Each voice should include the model name
        assert voices[0]["model"] == "test-tts:latest"

    def test_voices_model_not_found(self):
        """Querying voices for a non-existent model returns 404."""
        app = _build_app()
        client = TestClient(app)

        resp = client.get("/api/voices", params={"model": "no-such-model:latest"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_create_app_returns_fastapi_instance(self, tmp_path):
        """create_app should return a FastAPI instance with state configured."""
        from vox.server.app import create_app

        app = create_app(vox_home=tmp_path)
        assert isinstance(app, FastAPI)
        assert hasattr(app.state, "scheduler")
        assert hasattr(app.state, "registry")
        assert hasattr(app.state, "store")
        assert app.title == "Vox"
