from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.audio.codecs import encode_wav
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.cloned_voices import create_stored_voice
from vox.core.store import BlobStore
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    TranscribeResult,
    TranscriptSegment,
    VoiceInfo,
    WordTimestamp,
)
from vox.server.routes.health import router as health_router
from vox.server.routes.models import router as models_router
from vox.server.routes.synthesize import router as synth_router
from vox.server.routes.transcribe import router as transcribe_router
from vox.server.routes.voices import router as voices_router


def _wav_bytes(dur_s: float = 1.2, sr: int = 16_000) -> bytes:
    t = np.arange(int(dur_s * sr)) / sr
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return encode_wav(audio, sr)


class _FakeSTT(STTAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt", type=ModelType.STT,
            architectures=("fake",), default_sample_rate=16_000,
            supported_formats=(ModelFormat.ONNX,),
            supports_word_timestamps=True,
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True

    def transcribe(self, audio, **kwargs):
        words = (
            WordTimestamp(word="Alice", start_ms=0, end_ms=500),
            WordTimestamp(word="visited", start_ms=500, end_ms=900),
            WordTimestamp(word="Paris", start_ms=900, end_ms=1300),
        )
        seg = TranscriptSegment(
            text="Alice visited Paris",
            start_ms=0, end_ms=1300,
            words=words,
        )
        return TranscribeResult(
            text="Alice visited Paris",
            segments=(seg,),
            language="en",
            duration_ms=1300,
        )


class _FakeTTS(TTSAdapter):
    def __init__(self):
        self.last_kwargs: dict | None = None

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts", type=ModelType.TTS,
            architectures=("fake",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            supports_voice_cloning=True,
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **kwargs):
        self.last_kwargs = kwargs
        yield SynthesizeChunk(
            audio=np.full(2048, 0.0, dtype=np.float32).tobytes(),
            sample_rate=24_000, is_final=False,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class _DummyScheduler:
    def __init__(self, adapter):
        self._adapter = adapter
        self._loaded = []

    @asynccontextmanager
    async def acquire(self, _model):
        yield self._adapter

    def list_loaded(self):
        return self._loaded

    async def unload(self, _model):
        return True


def _build_app(*, store: BlobStore, stt=None, tts=None, loaded=None):
    app = FastAPI()
    app.state.store = store
    registry = MagicMock()
    registry.available_models.return_value = {
        "fake-stt": {"latest": {"type": "stt"}},
        "fake-tts": {"latest": {"type": "tts"}},
    }
    registry.resolve_model_ref.side_effect = lambda n, t, explicit_tag=False: (n, t or "latest")
    registry.lookup.return_value = None
    app.state.registry = registry
    sched = _DummyScheduler(stt or _FakeSTT())
    if tts is not None:

        sched_tts = _DummyScheduler(tts)
        app.state.scheduler = sched
        app.state.tts_scheduler = sched_tts
    else:
        app.state.scheduler = sched
    app.state.scheduler._loaded = loaded or []
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(transcribe_router)
    app.include_router(synth_router)
    app.include_router(voices_router)
    return app


class TestHealthAlias:
    def test_v1_health_returns_ok(self, tmp_path: Path):
        client = TestClient(_build_app(store=BlobStore(root=tmp_path)))
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_legacy_api_health_returns_ok(self, tmp_path: Path):
        client = TestClient(_build_app(store=BlobStore(root=tmp_path)))
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestModelsAlias:
    def test_v1_models_lists(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        client = TestClient(_build_app(store=store))
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json() == {"models": []}

    def test_v1_models_loaded_returns_empty(self, tmp_path: Path):
        client = TestClient(_build_app(store=BlobStore(root=tmp_path)))
        resp = client.get("/v1/models/loaded")
        assert resp.status_code == 200
        assert resp.json() == {"models": []}

    def test_v1_models_show_path_param_404(self, tmp_path: Path):
        client = TestClient(_build_app(store=BlobStore(root=tmp_path)))
        resp = client.get("/v1/models/nonexistent:v1")
        assert resp.status_code == 404

    def test_v1_models_delete_path_param_404(self, tmp_path: Path):
        client = TestClient(_build_app(store=BlobStore(root=tmp_path)))
        resp = client.delete("/v1/models/nonexistent:v1")
        assert resp.status_code == 404

    def test_v1_models_pull_rejects_unknown(self, tmp_path: Path):
        client = TestClient(_build_app(store=BlobStore(root=tmp_path)))
        resp = client.post("/v1/models/pull", json={"name": "nonexistent:v1"})
        assert resp.status_code == 404


class TestV1TranscriptionsVerboseJson:
    def test_default_is_thin_text_only(self, tmp_path: Path):
        app = _build_app(store=BlobStore(root=tmp_path), stt=_FakeSTT())
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "fake-stt:latest"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"text"}
        assert body["text"] == "Alice visited Paris"

    def test_verbose_json_includes_segments_words(self, tmp_path: Path):
        app = _build_app(store=BlobStore(root=tmp_path), stt=_FakeSTT())
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "fake-stt:latest", "response_format": "verbose_json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "Alice visited Paris"
        assert body["model"] == "fake-stt:latest"
        assert "segments" in body
        assert body["segments"][0]["text"] == "Alice visited Paris"
        assert body["segments"][0]["words"][0]["word"] == "Alice"

    def test_verbose_json_includes_entities_and_topics_when_present(self, tmp_path: Path):
        app = _build_app(store=BlobStore(root=tmp_path), stt=_FakeSTT())
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "fake-stt:latest", "response_format": "verbose_json", "language": "en"},
        )
        body = resp.json()

        assert "model" in body and "duration_ms" in body
        if "entities" in body:
            assert any(e.get("type") for e in body["entities"])

    def test_text_format_returns_plain(self, tmp_path: Path):
        app = _build_app(store=BlobStore(root=tmp_path), stt=_FakeSTT())
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "fake-stt:latest", "response_format": "text"},
        )
        assert resp.status_code == 200
        assert resp.text == "Alice visited Paris"


class TestV1SpeechRichFlags:
    def test_speech_accepts_stream_and_language(self, tmp_path: Path):
        tts = _FakeTTS()
        store = BlobStore(root=tmp_path)

        create_stored_voice(
            store,
            voice_id="voice1234",
            name="Roy",
            audio_bytes=_wav_bytes(),
            content_type="audio/wav",
            language="en",
        )
        app = FastAPI()
        app.state.store = store
        reg = MagicMock()
        reg.available_models.return_value = {"fake-tts": {"latest": {"type": "tts"}}}
        reg.resolve_model_ref.side_effect = lambda n, t, explicit_tag=False: (n, t or "latest")
        app.state.registry = reg
        app.state.scheduler = _DummyScheduler(tts)
        app.include_router(synth_router)
        client = TestClient(app)

        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "fake-tts:latest",
                "input": "hello",
                "voice": "voice1234",
                "language": "fr",
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200

        assert tts.last_kwargs is not None
        assert tts.last_kwargs["voice"] is None
        assert tts.last_kwargs["reference_audio"] is not None

        assert tts.last_kwargs["language"] == "fr"
