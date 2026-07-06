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

from tests.fakes import FakeScheduler
from vox.audio.codecs import encode_wav
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.store import BlobStore, Manifest, ManifestLayer
from vox.core.types import (
    AdapterInfo,
    DeviceMemoryInfo,
    LoadedModelInfo,
    ModelFormat,
    ModelInfo,
    ModelType,
    SynthesizeChunk,
    TranscribeResult,
    TranscriptSegment,
    VoiceInfo,
    VramPolicy,
    VramSnapshot,
    WordTimestamp,
)
from vox.operations.defaults import resolve_default_model
from vox.operations.errors import InvalidConfigError, MemoryBudgetExceededError, ModelInUseError
from vox.operations.transcription import format_hint_from_content_type


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
        self._enforce_error: Exception | None = None
        self._trim_idle_error: Exception | None = None
        self._unload_error: Exception | None = None
        self._acquire_error: Exception | None = None
        self.preloaded: list[str] = []
        self.trimmed: list[str] = []
        self.enforced: list[int] = []

    def list_loaded(self): return self._loaded
    def set_loaded(self, ms): self._loaded = ms
    def set_unload_result(self, v: bool): self._unload = v
    def set_enforce_error(self, error: Exception): self._enforce_error = error
    def set_trim_idle_error(self, error: Exception): self._trim_idle_error = error
    def set_unload_error(self, error: Exception): self._unload_error = error
    def set_acquire_error(self, error: Exception): self._acquire_error = error

    @asynccontextmanager
    async def acquire(self, name: str):
        if self._acquire_error is not None:
            raise self._acquire_error
        async with super().acquire(name) as adapter:
            yield adapter

    async def unload(self, name: str) -> bool:
        if self._unload_error is not None:
            raise self._unload_error
        return self._unload
    async def preload(self, name: str) -> None: self.preloaded.append(name)
    async def trim(self, name: str) -> bool:
        self.trimmed.append(name)
        return self._unload
    async def trim_idle(self, *, min_idle_seconds: int = 0) -> list[str]:
        if self._trim_idle_error is not None:
            raise self._trim_idle_error
        self.trimmed.append(f"idle:{min_idle_seconds}")
        return ["fake:latest"]
    async def enforce_memory_budget(self, *, additional_vram_bytes: int = 0) -> None:
        self.enforced.append(additional_vram_bytes)
        if self._enforce_error is not None:
            raise self._enforce_error
    def memory_snapshot(self):
        return VramSnapshot(
            policy=VramPolicy(max_vram_bytes=10_000, headroom_bytes=1_000, idle_trim_seconds=60),
            device=DeviceMemoryInfo(device="cuda", free_bytes=5_000, total_bytes=20_000),
            loaded_models=tuple(self._loaded),
            estimated_loaded_vram_bytes=4_000,
            active_model_count=0,
        )


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

    from vox.server.routes import health, models, synthesize, system, transcribe, voices

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(system.router)
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

    def test_ps_loaded_models_preserves_existing_contract_shape(self):
        scheduler = MockScheduler()
        scheduler.set_loaded(
            [
                LoadedModelInfo(
                    name="parakeet-stt-onnx",
                    tag="tdt-0.6b-v3",
                    type=ModelType.STT,
                    device="cuda",
                    vram_bytes=4096,
                    loaded_at=1.5,
                    last_used=2.5,
                    ref_count=3,
                    is_evictable=True,
                    backend_memory={"workspace_bytes": 256},
                )
            ]
        )
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.get("/v1/models/loaded")

        assert resp.status_code == 200
        assert resp.json() == {
            "models": [
                {
                    "name": "parakeet-stt-onnx",
                    "tag": "tdt-0.6b-v3",
                    "type": "stt",
                    "device": "cuda",
                    "vram_bytes": 4096,
                    "loaded_at": 1.5,
                    "last_used": 2.5,
                    "ref_count": 3,
                }
            ]
        }


class TestSystemMemory:
    def test_memory_status_returns_policy_and_models(self):
        scheduler = MockScheduler()
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.get("/v1/system/memory")

        assert resp.status_code == 200
        body = resp.json()
        assert body["policy"]["max_vram_bytes"] == 10_000
        assert body["device"]["device"] == "cuda"
        assert body["estimated_loaded_vram_bytes"] == 4_000

    def test_trim_idle_endpoint(self):
        scheduler = MockScheduler()
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.post("/v1/system/trim", json={"min_idle_seconds": 30})

        assert resp.status_code == 200
        assert resp.json()["trimmed"] == ["fake:latest"]
        assert scheduler.trimmed == ["idle:30"]

    def test_trim_idle_endpoint_maps_operation_errors(self):
        scheduler = MockScheduler()
        scheduler.set_trim_idle_error(MemoryBudgetExceededError("Cannot satisfy VRAM budget"))
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.post("/v1/system/trim", json={"min_idle_seconds": 30})

        assert resp.status_code == 507
        assert "Cannot satisfy VRAM budget" in resp.json()["detail"]

    def test_enforce_memory_budget_endpoint(self):
        scheduler = MockScheduler()
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.post("/v1/system/enforce-memory-budget", json={"additional_vram_bytes": 1024})

        assert resp.status_code == 200
        assert scheduler.enforced == [1024]

    def test_enforce_memory_budget_endpoint_maps_budget_failure_to_507(self):
        from vox.core.errors import ModelLoadError

        scheduler = MockScheduler()
        scheduler.set_enforce_error(ModelLoadError("Cannot satisfy VRAM budget"))
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.post("/v1/system/enforce-memory-budget", json={"additional_vram_bytes": 1024})

        assert resp.status_code == 507
        assert "Cannot satisfy VRAM budget" in resp.json()["detail"]

    def test_trim_model_endpoint_returns_conflict_when_active(self):
        scheduler = MockScheduler()
        scheduler.set_unload_result(False)
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.post("/v1/models/fake:latest/trim")

        assert resp.status_code == 409

    def test_unload_idle_endpoint_unloads_inactive_models(self):
        scheduler = MockScheduler()
        scheduler.set_loaded([
            LoadedModelInfo(name="fake", tag="latest", type=ModelType.STT, device="cuda", ref_count=0),
        ])
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.post("/v1/models/unload_idle")

        assert resp.status_code == 200
        assert resp.json()["unloaded"] == ["fake:latest"]

    def test_unload_idle_endpoint_maps_operation_errors(self):
        scheduler = MockScheduler()
        scheduler.set_loaded([
            LoadedModelInfo(name="fake", tag="latest", type=ModelType.STT, device="cuda", ref_count=0),
        ])
        scheduler.set_unload_error(ModelInUseError("fake:latest"))
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.post("/v1/models/unload_idle")

        assert resp.status_code == 409
        assert "fake:latest" in resp.json()["detail"]


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

    def test_pull_variant_field_is_used_for_resolution(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        registry = MagicMock()
        registry.lookup.return_value = {
            "architecture": "fake",
            "type": "tts",
            "variants": [
                {
                    "id": "onnx",
                    "requires": {},
                    "adapter": "fake",
                    "format": "onnx",
                    "source": "owner/repo",
                    "adapter_package": "",
                }
            ],
        }
        registry.resolve_model_ref.side_effect = lambda n, t, explicit_tag=False: (n, t)
        scheduler = MockScheduler()
        app = _build_app(registry=registry, store=store, scheduler=scheduler)
        client = TestClient(app)

        resp = client.post("/v1/models/pull", json={"name": "foo:latest", "variant": "mlx"})

        assert resp.status_code == 409
        assert "variant 'mlx' is not defined" in resp.text


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
        assert "segments" in body and "duration" in body and "processing_ms" in body
        assert body["duration"] == 1.0
        assert body["segments"][0]["id"] == 0
        assert body["segments"][0]["start"] == 0.0
        assert body["segments"][0]["end"] == 1.0
        assert body["segments"][0]["tokens"] == []
        assert "duration_ms" not in body
        assert "start_ms" not in body["segments"][0]
        assert "words" not in body

    def test_verbose_json_word_granularity_returns_top_level_words(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={
                "model": "test-stt:latest",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "segments" not in body
        assert body["words"] == [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]

    def test_verbose_json_accepts_json_encoded_timestamp_granularities(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={
                "model": "test-stt:latest",
                "response_format": "verbose_json",
                "timestamp_granularities": '["word","segment"]',
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["segments"][0]["start"] == 0.0
        assert body["words"][0] == {"word": "hello", "start": 0.0, "end": 0.5}

    def test_verbose_json_accepts_comma_separated_timestamp_granularities(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={
                "model": "test-stt:latest",
                "response_format": "verbose_json",
                "timestamp_granularities": "word,segment",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["segments"][0]["end"] == 1.0
        assert body["words"][1] == {"word": "world", "start": 0.5, "end": 1.0}

    def test_octet_stream_upload_allows_decoder_autodetection(self):
        assert format_hint_from_content_type("application/octet-stream") is None

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

    def test_request_builder_operation_errors_are_mapped_to_http(self, monkeypatch):
        from vox.server.routes import transcribe

        def fail_build(**_kwargs):
            raise InvalidConfigError("bad transcribe route config")

        monkeypatch.setattr(transcribe, "transcription_request_from_fields", fail_build)

        client = self._client()
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"model": "test-stt:latest"},
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "bad transcribe route config"


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

    def test_openai_speech_request_preserves_stream_and_response_format_fields(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts:latest",
                "input": "hello",
                "response_format": "pcm",
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert "audio/L16" in resp.headers["content-type"]
        assert resp.content[:4] != b"RIFF"

    def test_model_not_found_maps_to_404(self):
        client = self._client()
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "missing:latest", "input": "hello"},
        )
        assert resp.status_code == 404

    def test_model_load_budget_failure_maps_to_507(self):
        from vox.core.errors import ModelLoadError

        scheduler = MockScheduler()
        scheduler.register("test-tts:latest", FakeTTSAdapter())
        scheduler.set_acquire_error(ModelLoadError("Cannot satisfy VRAM budget"))
        client = TestClient(_build_app(scheduler=scheduler))

        resp = client.post(
            "/v1/audio/speech",
            json={"model": "test-tts:latest", "input": "hello"},
        )

        assert resp.status_code == 507
        assert "Cannot satisfy VRAM budget" in resp.json()["detail"]

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

    def test_request_builder_operation_errors_are_mapped_to_http(self, monkeypatch):
        from vox.server.routes import synthesize

        def fail_build(**_kwargs):
            raise InvalidConfigError("bad synthesize route config")

        monkeypatch.setattr(synthesize, "synthesis_request_from_fields", fail_build)

        client = self._client()
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "test-tts:latest", "input": "hello"},
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "bad synthesize route config"


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

    def test_create_voice_invalid_reference_maps_to_422(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        client = TestClient(_build_app(store=store))
        short_wav = encode_wav(np.full(4_000, 0.1, dtype=np.float32), 16_000)

        resp = client.post(
            "/v1/audio/voices",
            files={"audio_sample": ("sample.wav", io.BytesIO(short_wav), "audio/wav")},
            data={"name": "Roy"},
        )

        assert resp.status_code == 422
        assert "too short" in resp.json()["detail"]

    def test_create_voice_decode_failure_maps_to_400(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        client = TestClient(_build_app(store=store))

        resp = client.post(
            "/v1/audio/voices",
            files={"audio_sample": ("sample.wav", io.BytesIO(b"not an audio file"), "audio/wav")},
            data={"name": "Roy"},
        )

        assert resp.status_code == 400

    def test_create_voice_request_builder_operation_errors_are_mapped_to_http(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from vox.server.routes import voices

        def fail_build(**_kwargs):
            raise InvalidConfigError("bad voice route config")

        monkeypatch.setattr(voices, "create_voice_request_from_fields", fail_build)

        store = BlobStore(root=tmp_path)
        client = TestClient(_build_app(store=store))
        resp = client.post(
            "/v1/audio/voices",
            files={"audio_sample": ("sample.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
            data={"name": "Roy"},
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "bad voice route config"

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

    def test_get_voice_reference_route_adapts_operation_response(self, tmp_path: Path):
        from vox.core.cloned_voices import create_stored_voice

        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store, voice_id="voice1234", name="Roy",
            audio_bytes=encode_wav(np.full(16_000, 0.1, dtype=np.float32), 16_000),
            content_type="audio/wav",
        )
        client = TestClient(_build_app(store=store))

        resp = client.get("/v1/audio/voices/voice1234/reference")

        assert resp.status_code == 200
        assert "audio/wav" in resp.headers["content-type"]
        assert resp.headers["content-disposition"] == 'attachment; filename="voice1234.wav"'
        assert resp.content[:4] == b"RIFF"


class TestResolveDefaultModel:
    def test_resolve_default_model_prefers_pulled(self):
        pulled = ModelInfo(
            name="whisper", tag="large-v3", type=ModelType.STT,
            format=ModelFormat.ONNX, architecture="whisper", adapter="whisper",
            size_bytes=100,
        )
        store = MagicMock()
        store.list_models.return_value = [pulled]
        registry = MagicMock()
        assert resolve_default_model("stt", registry, store) == "whisper:large-v3"

    def test_resolve_default_model_falls_back_to_catalog(self):
        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {
            "whisper": {"large-v3": {"type": "stt", "source": "test"}},
        }
        assert resolve_default_model("stt", registry, store) == "whisper:large-v3"

    def test_resolve_default_model_returns_none_when_none_available(self):
        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {}
        assert resolve_default_model("stt", registry, store) is None


class TestFormatHintFromContentType:
    def test_format_hint_from_content_type_conversions(self):
        assert format_hint_from_content_type("audio/wav") == "wav"
        assert format_hint_from_content_type("audio/mpeg") == "mp3"
        assert format_hint_from_content_type("audio/x-wav") == "wav"
        assert format_hint_from_content_type("audio/x-flac") == "flac"
        assert format_hint_from_content_type("audio/ogg") == "ogg"
        assert format_hint_from_content_type("audio/webm") == "webm"
        assert format_hint_from_content_type("audio/flac") == "flac"
        assert format_hint_from_content_type(None) is None
        assert format_hint_from_content_type("") is None


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

    def test_create_app_can_enable_browser_cors(self, tmp_path, monkeypatch):
        from vox.server.app import create_app

        monkeypatch.setenv("VOX_CORS_ORIGINS", "http://localhost:8000")
        client = TestClient(create_app(vox_home=tmp_path))

        resp = client.options(
            "/v1/rtc/sessions",
            headers={
                "origin": "http://localhost:8000",
                "access-control-request-method": "POST",
                "access-control-request-headers": "authorization,content-type",
            },
        )

        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://localhost:8000"
