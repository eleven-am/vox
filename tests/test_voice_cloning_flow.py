from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.audio.codecs import encode_wav
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import (
    REFERENCE_MAX_SECONDS,
    REFERENCE_MIN_SECONDS,
    create_stored_voice,
    resolve_voice_request,
    validate_reference_audio,
)
from vox.core.errors import (
    ReferenceAudioInvalidError,
    VoiceCloningUnsupportedError,
    VoiceNotFoundError,
)
from vox.core.store import BlobStore
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.server.routes.voices import router as voices_router
from vox.server.routes.bidi import router as bidi_router


def _sine_wav(duration_s: float = 1.5, sr: int = 16_000, amp: float = 0.1) -> bytes:
    t = np.arange(int(duration_s * sr)) / sr
    audio = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return encode_wav(audio, sr)


class _FakeCloneTTSAdapter(TTSAdapter):
    """Records reference_audio kwargs and yields a few dummy chunks."""

    def __init__(self, supports_cloning: bool = True, sample_rate: int = 24_000) -> None:
        self._supports_cloning = supports_cloning
        self._sr = sample_rate
        self.last_synth_kwargs: dict | None = None

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-clone-tts",
            type=ModelType.TTS,
            architectures=("fake-clone",),
            default_sample_rate=self._sr,
            supported_formats=(ModelFormat.ONNX,),
            supports_voice_cloning=self._supports_cloning,
        )

    def load(self, *_args, **_kwargs) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default", language="en")]

    async def synthesize(self, text: str, **kwargs):
        self.last_synth_kwargs = kwargs
        yield SynthesizeChunk(
            audio=np.full(1024, 0.0, dtype=np.float32).tobytes(),
            sample_rate=self._sr,
            is_final=False,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=self._sr, is_final=True)


class TestValidateReferenceAudio:
    def test_accepts_valid_sample(self):
        audio = np.full(24_000, 0.1, dtype=np.float32)
        validate_reference_audio(audio, 24_000)

    def test_rejects_too_short(self):
        audio = np.full(12_000, 0.1, dtype=np.float32)
        with pytest.raises(ReferenceAudioInvalidError, match="too short"):
            validate_reference_audio(audio, 24_000)

    def test_rejects_too_long(self):
        audio = np.full(int(REFERENCE_MAX_SECONDS * 24_000) + 24_000, 0.1, dtype=np.float32)
        with pytest.raises(ReferenceAudioInvalidError, match="too long"):
            validate_reference_audio(audio, 24_000)

    def test_rejects_silent(self):
        audio = np.zeros(48_000, dtype=np.float32)
        with pytest.raises(ReferenceAudioInvalidError, match="silent"):
            validate_reference_audio(audio, 24_000)

    def test_rejects_clipped(self):
        audio = np.full(48_000, 1.0, dtype=np.float32)
        with pytest.raises(ReferenceAudioInvalidError, match="clipped"):
            validate_reference_audio(audio, 24_000)

    def test_rejects_empty(self):
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ReferenceAudioInvalidError, match="empty"):
            validate_reference_audio(audio, 24_000)

    def test_rejects_bad_sample_rate(self):
        audio = np.full(24_000, 0.1, dtype=np.float32)
        with pytest.raises(ReferenceAudioInvalidError, match="sample rate"):
            validate_reference_audio(audio, 0)

    def test_respects_custom_bounds(self):
        audio = np.full(12_000, 0.1, dtype=np.float32)
        validate_reference_audio(audio, 24_000, min_seconds=0.2)

    def test_small_clip_fraction_passes(self):

        audio = np.full(48_000, 0.1, dtype=np.float32)
        audio[:200] = 1.0
        validate_reference_audio(audio, 24_000)


class TestCreateStoredVoiceValidation:
    def test_rejects_silent_upload(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        silent = encode_wav(np.zeros(24_000, dtype=np.float32), 24_000)
        with pytest.raises(ReferenceAudioInvalidError):
            create_stored_voice(
                store,
                voice_id="v1",
                name="Silent",
                audio_bytes=silent,
                content_type="audio/wav",
            )

    def test_bypass_validate_flag(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        silent = encode_wav(np.zeros(24_000, dtype=np.float32), 24_000)
        voice = create_stored_voice(
            store,
            voice_id="v1",
            name="Silent",
            audio_bytes=silent,
            content_type="audio/wav",
            validate=False,
        )
        assert voice.id == "v1"


class TestResolveVoiceRequest:
    def test_no_voice_returns_nulls(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        adapter = _FakeCloneTTSAdapter()
        v, lang, ref_audio, ref_text = resolve_voice_request(adapter, store, None, "en")
        assert v is None and ref_audio is None and ref_text is None

    def test_passthrough_unknown_voice(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        adapter = _FakeCloneTTSAdapter()
        v, lang, ref_audio, _ = resolve_voice_request(adapter, store, "af_heart", "en")
        assert v == "af_heart"
        assert ref_audio is None

    def test_stored_voice_loads_reference(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store,
            voice_id="v1",
            name="Roy",
            audio_bytes=_sine_wav(),
            content_type="audio/wav",
            language="en",
            reference_text="hello there",
        )
        adapter = _FakeCloneTTSAdapter()
        v, lang, ref_audio, ref_text = resolve_voice_request(adapter, store, "v1", None)
        assert v is None
        assert ref_audio is not None
        assert len(ref_audio) > 0
        assert ref_text == "hello there"
        assert lang == "en"

    def test_explicit_language_overrides_stored(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store,
            voice_id="v1",
            name="Roy",
            audio_bytes=_sine_wav(),
            content_type="audio/wav",
            language="en",
            reference_text="",
        )
        adapter = _FakeCloneTTSAdapter()
        _, lang, _, _ = resolve_voice_request(adapter, store, "v1", "fr")
        assert lang == "fr"

    def test_raises_when_adapter_lacks_cloning(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store,
            voice_id="v1",
            name="Roy",
            audio_bytes=_sine_wav(),
            content_type="audio/wav",
        )
        adapter = _FakeCloneTTSAdapter(supports_cloning=False)
        with pytest.raises(VoiceCloningUnsupportedError):
            resolve_voice_request(adapter, store, "v1", None)


class _DummyScheduler:
    def __init__(self, adapter):
        self._adapter = adapter

    @asynccontextmanager
    async def acquire(self, _model):
        yield self._adapter

    def list_loaded(self):
        return []


def _build_app(store: BlobStore, adapter=None):
    app = FastAPI()
    app.state.store = store
    app.state.registry = MagicMock()
    app.state.registry.available_models.return_value = {
        "fake-clone-tts": {"latest": {"type": "tts"}}
    }
    app.state.scheduler = _DummyScheduler(adapter or _FakeCloneTTSAdapter())
    app.include_router(voices_router)
    app.include_router(bidi_router)
    return app


class TestReferenceAudioRetrieval:
    def test_returns_wav_bytes(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store,
            voice_id="abcd1234",
            name="Roy",
            audio_bytes=_sine_wav(),
            content_type="audio/wav",
        )
        client = TestClient(_build_app(store))
        resp = client.get("/v1/audio/voices/abcd1234/reference")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.headers["content-disposition"].endswith('"abcd1234.wav"')
        assert len(resp.content) > 100

    def test_openai_path_also_works(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store,
            voice_id="abcd1234",
            name="Roy",
            audio_bytes=_sine_wav(),
            content_type="audio/wav",
        )
        client = TestClient(_build_app(store))
        resp = client.get("/v1/audio/voices/abcd1234/reference")
        assert resp.status_code == 200

    def test_404_for_unknown_voice(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        client = TestClient(_build_app(store))
        resp = client.get("/v1/audio/voices/nonexistent/reference")
        assert resp.status_code == 404

    def test_422_on_silent_upload(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        client = TestClient(_build_app(store))
        silent = encode_wav(np.zeros(24_000, dtype=np.float32), 24_000)
        resp = client.post(
            "/v1/audio/voices",
            files={"audio_sample": ("silent.wav", io.BytesIO(silent), "audio/wav")},
            data={"name": "Silent"},
        )
        assert resp.status_code == 422
        assert "silent" in resp.json()["detail"].lower()


def _drain_until_done(ws, max_events: int = 100) -> list[dict]:
    """Iterate until we see a 'done' JSON message or disconnect. Bounded."""
    from starlette.websockets import WebSocketDisconnect

    json_msgs: list[dict] = []
    for _ in range(max_events):
        try:
            msg = ws.receive()
        except WebSocketDisconnect:
            break
        if msg.get("type") == "websocket.disconnect":
            break
        if "text" in msg and msg["text"] is not None:
            import json as _json
            payload = _json.loads(msg["text"])
            json_msgs.append(payload)
            if payload.get("type") == "done":
                break

    return json_msgs


class TestBidiTtsClonedVoice:
    def test_resolves_stored_voice_and_passes_reference_audio(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store,
            voice_id="roy12345",
            name="Roy",
            audio_bytes=_sine_wav(),
            content_type="audio/wav",
            language="en",
            reference_text="hello there",
        )
        adapter = _FakeCloneTTSAdapter()
        app = _build_app(store, adapter=adapter)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "fake-clone-tts:latest",
                "voice": "roy12345",
                "response_format": "pcm16",
                "language": None,
            })
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            ws.send_json({"type": "text", "text": "hello world"})
            ws.send_json({"type": "end"})

            events = _drain_until_done(ws)


        assert adapter.last_synth_kwargs is not None
        assert adapter.last_synth_kwargs["voice"] is None
        assert adapter.last_synth_kwargs["reference_audio"] is not None
        assert adapter.last_synth_kwargs["reference_text"] == "hello there"
        assert any(e.get("type") == "done" for e in events)

    def test_unknown_voice_name_passes_through_as_speaker(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        adapter = _FakeCloneTTSAdapter()
        app = _build_app(store, adapter=adapter)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "fake-clone-tts:latest",
                "voice": "af_heart",
                "response_format": "pcm16",
                "language": None,
            })
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "text", "text": "hi"})
            ws.send_json({"type": "end"})
            _drain_until_done(ws)

        assert adapter.last_synth_kwargs is not None
        assert adapter.last_synth_kwargs["voice"] == "af_heart"
        assert adapter.last_synth_kwargs["reference_audio"] is None

    def test_stored_voice_with_non_cloning_adapter_errors(self, tmp_path: Path):
        store = BlobStore(root=tmp_path)
        create_stored_voice(
            store,
            voice_id="roy12345",
            name="Roy",
            audio_bytes=_sine_wav(),
            content_type="audio/wav",
        )
        adapter = _FakeCloneTTSAdapter(supports_cloning=False)
        app = _build_app(store, adapter=adapter)
        client = TestClient(app)

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "fake-clone-tts:latest",
                "voice": "roy12345",
                "response_format": "pcm16",
                "language": None,
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "does not support cloned voices" in msg["message"]
