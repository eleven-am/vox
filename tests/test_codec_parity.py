from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vox.audio.codecs import encode_flac, encode_mp3, encode_opus, encode_pcm, encode_wav
from vox.audio.pipeline import prepare_for_output
from vox.core.adapter import TTSAdapter
from vox.core.store import BlobStore
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.server.routes.bidi import router as bidi_router
from vox.streaming.mp3 import Mp3StreamEncoder


def _sine(sr: int = 24_000, dur_s: float = 0.5, freq: float = 440.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestEncodeMp3:
    def test_produces_mp3_frames(self):
        audio = _sine()
        data = encode_mp3(audio, 24_000)
        assert len(data) > 500

        assert data[0] == 0xFF
        assert data[1] & 0xE0 == 0xE0

    def test_bitrate_kwarg_changes_size(self):
        audio = _sine(dur_s=1.0)
        low = encode_mp3(audio, 24_000, bitrate=64)
        high = encode_mp3(audio, 24_000, bitrate=192)
        assert len(high) > len(low)

    def test_contains_many_frames(self):
        """A 0.5s MP3 at 24kHz should contain multiple frame sync markers."""
        audio = _sine(dur_s=0.5)
        data = encode_mp3(audio, 24_000)

        sync_count = 0
        for i in range(len(data) - 1):
            if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
                sync_count += 1
        assert sync_count >= 5


class TestEncodeOpus:
    def test_produces_ogg_opus_container(self):
        audio = _sine()
        data = encode_opus(audio, 24_000)
        assert data[:4] == b"OggS"

    def test_roundtrip_via_soundfile(self):
        audio = _sine(dur_s=0.5)
        data = encode_opus(audio, 24_000)
        import soundfile as sf
        decoded, sr = sf.read(io.BytesIO(data), dtype="float32")
        assert sr == 24_000
        assert decoded.size > 0


class TestPrepareForOutput:
    @pytest.mark.parametrize("fmt,ct,min_size", [
        ("wav", "audio/wav", 1000),
        ("flac", "audio/flac", 500),
        ("pcm", "audio/L16", 1000),
        ("mp3", "audio/mpeg", 500),
        ("opus", "audio/opus", 500),
    ])
    def test_all_formats_produce_data(self, fmt, ct, min_size):
        audio = _sine(dur_s=0.5)
        data, content_type = prepare_for_output(audio, 24_000, fmt)
        assert content_type == ct
        assert len(data) >= min_size

    def test_unknown_format_raises(self):
        audio = _sine()
        with pytest.raises(ValueError, match="Unsupported output format"):
            prepare_for_output(audio, 24_000, "vorbis")


class TestCodecReturnTypes:
    @pytest.mark.parametrize(
        ("fn", "needs_rate"),
        [
            (encode_wav, True),
            (encode_flac, True),
            (encode_pcm, False),
            (encode_mp3, True),
            (encode_opus, True),
        ],
    )
    def test_encoders_return_bytes(self, fn, needs_rate):
        audio = _sine(dur_s=0.25)
        data = fn(audio, 24_000) if needs_rate else fn(audio)
        assert isinstance(data, bytes)


class TestMp3StreamEncoder:
    def test_single_chunk_produces_mp3(self):
        enc = Mp3StreamEncoder(source_rate=24_000)
        audio = _sine(dur_s=1.0)
        pcm = (audio * 32767).astype(np.int16).tobytes()
        body = enc.encode(pcm)
        tail = enc.flush()
        data = body + tail
        assert len(data) > 500
        assert data[0] == 0xFF
        assert data[1] & 0xE0 == 0xE0

    def test_multiple_chunks_concatenate_to_valid_mp3(self):
        enc = Mp3StreamEncoder(source_rate=24_000)
        audio = _sine(dur_s=2.0)
        pcm = (audio * 32767).astype(np.int16).tobytes()


        quarter = len(pcm) // 4
        data = b""
        for i in range(4):
            data += enc.encode(pcm[i * quarter : (i + 1) * quarter])
        data += enc.flush()


        assert 20_000 < len(data) < 60_000

        assert data[0] == 0xFF
        assert data[1] & 0xE0 == 0xE0

    def test_empty_encode_returns_empty_bytes(self):
        enc = Mp3StreamEncoder(source_rate=24_000)
        assert enc.encode(b"") == b""

    def test_flush_after_close_noop(self):
        enc = Mp3StreamEncoder(source_rate=24_000)
        enc.close()
        assert enc.flush() == b""
        assert enc.encode(b"\x00" * 100) == b""


class _MP3TTS(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="mp3-test-tts", type=ModelType.TTS,
            architectures=("test",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **_):
        sr = 24_000
        t = np.arange(sr // 2) / sr
        audio = (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        yield SynthesizeChunk(audio=audio.tobytes(), sample_rate=sr, is_final=False)
        yield SynthesizeChunk(audio=b"", sample_rate=sr, is_final=True)


class _DummyScheduler:
    def __init__(self, adapter):
        self._adapter = adapter

    @asynccontextmanager
    async def acquire(self, _model):
        yield self._adapter

    def list_loaded(self):
        return []


def _build_app(tmp_path: Path):
    app = FastAPI()
    app.state.store = BlobStore(root=tmp_path)
    reg = MagicMock()
    reg.available_models.return_value = {"mp3-test-tts": {"latest": {"type": "tts"}}}
    reg.resolve_model_ref.side_effect = lambda n, t, explicit_tag=False: (n, t or "latest")
    app.state.registry = reg
    app.state.scheduler = _DummyScheduler(_MP3TTS())
    app.include_router(bidi_router)
    return app


class TestBidiMp3Streaming:
    def test_mp3_accepted_and_bytes_are_valid_mp3(self, tmp_path: Path):
        client = TestClient(_build_app(tmp_path))
        mp3_chunks: list[bytes] = []

        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "mp3-test-tts:latest",
                "voice": "default",
                "response_format": "mp3",
            })
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["response_format"] == "mp3"

            ws.send_json({"type": "text", "text": "hello world"})
            ws.send_json({"type": "end"})

            for _ in range(50):
                try:
                    msg = ws.receive()
                except WebSocketDisconnect:
                    break
                if msg.get("type") == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"]:
                    mp3_chunks.append(msg["bytes"])
                    continue
                if "text" in msg and msg["text"]:
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "done":
                        break

        full = b"".join(mp3_chunks)
        assert len(full) > 500
        assert full[0] == 0xFF
        assert full[1] & 0xE0 == 0xE0

        sync_count = sum(
            1 for i in range(len(full) - 1)
            if full[i] == 0xFF and (full[i + 1] & 0xE0) == 0xE0
        )
        assert sync_count >= 5

    def test_unsupported_format_rejected(self, tmp_path: Path):
        client = TestClient(_build_app(tmp_path))
        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json({
                "type": "config",
                "model": "mp3-test-tts:latest",
                "voice": "default",
                "response_format": "wav",
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unsupported response_format" in msg["message"]
