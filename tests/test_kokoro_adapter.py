from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest


class _FakeInput:
    def __init__(self, name, type_):
        self.name = name
        self.type = type_
        self.shape = None


class _FakeSession:
    def __init__(self, model_path="fake-model.onnx", inputs=None):
        self._model_path = model_path
        self._inputs = inputs or []

    def get_inputs(self):
        return self._inputs

    def run(self, outputs, inputs):
        self.last_run = inputs
        return [np.array([0.0, 0.25, -0.25], dtype=np.float32)]


class _FakeKokoro:
    def __init__(self):
        self.stream_calls = []

    @classmethod
    def from_session(cls, session, voices_path):
        instance = cls()
        instance.sess = session
        instance.voices_path = voices_path
        instance.voices = {"legacy": np.zeros((1, 1, 256), dtype=np.float32)}
        return instance

    async def create_stream(self, text, voice, *, lang, speed):
        calls = getattr(self, "stream_calls", [])
        calls.append(
            {"text": text, "voice": voice, "lang": lang, "speed": speed}
        )
        self.stream_calls = calls
        yield np.array([0.0, 0.25, -0.25], dtype=np.float32), 24000

    def get_voices(self):
        return list(self.voices.keys())


class _FakeTokenizer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def tokenize(self, phonemes):
        return list(range(len(phonemes)))


class _FakeConfig:
    def __init__(self, model_path, voices_path, espeak_config=None):
        self.model_path = model_path
        self.voices_path = voices_path
        self.espeak_config = espeak_config


def _install_fake_modules(*, providers=None):
    fake_kokoro = ModuleType("kokoro_onnx")
    fake_kokoro.Kokoro = _FakeKokoro
    fake_kokoro.Tokenizer = _FakeTokenizer
    fake_kokoro.KoKoroConfig = _FakeConfig

    fake_ort = ModuleType("onnxruntime")
    fake_ort.InferenceSession = MagicMock(return_value=_FakeSession())
    fake_ort.get_available_providers = MagicMock(
        return_value=providers or ["CPUExecutionProvider"]
    )

    sys.modules["kokoro_onnx"] = fake_kokoro
    sys.modules["onnxruntime"] = fake_ort
    return fake_ort


def test_kokoro_load_supports_directory_layout(tmp_path: Path):
    fake_ort = _install_fake_modules()
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)

    model_dir = tmp_path / "kokoro"
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "onnx" / "model.onnx").write_bytes(b"onnx")
    np.arange(512, dtype=np.float32).tofile(model_dir / "voices" / "af_heart.bin")
    np.arange(768, dtype=np.float32).tofile(model_dir / "voices" / "bf_emma.bin")

    from vox_kokoro.adapter import KokoroAdapter

    adapter = KokoroAdapter()
    adapter.load(str(model_dir), "cpu")

    fake_ort.InferenceSession.assert_called_once()
    assert adapter.is_loaded is True
    assert sorted(adapter._kokoro.voices.keys()) == ["af_heart", "bf_emma"]
    assert adapter._kokoro.voices["af_heart"].shape == (2, 1, 256)
    assert [voice.id for voice in adapter.list_voices()] == ["af_heart", "bf_emma"]


def test_kokoro_load_supports_legacy_layout(tmp_path: Path):
    fake_ort = _install_fake_modules()
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)

    model_dir = tmp_path / "kokoro"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"onnx")
    (model_dir / "voices.bin").write_bytes(b"legacy")

    from vox_kokoro.adapter import KokoroAdapter

    adapter = KokoroAdapter()
    adapter.load(str(model_dir), "cpu")

    fake_ort.InferenceSession.assert_called_once()
    assert adapter.is_loaded is True
    assert adapter._kokoro.voices_path.endswith("voices.bin")


def test_kokoro_synthesize_uses_bcp47_language_tags(tmp_path: Path):
    _install_fake_modules()
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)

    model_dir = tmp_path / "kokoro"
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "onnx" / "model.onnx").write_bytes(b"onnx")
    np.arange(512, dtype=np.float32).tofile(model_dir / "voices" / "af_heart.bin")

    from vox_kokoro.adapter import KokoroAdapter

    adapter = KokoroAdapter()
    adapter.load(str(model_dir), "cpu")

    async def _collect():
        return [
            chunk
            async for chunk in adapter.synthesize("Hello world", voice="af_heart")
        ]

    chunks = asyncio.run(_collect())

    assert len(chunks) == 2
    assert adapter._kokoro.stream_calls == [
        {
            "text": "Hello world",
            "voice": "af_heart",
            "lang": "en-us",
            "speed": 1.0,
        }
    ]


def test_kokoro_patches_float_speed_runtime(tmp_path: Path):
    fake_ort = _install_fake_modules()
    fake_ort.InferenceSession = MagicMock(
        return_value=_FakeSession(
            model_path="fake-model.onnx",
            inputs=[
                _FakeInput("input_ids", "tensor(int64)"),
                _FakeInput("style", "tensor(float)"),
                _FakeInput("speed", "tensor(float)"),
            ],
        )
    )
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)

    model_dir = tmp_path / "kokoro"
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "onnx" / "model.onnx").write_bytes(b"onnx")
    np.arange(512, dtype=np.float32).tofile(model_dir / "voices" / "af_heart.bin")

    from vox_kokoro.adapter import KokoroAdapter

    adapter = KokoroAdapter()
    adapter.load(str(model_dir), "cpu")

    result_audio, sample_rate = adapter._kokoro._create_audio("abc", np.zeros((8, 1, 256), dtype=np.float32), 1.25)

    assert sample_rate == 24000
    assert result_audio.shape == (3,)
    assert adapter._kokoro.sess.last_run["speed"].dtype == np.float32


def test_kokoro_rejects_cuda_when_no_gpu_provider_is_available(tmp_path: Path):
    fake_ort = _install_fake_modules(providers=["CPUExecutionProvider"])
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)

    model_dir = tmp_path / "kokoro"
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "onnx" / "model.onnx").write_bytes(b"onnx")
    np.arange(512, dtype=np.float32).tofile(model_dir / "voices" / "af_heart.bin")

    from vox_kokoro.adapter import KokoroAdapter

    adapter = KokoroAdapter()

    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        adapter.load(str(model_dir), "cuda")

    fake_ort.InferenceSession.assert_not_called()
