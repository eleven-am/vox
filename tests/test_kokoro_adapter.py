from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

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


class _FakePipeline:
    inits = []

    def __init__(self, lang_code, model, device):
        self.lang_code = lang_code
        self.model = model
        self.device = device
        self.calls = []
        self._voices = ["af_heart", "bf_emma"]
        type(self).inits.append(
            {"lang_code": lang_code, "model": model, "device": device}
        )

    def __call__(self, text, *, voice, speed, split_pattern):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "split_pattern": split_pattern,
            }
        )
        yield "gs", "ps", np.array([0.0, 0.25, -0.25], dtype=np.float32)

    def get_voices(self):
        return self._voices


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


def _install_fake_native_modules():
    fake_kokoro = ModuleType("kokoro")
    fake_kokoro.KPipeline = _FakePipeline
    _FakePipeline.inits = []
    sys.modules["kokoro"] = fake_kokoro
    return fake_kokoro


def test_kokoro_load_supports_directory_layout(tmp_path: Path):
    fake_ort = _install_fake_modules()
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)
    sys.modules.pop("vox_kokoro.torch_adapter", None)

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
    sys.modules.pop("vox_kokoro.torch_adapter", None)

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
    sys.modules.pop("vox_kokoro.torch_adapter", None)

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
    sys.modules.pop("vox_kokoro.torch_adapter", None)

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
    sys.modules.pop("vox_kokoro.torch_adapter", None)

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


def test_kokoro_auto_falls_back_to_cpu_when_no_gpu_provider_is_available(tmp_path: Path):
    fake_ort = _install_fake_modules(providers=["CPUExecutionProvider"])
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)
    sys.modules.pop("vox_kokoro.torch_adapter", None)

    model_dir = tmp_path / "kokoro"
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "onnx" / "model.onnx").write_bytes(b"onnx")
    np.arange(512, dtype=np.float32).tofile(model_dir / "voices" / "af_heart.bin")

    from vox_kokoro.adapter import KokoroAdapter

    adapter = KokoroAdapter()
    adapter.load(str(model_dir), "auto")

    fake_ort.InferenceSession.assert_called_once()
    assert adapter._device == "cpu"


def test_kokoro_torch_loads_native_runtime_and_streams_audio(tmp_path: Path):
    fake_kokoro = _install_fake_native_modules()
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)
    sys.modules.pop("vox_kokoro.torch_adapter", None)

    model_dir = tmp_path / "kokoro"
    model_dir.mkdir(parents=True)
    (model_dir / "kokoro-v1_0.pth").write_bytes(b"weights")

    from vox_kokoro.torch_adapter import KokoroTorchAdapter

    adapter = KokoroTorchAdapter()
    with patch.object(KokoroTorchAdapter, "_import_runtime", return_value=fake_kokoro):
        adapter.load(str(model_dir), "cpu")

    assert adapter.is_loaded is True
    assert _FakePipeline.inits == [
        {"lang_code": "a", "model": str(model_dir / "kokoro-v1_0.pth"), "device": "cpu"}
    ]
    voice_ids = [voice.id for voice in adapter.list_voices()]
    assert voice_ids[:4] == ["af_alloy", "af_aoede", "af_bella", "af_heart"]
    assert "bf_emma" in voice_ids

    async def _collect():
        return [
            chunk
            async for chunk in adapter.synthesize("Hello world", voice="bf_emma")
        ]

    with patch.object(KokoroTorchAdapter, "_import_runtime", return_value=fake_kokoro):
        chunks = asyncio.run(_collect())

    assert len(chunks) == 2
    assert chunks[0].sample_rate == 24000
    assert chunks[0].audio
    assert chunks[1].is_final is True
    assert adapter._pipelines["b"].calls == [
        {
            "text": "Hello world",
            "voice": "bf_emma",
            "speed": 1.0,
            "split_pattern": r"\n+",
        }
    ]
    assert _FakePipeline.inits[-1] == {
        "lang_code": "b",
        "model": str(model_dir / "kokoro-v1_0.pth"),
        "device": "cpu",
    }


def test_kokoro_torch_honors_explicit_language_override(tmp_path: Path):
    fake_kokoro = _install_fake_native_modules()
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)
    sys.modules.pop("vox_kokoro.torch_adapter", None)

    model_dir = tmp_path / "kokoro"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pth").write_bytes(b"weights")

    from vox_kokoro.torch_adapter import KokoroTorchAdapter

    adapter = KokoroTorchAdapter()
    with patch.object(KokoroTorchAdapter, "_import_runtime", return_value=fake_kokoro):
        adapter.load(str(model_dir), "cpu")

    async def _collect():
        return [
            chunk
            async for chunk in adapter.synthesize("Hello world", voice="af_heart", language="en-gb")
        ]

    with patch.object(KokoroTorchAdapter, "_import_runtime", return_value=fake_kokoro):
        chunks = asyncio.run(_collect())

    assert len(chunks) == 2
    assert adapter._pipelines["b"].calls == [
        {
            "text": "Hello world",
            "voice": "af_heart",
            "speed": 1.0,
            "split_pattern": r"\n+",
        }
    ]


def test_kokoro_torch_list_voices_falls_back_to_official_presets(tmp_path: Path):
    fake_kokoro = _install_fake_native_modules()
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)
    sys.modules.pop("vox_kokoro.torch_adapter", None)

    model_dir = tmp_path / "kokoro"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pth").write_bytes(b"weights")

    from vox_kokoro.torch_adapter import KokoroTorchAdapter

    adapter = KokoroTorchAdapter()
    with patch.object(KokoroTorchAdapter, "_import_runtime", return_value=fake_kokoro):
        adapter.load(str(model_dir), "cpu")

    adapter._pipelines["a"].get_voices = MagicMock(return_value=[])
    adapter._pipelines["a"].voices = {}

    voice_ids = [voice.id for voice in adapter.list_voices()]
    assert voice_ids[:4] == ["af_alloy", "af_aoede", "af_bella", "af_heart"]
    assert "zm_yunyang" in voice_ids


def test_kokoro_torch_requires_runtime_package(tmp_path: Path):
    sys.modules.pop("kokoro", None)
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)
    sys.modules.pop("vox_kokoro.torch_adapter", None)

    model_dir = tmp_path / "kokoro"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pth").write_bytes(b"weights")

    from vox_kokoro.torch_adapter import KokoroTorchAdapter

    adapter = KokoroTorchAdapter()

    with (
        patch(
            "vox_kokoro.torch_adapter._install_runtime",
            side_effect=RuntimeError("official 'kokoro' runtime package"),
        ),
        pytest.raises(RuntimeError, match="official 'kokoro' runtime package"),
    ):
        adapter.load(str(model_dir), "cpu")


def test_kokoro_torch_clears_conflicting_hub_modules():
    sys.modules["huggingface_hub"] = ModuleType("huggingface_hub")
    sys.modules["huggingface_hub.file_download"] = ModuleType("huggingface_hub.file_download")
    sys.modules["tokenizers"] = ModuleType("tokenizers")
    sys.modules["safetensors"] = ModuleType("safetensors")
    sys.modules["spacy"] = ModuleType("spacy")
    sys.modules["spacy.cli"] = ModuleType("spacy.cli")
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)
    sys.modules.pop("vox_kokoro.torch_adapter", None)

    from vox_kokoro.torch_adapter import _clear_runtime_modules

    _clear_runtime_modules()

    assert "huggingface_hub" not in sys.modules
    assert "huggingface_hub.file_download" not in sys.modules
    assert "tokenizers" not in sys.modules
    assert "safetensors" not in sys.modules
    assert "spacy" not in sys.modules
    assert "spacy.cli" not in sys.modules


def test_kokoro_torch_runtime_bootstrap_installs_spacy_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    sys.modules.pop("vox_kokoro", None)
    sys.modules.pop("vox_kokoro.adapter", None)
    sys.modules.pop("vox_kokoro.torch_adapter", None)

    from vox_kokoro import torch_adapter as torch_adapter_module

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs):
        calls.append(cmd)
        return MagicMock(returncode=0, stderr="")

    spacy_model_checks = iter([False, True])
    with (
        patch("vox_kokoro.torch_adapter.subprocess.run", side_effect=_fake_run),
        patch(
            "vox_kokoro.torch_adapter._spacy_model_installed",
            side_effect=lambda runtime_root: next(spacy_model_checks),
        ),
    ):
        torch_adapter_module._install_runtime()

    assert len(calls) == 2
    assert "kokoro>=0.9.4,<1.0.0" in calls[0]
    assert any("en_core_web_sm-3.8.0-py3-none-any.whl" in part for part in calls[1])
