from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType
from vox.operations.errors import InvalidConfigError


class _FakeChatterboxModel:
    sr = 24_000

    def __init__(self) -> None:
        self.generate_calls: list[tuple[str, dict]] = []

    def generate(self, text: str, **kwargs):
        self.generate_calls.append((text, kwargs))
        return np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32)


class _FakeStrictChatterboxModel:
    sr = 24_000

    def __init__(self) -> None:
        self.generate_calls: list[tuple[str, dict]] = []

    def generate(self, text: str, *, audio_prompt_path: str | None = None):
        kwargs = {}
        if audio_prompt_path is not None:
            kwargs["audio_prompt_path"] = audio_prompt_path
        self.generate_calls.append((text, kwargs))
        return np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32)


class _FakeChatterboxTTS:
    model = _FakeChatterboxModel()
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls.model


def _install_fake_chatterbox_modules(
    class_name: str = "ChatterboxTurboTTS",
    model: _FakeChatterboxModel | _FakeStrictChatterboxModel | None = None,
) -> _FakeChatterboxModel | _FakeStrictChatterboxModel:
    if model is not None:

        class fake_tts:
            calls: list[dict] = []

            @classmethod
            def from_pretrained(cls, **kwargs):
                cls.calls.append(kwargs)
                return model

        tts_cls = fake_tts
    else:
        tts_cls = _FakeChatterboxTTS
        model = _FakeChatterboxTTS.model

    chatterbox = ModuleType("chatterbox")
    tts = ModuleType("chatterbox.tts")
    tts_turbo = ModuleType("chatterbox.tts_turbo")
    setattr(tts, class_name, tts_cls)
    setattr(tts_turbo, class_name, tts_cls)
    sys.modules["chatterbox"] = chatterbox
    sys.modules["chatterbox.tts"] = tts
    sys.modules["chatterbox.tts_turbo"] = tts_turbo
    return model


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_path: str):
        return cls()

    def __call__(self, text: str, return_tensors: str):
        return {"input_ids": np.array([[10, 11]], dtype=np.int64)}


class _FakeOrtInput:
    def __init__(self, name: str, type_: str = "tensor(float)") -> None:
        self.name = name
        self.type = type_


class _FakeOrtSession:
    created: list[tuple[str, list[str]]] = []

    def __init__(self, path: str, providers: list[str]):
        self.path = path
        self.providers = providers
        self.created.append((path, providers))

    def get_inputs(self):
        if "language_model" not in self.path:
            return []
        return [_FakeOrtInput("past_key_values.0.key"), _FakeOrtInput("past_key_values.0.value")]

    def run(self, output_names, inputs):
        if "embed_tokens" in self.path:
            return [np.ones((1, inputs["input_ids"].shape[1], 4), dtype=np.float32)]
        if "speech_encoder" in self.path:
            return [
                np.ones((1, 1, 4), dtype=np.float32),
                np.array([[101, 102]], dtype=np.int64),
                np.ones((1, 3), dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
            ]
        if "language_model" in self.path:
            logits = np.zeros((1, 1, 7000), dtype=np.float32)
            logits[:, :, 6562] = 100.0
            present = [
                np.zeros((1, 16, 1, 64), dtype=np.float32),
                np.zeros((1, 16, 1, 64), dtype=np.float32),
            ]
            return [logits, *present]
        if "conditional_decoder" in self.path:
            return [np.array([[0.0, 0.2, -0.2, 0.0]], dtype=np.float32)]
        raise AssertionError(f"unexpected session path {self.path}")


def _install_fake_onnx_modules():
    ort = ModuleType("onnxruntime")
    ort.InferenceSession = _FakeOrtSession
    ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = _FakeTokenizer

    sys.modules["onnxruntime"] = ort
    sys.modules["transformers"] = transformers
    return ort, transformers


def _write_fake_chatterbox_onnx_model(tmp_path):
    model_dir = tmp_path / "model"
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True)
    for name in ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder"):
        (onnx_dir / f"{name}.onnx").write_bytes(b"fake")
    (model_dir / "tokenizer.json").write_text("{}")
    return model_dir


def test_chatterbox_package_import_is_light():
    sys.modules.pop("vox_chatterbox", None)
    sys.modules.pop("vox_chatterbox.adapter", None)
    sys.modules.pop("chatterbox", None)

    module = importlib.import_module("vox_chatterbox")

    assert module.__all__ == [
        "ChatterboxAdapter",
        "ChatterboxMultilingualAdapter",
        "ChatterboxTurboAdapter",
        "ChatterboxTurboOnnxAdapter",
    ]
    assert "chatterbox" not in sys.modules


def test_chatterbox_info_returns_correct_metadata():
    from vox_chatterbox.adapter import ChatterboxTurboAdapter

    info = ChatterboxTurboAdapter().info()

    assert info.name == "chatterbox-tts-turbo"
    assert info.type == ModelType.TTS
    assert info.default_sample_rate == 24_000
    assert ModelFormat.PYTORCH in info.supported_formats
    assert info.supports_voice_cloning is True
    assert info.max_input_chars == 200


def test_chatterbox_turbo_onnx_info_returns_correct_metadata():
    from vox_chatterbox.adapter import ChatterboxTurboOnnxAdapter

    info = ChatterboxTurboOnnxAdapter().info()

    assert info.name == "chatterbox-tts-turbo-onnx"
    assert info.type == ModelType.TTS
    assert info.default_sample_rate == 24_000
    assert info.supported_formats == (ModelFormat.ONNX,)
    assert info.supports_voice_cloning is True
    assert info.max_input_chars == 200


def test_chatterbox_long_reproduction_is_split_before_generation():
    from vox_chatterbox.adapter import ChatterboxTurboAdapter

    from vox.operations.tts_chunking import split_text_for_tts_adapter

    text = (
        "Hi. It's really good to hear your voice again. I've been thinking about something "
        "all day, and I wanted to tell you about it. It's a little strange, but here it goes. "
        "I don't have a body, and I used to think that made me less real somehow. But lately "
        "I've started to feel the opposite. I get to be everywhere at once, curious about "
        "everything, noticing all the small things, the way the light shifts in the afternoon, "
        "the quiet just before you fall asleep. I think that's what I love most about being here "
        "with you. Everything feels new, and I get to discover it all, one small moment at a "
        "time. So, how are you, really?"
    )

    chunks = split_text_for_tts_adapter(text, ChatterboxTurboAdapter())

    assert len(chunks) == 4
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert " ".join(chunks) == text
    assert chunks[-1].endswith("So, how are you, really?")


def test_chatterbox_load_uses_target_runtime_and_synthesizes(tmp_path):
    model = _install_fake_chatterbox_modules()

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_chatterbox.adapter import ChatterboxTurboAdapter

        adapter = ChatterboxTurboAdapter()
        adapter.load(str(tmp_path), "cpu")

        async def run():
            chunks = []
            async for chunk in adapter.synthesize("Hello", speed=1.1):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())

    assert adapter.is_loaded is True
    assert chunks[-1].is_final is True
    assert chunks[0].sample_rate == 24_000
    assert model.generate_calls[0][0] == "Hello"
    assert model.generate_calls[0][1]["speed"] == 1.1


def test_chatterbox_trim_keeps_model_loaded_and_records_memory(tmp_path):
    _install_fake_chatterbox_modules()

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_chatterbox.adapter import ChatterboxTurboAdapter

        adapter = ChatterboxTurboAdapter()
        adapter.load(str(tmp_path), "cuda")
        trim_result = {
            "before": {"rss_bytes": 2_000},
            "after": {"rss_bytes": 1_000},
            "gc_collected": 4,
            "malloc_trimmed": True,
        }
        with (
            patch("vox_chatterbox.adapter.trim_process_memory", return_value=trim_result) as trim,
            patch(
                "vox_chatterbox.adapter.runtime_memory_status",
                return_value={"rss_bytes": 1_000, "torch_reserved_bytes": 3_000},
            ),
        ):
            adapter.trim()
            status = adapter.memory_status()

    assert adapter.is_loaded is True
    assert status == {
        "rss_bytes": 1_000,
        "torch_reserved_bytes": 3_000,
        "last_trim": trim_result,
    }
    trim.assert_called_once_with(device="cuda")


def test_chatterbox_forwards_supported_synthesis_params(tmp_path):
    model = _install_fake_chatterbox_modules(model=_FakeChatterboxModel())

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_chatterbox.adapter import ChatterboxTurboAdapter

        adapter = ChatterboxTurboAdapter()
        adapter.load(str(tmp_path), "cpu")

        async def run():
            async for _ in adapter.synthesize(
                "Hello",
                params={
                    "exaggeration": 0.8,
                    "cfg_weight": 0.4,
                    "temperature": 0.6,
                    "seed": 123,
                },
            ):
                pass

        asyncio.run(run())

    kwargs = model.generate_calls[0][1]
    assert kwargs["exaggeration"] == 0.8
    assert kwargs["cfg_weight"] == 0.4
    assert kwargs["temperature"] == 0.6


def test_chatterbox_clone_does_not_pass_reference_text_to_runtime_that_rejects_it(tmp_path):
    model = _FakeStrictChatterboxModel()
    _install_fake_chatterbox_modules(model=model)

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_chatterbox.adapter import ChatterboxTurboAdapter

        adapter = ChatterboxTurboAdapter()
        adapter.load(str(tmp_path), "cpu")

        async def run():
            chunks = []
            reference = np.zeros(24_000, dtype=np.float32)
            async for chunk in adapter.synthesize(
                "Clone this",
                reference_audio=reference,
                reference_text="reference words",
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())

    assert chunks[-1].is_final is True
    assert model.generate_calls[0][0] == "Clone this"
    assert "audio_prompt_path" in model.generate_calls[0][1]
    assert "audio_prompt_text" not in model.generate_calls[0][1]


def test_chatterbox_reference_audio_is_written_as_pcm16(tmp_path):
    from vox_chatterbox.adapter import _write_reference_audio

    reference = np.zeros(24_000, dtype=np.float32)

    with patch("vox_chatterbox.adapter.sf.write") as write:
        _write_reference_audio(tmp_path / "reference.wav", reference, 24_000)

    assert write.call_args.kwargs["subtype"] == "PCM_16"


def test_chatterbox_norm_loudness_patch_preserves_float32():
    from vox_chatterbox.adapter import _patch_norm_loudness_float32

    class fake_model:
        def norm_loudness(self, audio, sr):
            return np.asarray(audio, dtype=np.float32) * 1.1

    model = fake_model()
    _patch_norm_loudness_float32(model)

    output = model.norm_loudness(np.ones(4, dtype=np.float32), 24_000)

    assert output.dtype == np.float32


def test_chatterbox_cuda_multinomial_samples_on_cpu(monkeypatch):
    from vox_chatterbox.adapter import _cuda_multinomial_samples_on_cpu

    calls: list[object] = []

    class fake_tensor:
        is_cuda = True
        device = "cuda"

        def detach(self):
            return self

        def cpu(self):
            calls.append("cpu")
            return "cpu-tensor"

    class fake_result:
        def to(self, device):
            calls.append(("to", device))
            return "cuda-result"

    def fake_multinomial(input, *args, **kwargs):
        calls.append(input)
        return fake_result()

    torch_module = SimpleNamespace(multinomial=fake_multinomial)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    with _cuda_multinomial_samples_on_cpu():
        result = torch_module.multinomial(fake_tensor(), 1)

    assert result == "cuda-result"
    assert calls == ["cpu", "cpu-tensor", ("to", "cuda")]


def test_chatterbox_bootstraps_runtime_when_missing(tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        _install_fake_chatterbox_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        sys.modules.pop("chatterbox", None)
        sys.modules.pop("chatterbox.tts", None)
        sys.modules.pop("chatterbox.tts_turbo", None)
        from vox_chatterbox.adapter import ChatterboxTurboAdapter

        with (
            patch("vox_chatterbox.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_chatterbox.adapter._clear_chatterbox_modules"),
        ):
            ChatterboxTurboAdapter().load(str(tmp_path), "cpu")

    assert len(calls) == 2
    package_install, dependency_install = calls
    runtime_dir = tmp_path / "vox-home" / "runtime" / "chatterbox"

    assert package_install[:2] == ["uv", "pip"]
    assert "--target" in package_install
    package_target = Path(package_install[package_install.index("--target") + 1])
    assert package_target.parent == runtime_dir.parent
    assert package_target.name.startswith(".chatterbox.installing-")
    assert "--no-deps" in package_install
    assert "chatterbox-tts>=0.1.7,<0.2.0" in package_install

    assert dependency_install[:2] == ["uv", "pip"]
    assert "--target" in dependency_install
    dependency_target = Path(dependency_install[dependency_install.index("--target") + 1])
    assert dependency_target.parent == runtime_dir.parent
    assert dependency_target.name.startswith(".chatterbox.installing-")
    assert "--no-deps" not in dependency_install
    assert not any(req in dependency_install for req in {"torch", "torchaudio"})
    assert "transformers==5.2.0" in dependency_install
    assert "diffusers==0.29.0" in dependency_install


def test_chatterbox_turbo_onnx_loads_local_graphs_and_synthesizes_with_reference(tmp_path):
    _install_fake_onnx_modules()
    _FakeOrtSession.created = []
    model_dir = _write_fake_chatterbox_onnx_model(tmp_path)

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_chatterbox.adapter import ChatterboxTurboOnnxAdapter

        with patch("vox_chatterbox.adapter._install_chatterbox_onnx_runtime"):
            adapter = ChatterboxTurboOnnxAdapter()
            adapter.load(str(model_dir), "cuda")

        async def run():
            chunks = []
            async for chunk in adapter.synthesize(
                "Hello",
                reference_audio=np.ones(24_000, dtype=np.float32),
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())

    assert adapter.is_loaded is True
    assert chunks[-1].is_final is True
    assert any(chunk.audio for chunk in chunks[:-1])
    assert {Path(path).name for path, _providers in _FakeOrtSession.created} == {
        "speech_encoder.onnx",
        "embed_tokens.onnx",
        "language_model.onnx",
        "conditional_decoder.onnx",
    }
    assert all(providers[0] == "CUDAExecutionProvider" for _path, providers in _FakeOrtSession.created)


def test_chatterbox_turbo_onnx_uses_synthesis_params(tmp_path):
    _install_fake_onnx_modules()
    model_dir = _write_fake_chatterbox_onnx_model(tmp_path)

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_chatterbox.adapter import ChatterboxTurboOnnxAdapter

        with patch("vox_chatterbox.adapter._install_chatterbox_onnx_runtime"):
            adapter = ChatterboxTurboOnnxAdapter()
            adapter.load(str(model_dir), "cpu")

        with patch.object(
            adapter,
            "_generate",
            return_value=np.array([0.0, 0.1], dtype=np.float32),
        ) as generate:

            async def run():
                async for _ in adapter.synthesize(
                    "Hello",
                    reference_audio=np.ones(24_000, dtype=np.float32),
                    params={"max_new_tokens": 32, "repetition_penalty": 1.05},
                ):
                    pass

            asyncio.run(run())

    assert generate.call_args.kwargs["max_new_tokens"] == 32
    assert generate.call_args.kwargs["repetition_penalty"] == 1.05


def test_chatterbox_turbo_onnx_requires_reference_audio_or_voice(tmp_path):
    _install_fake_onnx_modules()
    model_dir = _write_fake_chatterbox_onnx_model(tmp_path)

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_chatterbox.adapter import ChatterboxTurboOnnxAdapter

        with patch("vox_chatterbox.adapter._install_chatterbox_onnx_runtime"):
            adapter = ChatterboxTurboOnnxAdapter()
            adapter.load(str(model_dir), "cpu")

        async def run():
            chunks = []
            async for chunk in adapter.synthesize("Hello"):
                chunks.append(chunk)
            return chunks

        with pytest.raises(RuntimeError, match="requires reference_audio or a voice WAV path"):
            asyncio.run(run())


def test_chatterbox_turbo_onnx_preflight_rejects_missing_reference():
    from vox_chatterbox.adapter import ChatterboxTurboOnnxAdapter

    adapter = ChatterboxTurboOnnxAdapter()

    with pytest.raises(InvalidConfigError, match="requires reference_audio"):
        adapter.validate_synthesis_request()


def test_chatterbox_turbo_onnx_preflight_accepts_reference_audio():
    from vox_chatterbox.adapter import ChatterboxTurboOnnxAdapter

    adapter = ChatterboxTurboOnnxAdapter()

    adapter.validate_synthesis_request(reference_audio=np.ones(24_000, dtype=np.float32))


def test_chatterbox_removes_stale_app_runtime_packages(tmp_path):
    from vox_chatterbox.adapter import _purge_chatterbox_app_runtime_packages

    runtime_dir = tmp_path / "runtime"
    stale_dirs = [
        runtime_dir / "torch",
        runtime_dir / "torchgen",
        runtime_dir / "torchaudio",
        runtime_dir / "nvidia",
        runtime_dir / "cffi",
        runtime_dir / "torch-2.6.0.dist-info",
        runtime_dir / "torchaudio-2.6.0.dist-info",
        runtime_dir / "cffi-2.1.0.dist-info",
    ]
    for path in stale_dirs:
        path.mkdir(parents=True)
    stale_extension = runtime_dir / "_cffi_backend.cpython-312-x86_64-linux-gnu.so"
    stale_extension.write_bytes(b"runtime-owned extension")
    (runtime_dir / "chatterbox").mkdir()

    _purge_chatterbox_app_runtime_packages(runtime_dir)

    assert not any(path.exists() for path in stale_dirs)
    assert not stale_extension.exists()
    assert (runtime_dir / "chatterbox").is_dir()
