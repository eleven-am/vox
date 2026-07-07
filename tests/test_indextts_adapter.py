from __future__ import annotations

import asyncio
import builtins
import importlib
import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from vox.core.types import ModelFormat, ModelType
from vox.operations.errors import InvalidConfigError


def test_indextts_package_metadata_is_lightweight():
    pyproject = Path(__file__).parents[1] / "adapters" / "vox-indextts" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    dependencies = data["project"]["dependencies"]
    assert data["project"]["version"] == "0.1.21"
    assert not any(dep.startswith(("torch", "torchaudio", "indextts")) for dep in dependencies)


class _FakeIndexTTS2:
    instances: list[_FakeIndexTTS2] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.infer_calls: list[dict] = []
        _FakeIndexTTS2.instances.append(self)

    def infer(self, **kwargs):
        self.infer_calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        sf.write(output_path, np.array([0.0, 0.5, -0.5], dtype=np.float32), 24_000)
        return str(output_path)


def _indextts_runtime_file() -> str:
    runtime_root = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox"))) / "runtime" / "indextts"
    return str(runtime_root / "indextts" / "infer_v2.py")


def _install_fake_indextts_modules(*, module_file: str | None = None) -> None:
    package = ModuleType("indextts")
    infer_v2 = ModuleType("indextts.infer_v2")
    package.__file__ = str(Path(module_file or _indextts_runtime_file()).parent / "__init__.py")
    infer_v2.__file__ = module_file or _indextts_runtime_file()
    infer_v2.IndexTTS2 = _FakeIndexTTS2
    sys.modules["indextts"] = package
    sys.modules["indextts.infer_v2"] = infer_v2


def _install_stale_indextts_modules(*, module_file: str | None = None) -> None:
    package = ModuleType("indextts")
    infer_v2 = ModuleType("indextts.infer_v2")
    package.__file__ = str(Path(module_file or _indextts_runtime_file()).parent / "__init__.py")
    infer_v2.__file__ = module_file or _indextts_runtime_file()
    sys.modules["indextts"] = package
    sys.modules["indextts.infer_v2"] = infer_v2


def test_indextts_package_import_is_light():
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        heavy_modules = ("indextts", "torch", "torchaudio")
        if name in heavy_modules or name.startswith(tuple(f"{module}." for module in heavy_modules)):
            raise AssertionError(f"vox-indextts imported heavy runtime dependency during package import: {name}")
        return original_import(name, *args, **kwargs)

    sys.modules.pop("vox_indextts", None)
    sys.modules.pop("vox_indextts.adapter", None)
    sys.modules.pop("indextts", None)

    with patch.object(builtins, "__import__", side_effect=guarded_import):
        module = importlib.import_module("vox_indextts")

    assert module.__all__ == ["IndexTTSAdapter"]
    assert "indextts" not in sys.modules


def test_indextts_info_returns_correct_metadata():
    from vox_indextts.adapter import IndexTTSAdapter

    info = IndexTTSAdapter().info()

    assert info.name == "indextts-tts-torch"
    assert info.type == ModelType.TTS
    assert info.default_sample_rate == 24_000
    assert ModelFormat.PYTORCH in info.supported_formats
    assert info.supports_voice_cloning is True


def test_indextts_synthesis_parameters_expose_emotion_controls():
    from vox_indextts.adapter import IndexTTSAdapter

    params = {param.name: param for param in IndexTTSAdapter().synthesis_parameters()}

    assert {
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "num_beams",
        "repetition_penalty",
        "length_penalty",
        "max_mel_tokens",
        "max_text_tokens_per_segment",
        "emo_alpha",
        "use_emo_text",
        "emo_text",
        "emo_audio_prompt",
        "use_random",
        "emotion_happy",
        "emotion_angry",
        "emotion_sad",
        "emotion_afraid",
        "emotion_disgusted",
        "emotion_melancholic",
        "emotion_surprised",
        "emotion_calm",
    } == set(params)
    assert params["do_sample"].type == "boolean"
    assert params["temperature"].default == 0.8
    assert params["top_p"].max_value == 1.0
    assert params["top_k"].type == "integer"
    assert params["num_beams"].max_value == 10
    assert params["repetition_penalty"].max_value == 20.0
    assert params["length_penalty"].min_value == -2.0
    assert params["max_mel_tokens"].default == 1500
    assert params["max_text_tokens_per_segment"].default == 120
    assert params["emo_alpha"].min_value == 0.0
    assert params["emo_alpha"].max_value == 1.0
    assert params["use_emo_text"].type == "boolean"
    assert params["emo_text"].type == "string"
    assert params["emo_audio_prompt"].type == "string"
    assert params["use_random"].default is False
    assert params["emotion_sad"].max_value == 1.0


def test_indextts_readme_uses_public_model_reference():
    readme = Path(__file__).parents[1] / "adapters" / "vox-indextts" / "README.md"
    text = readme.read_text(encoding="utf-8")

    assert "vox pull indextts-tts:2" in text
    assert "vox run indextts-tts:2" in text
    assert "vox pull indextts-tts-torch:2" not in text


def test_indextts_load_rejects_cpu_before_runtime_install(tmp_path):
    from vox_indextts.adapter import IndexTTSAdapter

    with (
        patch("vox_indextts.adapter._load_indextts_class") as load_indextts_class,
        pytest.raises(RuntimeError, match="requires a Linux x86_64 CUDA runtime"),
    ):
        IndexTTSAdapter().load(str(tmp_path), "cpu")

    load_indextts_class.assert_not_called()


def test_indextts_load_uses_config_and_synthesizes_with_reference_audio(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("fake: true\n")
    emo_audio = tmp_path / "emotion.wav"
    sf.write(emo_audio, np.array([0.0, 0.25, -0.25], dtype=np.float32), 24_000)

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_indextts_modules()
        from vox_indextts.adapter import IndexTTSAdapter

        adapter = IndexTTSAdapter()
        with patch("vox_indextts.adapter._clear_indextts_modules"):
            adapter.load(str(model_dir), "cuda")

        async def run():
            chunks = []
            async for chunk in adapter.synthesize(
                "Hello",
                reference_audio=np.zeros(2400, dtype=np.float32),
                params={
                    "emo_alpha": 0.6,
                    "emo_text": "calm but curious",
                    "emo_audio_prompt": str(emo_audio),
                    "emotion_happy": 0.2,
                    "emotion_calm": 0.7,
                    "use_random": False,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.85,
                    "top_k": 40,
                    "num_beams": 2,
                    "repetition_penalty": 8.5,
                    "length_penalty": -0.1,
                    "max_mel_tokens": 1200,
                    "max_text_tokens_per_segment": 96,
                },
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())

    instance = _FakeIndexTTS2.instances[-1]
    assert instance.kwargs["cfg_path"].endswith("config.yaml")
    assert instance.kwargs["model_dir"] == str(model_dir)
    assert instance.infer_calls[0]["text"] == "Hello"
    assert instance.infer_calls[0]["emo_alpha"] == 0.6
    assert instance.infer_calls[0]["emo_text"] == "calm but curious"
    assert instance.infer_calls[0]["emo_audio_prompt"] == str(emo_audio)
    assert instance.infer_calls[0]["use_emo_text"] is True
    assert instance.infer_calls[0]["emo_vector"] == [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]
    assert instance.infer_calls[0]["use_random"] is False
    assert instance.infer_calls[0]["do_sample"] is True
    assert instance.infer_calls[0]["temperature"] == 0.7
    assert instance.infer_calls[0]["top_p"] == 0.85
    assert instance.infer_calls[0]["top_k"] == 40
    assert instance.infer_calls[0]["num_beams"] == 2
    assert instance.infer_calls[0]["repetition_penalty"] == 8.5
    assert instance.infer_calls[0]["length_penalty"] == -0.1
    assert instance.infer_calls[0]["max_mel_tokens"] == 1200
    assert instance.infer_calls[0]["max_text_tokens_per_segment"] == 96
    assert chunks[-1].is_final is True
    assert chunks[0].sample_rate == 24_000


def test_indextts_boolean_params_parse_common_string_values(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("fake: true\n")

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_indextts_modules()
        from vox_indextts.adapter import IndexTTSAdapter

        adapter = IndexTTSAdapter()
        with patch("vox_indextts.adapter._clear_indextts_modules"):
            adapter.load(str(model_dir), "cuda")

        async def run():
            chunks = []
            async for chunk in adapter.synthesize(
                "Hello",
                reference_audio=np.zeros(2400, dtype=np.float32),
                params={
                    "use_emo_text": "false",
                    "use_random": "0",
                    "do_sample": "no",
                },
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())

    instance = _FakeIndexTTS2.instances[-1]
    assert instance.infer_calls[0]["use_emo_text"] is False
    assert instance.infer_calls[0]["use_random"] is False
    assert instance.infer_calls[0]["do_sample"] is False
    assert chunks[-1].is_final is True


def test_indextts_boolean_params_reject_ambiguous_values(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("fake: true\n")

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_indextts_modules()
        from vox_indextts.adapter import IndexTTSAdapter

        adapter = IndexTTSAdapter()
        with patch("vox_indextts.adapter._clear_indextts_modules"):
            adapter.load(str(model_dir), "cuda")

        async def run() -> None:
            async for _chunk in adapter.synthesize(
                "Hello",
                reference_audio=np.zeros(2400, dtype=np.float32),
                params={"do_sample": "sometimes"},
            ):
                pass

        with pytest.raises(InvalidConfigError, match="do_sample must be a boolean"):
            asyncio.run(run())


def test_indextts_constructor_does_not_swallow_internal_type_error(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("fake: true\n")

    class BrokenIndexTTS2:
        calls = 0

        def __init__(
            self,
            cfg_path="checkpoints/config.yaml",
            model_dir="checkpoints",
            use_fp16=False,
            device=None,
            use_cuda_kernel=None,
            use_deepspeed=False,
        ) -> None:
            BrokenIndexTTS2.calls += 1
            raise TypeError("internal upstream constructor failure")

    from vox_indextts.adapter import _construct_model

    with pytest.raises(TypeError, match="internal upstream constructor failure"):
        _construct_model(BrokenIndexTTS2, model_dir, "cuda")

    assert BrokenIndexTTS2.calls == 1


def test_indextts_requires_reference_audio_or_voice_path(tmp_path):
    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_indextts_modules()
        from vox_indextts.adapter import IndexTTSAdapter

        adapter = IndexTTSAdapter()
        with patch("vox_indextts.adapter._clear_indextts_modules"):
            adapter.load(str(tmp_path), "cuda")

        async def run():
            async for _ in adapter.synthesize("Hello"):
                pass

        with pytest.raises(ValueError, match="reference_audio or a voice path"):
            asyncio.run(run())


def test_indextts_preflight_requires_reference_audio_or_voice_path():
    from vox_indextts.adapter import IndexTTSAdapter

    with pytest.raises(InvalidConfigError, match="reference_audio"):
        IndexTTSAdapter().validate_synthesis_request()


def test_indextts_preflight_rejects_excessive_emotion_vector():
    from vox_indextts.adapter import IndexTTSAdapter

    with pytest.raises(InvalidConfigError, match="sum to 1.5"):
        IndexTTSAdapter().validate_synthesis_request(
            reference_audio=np.ones(2400, dtype=np.float32),
            params={
                "emotion_happy": 1.0,
                "emotion_calm": 0.6,
            },
        )


def test_indextts_preflight_rejects_missing_emotion_audio_prompt(tmp_path):
    from vox_indextts.adapter import IndexTTSAdapter

    with pytest.raises(InvalidConfigError, match="emo_audio_prompt does not exist"):
        IndexTTSAdapter().validate_synthesis_request(
            reference_audio=np.ones(2400, dtype=np.float32),
            params={"emo_audio_prompt": str(tmp_path / "missing.wav")},
        )


def test_indextts_preflight_accepts_emotion_audio_prompt(tmp_path):
    from vox_indextts.adapter import IndexTTSAdapter

    emo_audio = tmp_path / "emotion.wav"
    sf.write(emo_audio, np.array([0.0, 0.25, -0.25], dtype=np.float32), 24_000)

    IndexTTSAdapter().validate_synthesis_request(
        reference_audio=np.ones(2400, dtype=np.float32),
        params={"emo_audio_prompt": str(emo_audio)},
    )


def test_indextts_bootstraps_runtime_when_missing(tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "transformers==4.52.1" in cmd:
            _install_fake_indextts_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        sys.modules.pop("indextts", None)
        sys.modules.pop("indextts.infer_v2", None)
        from vox_indextts.adapter import IndexTTSAdapter

        with (
            patch("vox_indextts.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_indextts.adapter._clear_indextts_modules"),
        ):
            IndexTTSAdapter().load(str(tmp_path), "cuda")

    assert calls
    assert calls[0][:2] == ["uv", "pip"]
    assert "--target" in calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "indextts") in calls[0]
    assert "git+https://github.com/index-tts/index-tts.git" in calls[0]
    assert "--no-deps" in calls[0]
    assert "--upgrade" not in calls[0]
    dependency_commands = " ".join(" ".join(call) for call in calls[1:])
    assert "torch" not in dependency_commands
    assert "torchaudio" not in dependency_commands
    assert "numpy>=2.0,<2.4" in calls[1]
    assert "matplotlib>=3.10,<3.11" in calls[1]
    assert "protobuf==3.19.6" in calls[1]
    assert "transformers==4.52.1" in calls[1]


def test_indextts_prepare_runtime_bootstraps_without_loading_model(tmp_path):
    calls: list[list[str]] = []
    _FakeIndexTTS2.instances.clear()

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "transformers==4.52.1" in cmd:
            _install_fake_indextts_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        sys.modules.pop("indextts", None)
        sys.modules.pop("indextts.infer_v2", None)
        from vox_indextts.adapter import IndexTTSAdapter

        with (
            patch("vox_indextts.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_indextts.adapter._clear_indextts_modules"),
        ):
            IndexTTSAdapter().prepare_runtime()

    assert calls
    assert _FakeIndexTTS2.instances == []
    assert "git+https://github.com/index-tts/index-tts.git" in calls[0]
    assert "--no-deps" in calls[0]
    assert "--upgrade" not in calls[0]
    assert "numpy>=2.0,<2.4" in calls[1]
    assert "protobuf==3.19.6" in calls[1]
    assert "transformers==4.52.1" in calls[1]


def test_indextts_repairs_runtime_when_symbol_is_missing(tmp_path):
    calls: list[list[str]] = []
    _FakeIndexTTS2.instances.clear()

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "transformers==4.52.1" in cmd:
            _install_fake_indextts_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_stale_indextts_modules()
        from vox_indextts.adapter import IndexTTSAdapter

        with (
            patch("vox_indextts.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_indextts.adapter._clear_indextts_modules"),
        ):
            IndexTTSAdapter().prepare_runtime()

    assert calls
    assert _FakeIndexTTS2.instances == []
    assert "git+https://github.com/index-tts/index-tts.git" in calls[0]
    assert "--no-deps" in calls[0]
    assert "transformers==4.52.1" in calls[1]


def test_indextts_repairs_runtime_when_import_probe_is_broken(tmp_path):
    calls: list[list[str]] = []
    _FakeIndexTTS2.instances.clear()

    def fake_run(cmd, timeout):
        calls.append(cmd)
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    probe_results = [ValueError("broken runtime metadata"), _FakeIndexTTS2]

    def fake_probe():
        result = probe_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_indextts.adapter import IndexTTSAdapter

        with (
            patch("vox_indextts.adapter._run_install_command", side_effect=fake_run),
            patch("vox_indextts.adapter._indextts_class_from_runtime", side_effect=fake_probe),
            patch("vox_indextts.adapter._clear_indextts_modules"),
        ):
            IndexTTSAdapter().prepare_runtime()

    assert calls
    assert _FakeIndexTTS2.instances == []
    assert "git+https://github.com/index-tts/index-tts.git" in calls[0]
    assert "--no-deps" in calls[0]
    assert "transformers==4.52.1" in calls[1]


def test_indextts_clears_sibling_runtime_transformers_before_probe(tmp_path):
    stale_audiotools = ModuleType("audiotools")
    stale_transformers = ModuleType("transformers")
    stale_cache_utils = ModuleType("transformers.cache_utils")
    stale_google = ModuleType("google")
    stale_protobuf = ModuleType("google.protobuf")
    stale_tensorboard = ModuleType("tensorboard")
    stale_torch_utils_tensorboard = ModuleType("torch.utils.tensorboard")
    sys.modules["audiotools"] = stale_audiotools
    sys.modules["transformers"] = stale_transformers
    sys.modules["transformers.cache_utils"] = stale_cache_utils
    sys.modules["google"] = stale_google
    sys.modules["google.protobuf"] = stale_protobuf
    sys.modules["tensorboard"] = stale_tensorboard
    sys.modules["torch.utils.tensorboard"] = stale_torch_utils_tensorboard

    def fake_probe():
        assert "audiotools" not in sys.modules
        assert "transformers" not in sys.modules
        assert "transformers.cache_utils" not in sys.modules
        assert "google" not in sys.modules
        assert "google.protobuf" not in sys.modules
        assert "tensorboard" not in sys.modules
        assert "torch.utils.tensorboard" not in sys.modules
        return _FakeIndexTTS2

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_indextts.adapter import _load_indextts_class

        with patch("vox_indextts.adapter._indextts_class_from_runtime", side_effect=fake_probe):
            assert _load_indextts_class() is _FakeIndexTTS2


def test_indextts_numpy_compatibility_adds_bool8_alias(monkeypatch):
    from vox_indextts import adapter

    monkeypatch.delattr(adapter.np, "bool8", raising=False)

    adapter._apply_numpy_compatibility()

    assert adapter.np.bool8 is adapter.np.bool_


def test_indextts_patches_torchaudio_save_to_soundfile(tmp_path, monkeypatch):
    fake_torchaudio = ModuleType("torchaudio")

    def broken_save(*args, **kwargs):
        raise ImportError("torchcodec missing")

    fake_torchaudio.save = broken_save
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    from vox_indextts import adapter

    adapter._patch_torchaudio_save()

    output = tmp_path / "out.wav"
    fake_torchaudio.save(output, np.array([[0, 100, -100]], dtype=np.int16), 24_000)

    audio, sample_rate = sf.read(output, dtype="int16", always_2d=False)
    assert sample_rate == 24_000
    assert audio.shape == (3,)
    assert audio.tolist() == [0, 100, -100]


def test_indextts_removes_forbidden_torch_runtime_packages_before_probe(tmp_path):
    vox_home = tmp_path / "vox-home"
    runtime = vox_home / "runtime" / "indextts"
    for relative in (
        "torch",
        "torch-2.10.0.dist-info",
        "torchaudio",
        "torchaudio-2.10.0.dist-info",
        "nvidia",
        "nvidia_cuda_runtime_cu13-13.0.0.dist-info",
        "cuda",
        "cuda_toolkit-13.0.2.dist-info",
        "triton",
        "triton-3.5.0.dist-info",
    ):
        path = runtime / relative
        path.mkdir(parents=True)
        (path / "marker").write_text("stale", encoding="utf-8")

    def fake_probe():
        for path in runtime.iterdir():
            assert path.name in {"_vox_runtime_fallback_paths.pth"}
        return _FakeIndexTTS2

    with patch.dict(os.environ, {"VOX_HOME": str(vox_home)}):
        from vox_indextts.adapter import _load_indextts_class

        with patch("vox_indextts.adapter._indextts_class_from_runtime", side_effect=fake_probe):
            assert _load_indextts_class() is _FakeIndexTTS2


def test_indextts_removes_stale_matplotlib_before_runtime_repair(tmp_path):
    vox_home = tmp_path / "vox-home"
    runtime = vox_home / "runtime" / "indextts"
    for relative in (
        "matplotlib",
        "matplotlib-3.8.2.dist-info",
        "matplotlib-3.9.4.dist-info",
        "matplotlib.libs",
        "numpy",
        "numpy-1.26.4.dist-info",
        "numpy.libs",
    ):
        path = runtime / relative
        path.mkdir(parents=True)
        (path / "marker").write_text("stale", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "transformers==4.52.1" in cmd:
            _install_fake_indextts_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(vox_home)}):
        from vox_indextts.adapter import IndexTTSAdapter

        with (
            patch("vox_indextts.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_indextts.adapter._clear_indextts_modules"),
        ):
            IndexTTSAdapter().prepare_runtime()

    assert calls
    assert not (runtime / "matplotlib").exists()
    assert not (runtime / "matplotlib-3.8.2.dist-info").exists()
    assert not (runtime / "matplotlib-3.9.4.dist-info").exists()
    assert not (runtime / "matplotlib.libs").exists()
    assert not (runtime / "numpy").exists()
    assert not (runtime / "numpy-1.26.4.dist-info").exists()
    assert not (runtime / "numpy.libs").exists()


def test_indextts_runtime_probe_rejects_app_env_indextts_module(tmp_path):
    app_module_path = tmp_path / "app-env" / "indextts" / "infer_v2.py"
    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_indextts_modules(module_file=str(app_module_path))
        from vox_indextts.adapter import _indextts_class_from_runtime

        assert _indextts_class_from_runtime() is None


def test_indextts_runtime_probe_accepts_indextts_module_from_runtime(tmp_path):
    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_indextts_modules()
        from vox_indextts.adapter import _indextts_class_from_runtime

        assert _indextts_class_from_runtime() is _FakeIndexTTS2
