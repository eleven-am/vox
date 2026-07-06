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
    assert data["project"]["version"] == "0.1.6"
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


def test_indextts_readme_uses_public_model_reference():
    readme = Path(__file__).parents[1] / "adapters" / "vox-indextts" / "README.md"
    text = readme.read_text(encoding="utf-8")

    assert "vox pull indextts-tts:2" in text
    assert "vox run indextts-tts:2" in text
    assert "vox pull indextts-tts-torch:2" not in text


def test_indextts_load_rejects_cpu_before_runtime_install(tmp_path):
    from vox_indextts.adapter import IndexTTSAdapter

    with patch("vox_indextts.adapter._load_indextts_class") as load_indextts_class:
        with pytest.raises(RuntimeError, match="requires a Linux x86_64 CUDA runtime"):
            IndexTTSAdapter().load(str(tmp_path), "cpu")

    load_indextts_class.assert_not_called()


def test_indextts_load_uses_config_and_synthesizes_with_reference_audio(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("fake: true\n")

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_indextts_modules()
        from vox_indextts.adapter import IndexTTSAdapter

        adapter = IndexTTSAdapter()
        adapter.load(str(model_dir), "cuda")

        async def run():
            chunks = []
            async for chunk in adapter.synthesize(
                "Hello",
                reference_audio=np.zeros(2400, dtype=np.float32),
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())

    instance = _FakeIndexTTS2.instances[-1]
    assert instance.kwargs["cfg_path"].endswith("config.yaml")
    assert instance.kwargs["model_dir"] == str(model_dir)
    assert instance.infer_calls[0]["text"] == "Hello"
    assert chunks[-1].is_final is True
    assert chunks[0].sample_rate == 24_000


def test_indextts_requires_reference_audio_or_voice_path(tmp_path):
    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_indextts_modules()
        from vox_indextts.adapter import IndexTTSAdapter

        adapter = IndexTTSAdapter()
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
