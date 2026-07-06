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

import pytest

from vox.core.types import ModelFormat, ModelType
from vox.operations.errors import InvalidConfigError


def test_orpheus_package_metadata_is_lightweight():
    pyproject = Path(__file__).parents[1] / "adapters" / "vox-orpheus" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    dependencies = data["project"]["dependencies"]
    assert data["project"]["version"] == "0.1.3"
    assert not any(dep.startswith(("torch", "vllm", "orpheus-speech", "snac")) for dep in dependencies)


class _FakeOrpheusModel:
    instances: list[_FakeOrpheusModel] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[dict] = []
        _FakeOrpheusModel.instances.append(self)

    def generate_speech(self, **kwargs):
        self.calls.append(kwargs)
        yield b"\x00\x00\x00@\x00\xc0"


def _install_fake_orpheus_modules() -> None:
    module = ModuleType("orpheus_tts")
    module.OrpheusModel = _FakeOrpheusModel
    sys.modules["orpheus_tts"] = module


def test_orpheus_package_import_is_light():
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        heavy_modules = ("orpheus_tts", "vllm", "torch", "snac")
        if name in heavy_modules or name.startswith(tuple(f"{module}." for module in heavy_modules)):
            raise AssertionError(f"vox-orpheus imported heavy runtime dependency during package import: {name}")
        return original_import(name, *args, **kwargs)

    sys.modules.pop("vox_orpheus", None)
    sys.modules.pop("vox_orpheus.adapter", None)
    sys.modules.pop("orpheus_tts", None)

    with patch.object(builtins, "__import__", side_effect=guarded_import):
        module = importlib.import_module("vox_orpheus")

    assert module.__all__ == ["OrpheusAdapter"]
    assert "orpheus_tts" not in sys.modules


def test_orpheus_info_returns_correct_metadata():
    from vox_orpheus.adapter import OrpheusAdapter

    info = OrpheusAdapter().info()

    assert info.name == "orpheus-tts-vllm"
    assert info.type == ModelType.TTS
    assert info.default_sample_rate == 24_000
    assert ModelFormat.PYTORCH in info.supported_formats
    assert info.supports_streaming is True
    assert info.supports_voice_cloning is False


def test_orpheus_load_rejects_cpu_before_runtime_install(tmp_path):
    from vox_orpheus.adapter import OrpheusAdapter

    with patch("vox_orpheus.adapter._load_orpheus_model_class") as load_model_class:
        with pytest.raises(RuntimeError, match="requires a Linux x86_64 CUDA runtime"):
            OrpheusAdapter().load(str(tmp_path), "cpu")

    load_model_class.assert_not_called()


def test_orpheus_load_and_synthesize(tmp_path):
    _install_fake_orpheus_modules()
    from vox_orpheus.adapter import OrpheusAdapter

    adapter = OrpheusAdapter()
    adapter.load(str(tmp_path), "cuda", _source="canopylabs/orpheus-tts-0.1-finetune-prod")

    async def run():
        chunks = []
        async for chunk in adapter.synthesize("Hello", voice="tara"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    instance = _FakeOrpheusModel.instances[-1]

    assert instance.args[0] == "canopylabs/orpheus-tts-0.1-finetune-prod"
    assert instance.calls[0] == {"prompt": "Hello", "voice": "tara"}
    assert chunks[-1].is_final is True
    assert chunks[0].sample_rate == 24_000
    assert chunks[0].audio


def test_orpheus_rejects_reference_audio(tmp_path):
    _install_fake_orpheus_modules()
    from vox_orpheus.adapter import OrpheusAdapter

    adapter = OrpheusAdapter()
    adapter.load(str(tmp_path), "cuda")

    async def run():
        async for _ in adapter.synthesize("Hello", reference_text="x"):
            pass

    with pytest.raises(ValueError, match="reference_audio/reference_text"):
        asyncio.run(run())


def test_orpheus_preflight_rejects_reference_audio():
    from vox_orpheus.adapter import OrpheusAdapter

    with pytest.raises(InvalidConfigError, match="reference_audio/reference_text"):
        OrpheusAdapter().validate_synthesis_request(reference_text="x")


def test_orpheus_bootstraps_runtime_when_missing(tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        _install_fake_orpheus_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        sys.modules.pop("orpheus_tts", None)
        from vox_orpheus.adapter import OrpheusAdapter

        with (
            patch("vox_orpheus.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_orpheus.adapter._clear_orpheus_modules"),
        ):
            OrpheusAdapter().load(str(tmp_path), "cuda")

    assert calls
    assert calls[0][:2] == ["uv", "pip"]
    assert "--target" in calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "orpheus") in calls[0]
    assert "orpheus-speech==0.1.0" in calls[0]


def test_orpheus_prepare_runtime_bootstraps_without_loading_model(tmp_path):
    calls: list[list[str]] = []
    _FakeOrpheusModel.instances.clear()

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        _install_fake_orpheus_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        sys.modules.pop("orpheus_tts", None)
        from vox_orpheus.adapter import OrpheusAdapter

        with (
            patch("vox_orpheus.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_orpheus.adapter._clear_orpheus_modules"),
        ):
            OrpheusAdapter().prepare_runtime()

    assert calls
    assert _FakeOrpheusModel.instances == []
    assert "--target" in calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "orpheus") in calls[0]
    assert "orpheus-speech==0.1.0" in calls[0]
