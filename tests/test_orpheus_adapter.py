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
    assert data["project"]["version"] == "0.1.7"
    assert not any(dep.startswith(("torch", "vllm", "orpheus-speech", "snac")) for dep in dependencies)


def test_orpheus_readme_uses_public_model_reference():
    readme = Path(__file__).parents[1] / "adapters" / "vox-orpheus" / "README.md"
    text = readme.read_text()

    assert "vox pull orpheus-tts:medium-3b" in text
    assert "vox run orpheus-tts:medium-3b" in text
    assert "vox pull orpheus-tts-vllm:medium-3b" not in text


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


def _orpheus_runtime_file() -> str:
    runtime_root = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox"))) / "runtime" / "orpheus"
    return str(runtime_root / "orpheus_tts" / "__init__.py")


def _install_fake_orpheus_modules(*, module_file: str | None = None) -> None:
    module = ModuleType("orpheus_tts")
    module.__file__ = module_file or _orpheus_runtime_file()
    module.OrpheusModel = _FakeOrpheusModel
    sys.modules["orpheus_tts"] = module


def _install_stale_orpheus_module(*, module_file: str | None = None) -> None:
    module = ModuleType("orpheus_tts")
    module.__file__ = module_file or _orpheus_runtime_file()
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


def test_orpheus_synthesis_parameters_expose_generation_controls():
    from vox_orpheus.adapter import OrpheusAdapter

    params = {param.name: param for param in OrpheusAdapter().synthesis_parameters()}

    assert set(params) == {"temperature", "top_p", "repetition_penalty", "max_tokens"}
    assert params["temperature"].default == 0.6
    assert params["top_p"].max_value == 1.0
    assert params["repetition_penalty"].min_value == 1.0
    assert params["max_tokens"].type == "integer"


def test_orpheus_load_rejects_cpu_before_runtime_install(tmp_path):
    from vox_orpheus.adapter import OrpheusAdapter

    with (
        patch("vox_orpheus.adapter._load_orpheus_model_class") as load_model_class,
        pytest.raises(RuntimeError, match="requires a Linux x86_64 CUDA runtime"),
    ):
        OrpheusAdapter().load(str(tmp_path), "cpu")

    load_model_class.assert_not_called()


def test_orpheus_load_and_synthesize(tmp_path):
    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_orpheus_modules()
        from vox_orpheus.adapter import OrpheusAdapter

        adapter = OrpheusAdapter()
        adapter.load(str(tmp_path), "cuda", _source="canopylabs/orpheus-tts-0.1-finetune-prod")

        async def run():
            chunks = []
            async for chunk in adapter.synthesize(
                "Hello",
                voice="tara",
                params={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "max_tokens": 256,
                },
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
    instance = _FakeOrpheusModel.instances[-1]

    assert instance.args[0] == "canopylabs/orpheus-tts-0.1-finetune-prod"
    assert instance.calls[0] == {
        "prompt": "Hello",
        "voice": "tara",
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "max_tokens": 256,
    }
    assert chunks[-1].is_final is True
    assert chunks[0].sample_rate == 24_000
    assert chunks[0].audio


def test_orpheus_rejects_reference_audio(tmp_path):
    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
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


def test_orpheus_preflight_rejects_unknown_voice():
    from vox_orpheus.adapter import OrpheusAdapter

    with pytest.raises(InvalidConfigError, match="Unsupported Orpheus voice"):
        OrpheusAdapter().validate_synthesis_request(voice="unknown")


def test_orpheus_rejects_runtime_that_cannot_accept_requested_params(tmp_path):
    class RuntimeWithoutParams:
        def generate_speech(self, prompt: str, voice: str):
            yield b"\x00\x00"

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_orpheus.adapter import OrpheusAdapter

        adapter = OrpheusAdapter()
        adapter._model = RuntimeWithoutParams()

        async def run():
            async for _ in adapter.synthesize("Hello", params={"temperature": 0.8}):
                pass

        with pytest.raises(RuntimeError, match="does not accept synthesis parameter 'temperature'"):
            asyncio.run(run())


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


def test_orpheus_repairs_runtime_when_symbol_is_missing(tmp_path):
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
        _install_stale_orpheus_module()
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


def test_orpheus_repairs_runtime_when_import_probe_is_broken(tmp_path):
    calls: list[list[str]] = []
    _FakeOrpheusModel.instances.clear()

    def fake_run(cmd, timeout):
        calls.append(cmd)
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    probe_results = [ValueError("broken runtime metadata"), _FakeOrpheusModel]

    def fake_probe():
        result = probe_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_orpheus.adapter import OrpheusAdapter

        with (
            patch("vox_orpheus.adapter._run_install_command", side_effect=fake_run),
            patch("vox_orpheus.adapter._orpheus_model_class_from_runtime", side_effect=fake_probe),
            patch("vox_orpheus.adapter._clear_orpheus_modules"),
        ):
            OrpheusAdapter().prepare_runtime()

    assert calls
    assert _FakeOrpheusModel.instances == []
    assert "--target" in calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "orpheus") in calls[0]


def test_orpheus_runtime_probe_rejects_app_env_orpheus_module(tmp_path):
    app_module_path = tmp_path / "app-env" / "orpheus_tts" / "__init__.py"
    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_orpheus_modules(module_file=str(app_module_path))
        from vox_orpheus.adapter import _orpheus_model_class_from_runtime

        assert _orpheus_model_class_from_runtime() is None


def test_orpheus_runtime_probe_accepts_orpheus_module_from_runtime(tmp_path):
    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        _install_fake_orpheus_modules()
        from vox_orpheus.adapter import _orpheus_model_class_from_runtime

        assert _orpheus_model_class_from_runtime() is _FakeOrpheusModel
