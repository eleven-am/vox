from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from vox.core.types import ModelFormat, ModelType
from vox.operations.errors import InvalidConfigError


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


def _install_fake_indextts_modules() -> None:
    package = ModuleType("indextts")
    infer_v2 = ModuleType("indextts.infer_v2")
    infer_v2.IndexTTS2 = _FakeIndexTTS2
    sys.modules["indextts"] = package
    sys.modules["indextts.infer_v2"] = infer_v2


def test_indextts_package_import_is_light():
    sys.modules.pop("vox_indextts", None)
    sys.modules.pop("vox_indextts.adapter", None)
    sys.modules.pop("indextts", None)

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


def test_indextts_load_uses_config_and_synthesizes_with_reference_audio(tmp_path):
    _install_fake_indextts_modules()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("fake: true\n")

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_indextts.adapter import IndexTTSAdapter

        adapter = IndexTTSAdapter()
        adapter.load(str(model_dir), "cpu")

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
    _install_fake_indextts_modules()
    from vox_indextts.adapter import IndexTTSAdapter

    adapter = IndexTTSAdapter()
    adapter.load(str(tmp_path), "cpu")

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
            IndexTTSAdapter().load(str(tmp_path), "cpu")

    assert calls
    assert calls[0][:2] == ["uv", "pip"]
    assert "--target" in calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "indextts") in calls[0]
    assert "git+https://github.com/index-tts/index-tts.git" in calls[0]
