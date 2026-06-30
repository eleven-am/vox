from __future__ import annotations

import asyncio
import importlib
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np

from vox.core.types import ModelFormat, ModelType


class _FakeChatterboxModel:
    sr = 24_000

    def __init__(self) -> None:
        self.generate_calls: list[tuple[str, dict]] = []

    def generate(self, text: str, **kwargs):
        self.generate_calls.append((text, kwargs))
        return np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32)


class _FakeChatterboxTTS:
    model = _FakeChatterboxModel()
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls.model


def _install_fake_chatterbox_modules(class_name: str = "ChatterboxTurboTTS") -> _FakeChatterboxModel:
    chatterbox = ModuleType("chatterbox")
    tts = ModuleType("chatterbox.tts")
    tts_turbo = ModuleType("chatterbox.tts_turbo")
    setattr(tts, class_name, _FakeChatterboxTTS)
    setattr(tts_turbo, class_name, _FakeChatterboxTTS)
    sys.modules["chatterbox"] = chatterbox
    sys.modules["chatterbox.tts"] = tts
    sys.modules["chatterbox.tts_turbo"] = tts_turbo
    return _FakeChatterboxTTS.model


def test_chatterbox_package_import_is_light():
    sys.modules.pop("vox_chatterbox", None)
    sys.modules.pop("vox_chatterbox.adapter", None)
    sys.modules.pop("chatterbox", None)

    module = importlib.import_module("vox_chatterbox")

    assert module.__all__ == [
        "ChatterboxAdapter",
        "ChatterboxMultilingualAdapter",
        "ChatterboxTurboAdapter",
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

    assert calls
    assert calls[0][:2] == ["uv", "pip"]
    assert "--target" in calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "chatterbox") in calls[0]
    assert "chatterbox-tts>=0.1.7,<0.2.0" in calls[0]
