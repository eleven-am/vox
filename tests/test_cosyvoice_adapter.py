from __future__ import annotations

import asyncio
import importlib
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType
from vox.operations.errors import InvalidConfigError


class _FakeCosyVoice2:
    instances: list[_FakeCosyVoice2] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[dict] = []
        _FakeCosyVoice2.instances.append(self)

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, **kwargs):
        self.calls.append(
            {
                "tts_text": tts_text,
                "prompt_text": prompt_text,
                "prompt_wav": prompt_wav,
                **kwargs,
            }
        )
        yield {"tts_speech": np.array([0.0, 0.25, -0.25], dtype=np.float32)}


def _install_fake_cosyvoice_modules() -> None:
    package = ModuleType("cosyvoice")
    cli = ModuleType("cosyvoice.cli")
    cosyvoice = ModuleType("cosyvoice.cli.cosyvoice")
    cosyvoice.CosyVoice2 = _FakeCosyVoice2
    sys.modules["cosyvoice"] = package
    sys.modules["cosyvoice.cli"] = cli
    sys.modules["cosyvoice.cli.cosyvoice"] = cosyvoice


def test_cosyvoice_package_import_is_light():
    sys.modules.pop("vox_cosyvoice", None)
    sys.modules.pop("vox_cosyvoice.adapter", None)
    sys.modules.pop("cosyvoice", None)

    module = importlib.import_module("vox_cosyvoice")

    assert module.__all__ == ["CosyVoice2Adapter"]
    assert "cosyvoice" not in sys.modules


def test_cosyvoice_info_returns_correct_metadata():
    from vox_cosyvoice.adapter import CosyVoice2Adapter

    info = CosyVoice2Adapter().info()

    assert info.name == "cosyvoice2-tts-torch"
    assert info.type == ModelType.TTS
    assert info.default_sample_rate == 24_000
    assert ModelFormat.PYTORCH in info.supported_formats
    assert info.supports_streaming is True
    assert info.supports_voice_cloning is True


def test_cosyvoice_load_and_synthesize_with_reference_audio(tmp_path):
    _install_fake_cosyvoice_modules()
    from vox_cosyvoice.adapter import CosyVoice2Adapter

    adapter = CosyVoice2Adapter()
    adapter.load(str(tmp_path), "cuda")

    async def run():
        chunks = []
        async for chunk in adapter.synthesize(
            "Hello",
            reference_audio=np.zeros(2400, dtype=np.float32),
            reference_text="Reference",
            speed=1.2,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    instance = _FakeCosyVoice2.instances[-1]

    assert instance.kwargs["fp16"] is True
    assert instance.calls[0]["tts_text"] == "Hello"
    assert instance.calls[0]["prompt_text"] == "Reference"
    assert instance.calls[0]["prompt_wav"].endswith("reference.wav")
    assert instance.calls[0]["stream"] is True
    assert instance.calls[0]["speed"] == 1.2
    assert chunks[-1].is_final is True
    assert chunks[0].sample_rate == 24_000


def test_cosyvoice_requires_reference_or_saved_speaker(tmp_path):
    _install_fake_cosyvoice_modules()
    from vox_cosyvoice.adapter import CosyVoice2Adapter

    adapter = CosyVoice2Adapter()
    adapter.load(str(tmp_path), "cpu")

    async def run():
        async for _ in adapter.synthesize("Hello"):
            pass

    with pytest.raises(ValueError, match="reference_audio"):
        asyncio.run(run())


def test_cosyvoice_preflight_requires_reference_or_speaker():
    from vox_cosyvoice.adapter import CosyVoice2Adapter

    with pytest.raises(InvalidConfigError, match="reference_audio"):
        CosyVoice2Adapter().validate_synthesis_request()


def test_cosyvoice_bootstraps_runtime_when_missing(tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        _install_fake_cosyvoice_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        sys.modules.pop("cosyvoice", None)
        sys.modules.pop("cosyvoice.cli", None)
        sys.modules.pop("cosyvoice.cli.cosyvoice", None)
        from vox_cosyvoice.adapter import CosyVoice2Adapter

        with (
            patch("vox_cosyvoice.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_cosyvoice.adapter._clear_cosyvoice_modules"),
        ):
            CosyVoice2Adapter().load(str(tmp_path), "cpu")

    assert calls
    assert calls[0][:2] == ["uv", "pip"]
    assert "--target" in calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "cosyvoice") in calls[0]
    assert "git+https://github.com/FunAudioLLM/CosyVoice.git" in calls[0]
