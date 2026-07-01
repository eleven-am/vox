from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np

from vox.core.types import ModelFormat, ModelType


class _FakeSparkTTS:
    instances: list[_FakeSparkTTS] = []

    def __init__(self, *args) -> None:
        self.args = args
        self.calls: list[dict] = []
        _FakeSparkTTS.instances.append(self)

    def inference(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return np.array([0.0, 0.25, -0.25], dtype=np.float32)


def _install_fake_spark_modules() -> None:
    cli = ModuleType("cli")
    spark = ModuleType("cli.SparkTTS")
    spark.SparkTTS = _FakeSparkTTS
    sys.modules["cli"] = cli
    sys.modules["cli.SparkTTS"] = spark


def test_spark_package_import_is_light():
    sys.modules.pop("vox_spark", None)
    sys.modules.pop("vox_spark.adapter", None)
    sys.modules.pop("cli", None)
    sys.modules.pop("cli.SparkTTS", None)

    module = importlib.import_module("vox_spark")

    assert module.__all__ == ["SparkTTSAdapter"]
    assert "cli.SparkTTS" not in sys.modules


def test_spark_info_returns_correct_metadata():
    from vox_spark.adapter import SparkTTSAdapter

    info = SparkTTSAdapter().info()

    assert info.name == "spark-tts-torch"
    assert info.type == ModelType.TTS
    assert info.default_sample_rate == 16_000
    assert ModelFormat.PYTORCH in info.supported_formats
    assert info.supports_streaming is False
    assert info.supports_voice_cloning is True


def test_spark_load_and_synthesize_with_reference_audio(tmp_path):
    _install_fake_spark_modules()
    from vox_spark.adapter import SparkTTSAdapter

    adapter = SparkTTSAdapter()
    with patch("vox_spark.adapter._torch_device", return_value="cpu-device"):
        adapter.load(str(tmp_path), "cpu")

    async def run():
        chunks = []
        async for chunk in adapter.synthesize(
            "Hello",
            reference_audio=np.zeros(1600, dtype=np.float32),
            reference_text="Reference",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    instance = _FakeSparkTTS.instances[-1]

    assert instance.args == (Path(tmp_path), "cpu-device")
    assert instance.calls[0]["text"] == "Hello"
    assert instance.calls[0]["prompt_text"] == "Reference"
    assert instance.calls[0]["prompt_speech_path"].name == "reference.wav"
    assert chunks[0].sample_rate == 16_000
    assert chunks[-1].is_final is True


def test_spark_uses_generated_voice_when_no_reference(tmp_path):
    _install_fake_spark_modules()
    from vox_spark.adapter import SparkTTSAdapter

    adapter = SparkTTSAdapter()
    with patch("vox_spark.adapter._torch_device", return_value="cpu-device"):
        adapter.load(str(tmp_path), "cpu")

    async def run():
        async for _ in adapter.synthesize("Hello", speed=1.5):
            pass

    asyncio.run(run())
    call = _FakeSparkTTS.instances[-1].calls[0]

    assert call["gender"] == "female"
    assert call["pitch"] == "moderate"
    assert call["speed"] == "high"


def test_spark_bootstraps_runtime_when_missing(tmp_path):
    install_calls: list[list[str]] = []
    git_calls: list[list[str]] = []

    def fake_git(cmd, timeout):
        git_calls.append(cmd)
        target = Path(cmd[-1])
        (target / "cli").mkdir(parents=True, exist_ok=True)
        (target / "cli" / "SparkTTS.py").write_text("class SparkTTS: pass\n")
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    def fake_install(cmd, timeout):
        install_calls.append(cmd)
        _install_fake_spark_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        sys.modules.pop("cli", None)
        sys.modules.pop("cli.SparkTTS", None)
        from vox_spark.adapter import SparkTTSAdapter

        with (
            patch("vox_spark.adapter._run_git_command", side_effect=fake_git),
            patch("vox_spark.adapter._run_install_command", side_effect=fake_install),
            patch("vox_spark.adapter._clear_spark_modules"),
            patch("vox_spark.adapter._torch_device", return_value="cpu-device"),
        ):
            SparkTTSAdapter().load(str(tmp_path), "cpu")

    assert git_calls
    assert git_calls[0][:4] == ["git", "clone", "--depth", "1"]
    assert install_calls
    assert "--target" in install_calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "spark") in install_calls[0]
    assert "transformers==4.46.2" in install_calls[0]
