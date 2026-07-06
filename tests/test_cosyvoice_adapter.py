from __future__ import annotations

import asyncio
import importlib
import os
import sys
import tomllib
from pathlib import Path
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


def test_cosyvoice_package_metadata_version():
    pyproject = Path(__file__).parents[1] / "adapters" / "vox-cosyvoice" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["version"] == "0.1.5"


def test_cosyvoice_load_rejects_cpu_before_runtime_install(tmp_path):
    from vox_cosyvoice.adapter import CosyVoice2Adapter

    with patch("vox_cosyvoice.adapter._load_cosyvoice_class") as load_cosyvoice_class:
        with pytest.raises(RuntimeError, match="requires a Linux x86_64 CUDA runtime"):
            CosyVoice2Adapter().load(str(tmp_path), "cpu")

    load_cosyvoice_class.assert_not_called()


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
    adapter.load(str(tmp_path), "cuda")

    async def run():
        async for _ in adapter.synthesize("Hello"):
            pass

    with pytest.raises(ValueError, match="reference_audio"):
        asyncio.run(run())


def test_cosyvoice_preflight_requires_reference_or_speaker():
    from vox_cosyvoice.adapter import CosyVoice2Adapter

    with pytest.raises(InvalidConfigError, match="reference_audio"):
        CosyVoice2Adapter().validate_synthesis_request()


def test_cosyvoice_torchaudio_soundfile_loader_avoids_torchcodec(tmp_path):
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    sf = pytest.importorskip("soundfile")
    from vox_cosyvoice.adapter import _patch_torchaudio_soundfile_loader

    wav_path = tmp_path / "reference.wav"
    sf.write(wav_path, np.array([0.0, 0.1, -0.1], dtype=np.float32), 24_000)

    _patch_torchaudio_soundfile_loader()
    audio, sample_rate = torchaudio.load(str(wav_path), backend="soundfile")

    assert sample_rate == 24_000
    assert isinstance(audio, torch.Tensor)
    assert audio.shape == (1, 3)


def test_cosyvoice_bootstraps_runtime_when_missing(tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            source_root = Path(cmd[-1])
            (source_root / "cosyvoice" / "cli").mkdir(parents=True)
            (source_root / "cosyvoice" / "__init__.py").write_text("")
            (source_root / "cosyvoice" / "cli" / "__init__.py").write_text("")
            (source_root / "cosyvoice" / "cli" / "cosyvoice.py").write_text("")
            (source_root / "third_party" / "Matcha-TTS" / "matcha").mkdir(parents=True)
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
            CosyVoice2Adapter().load(str(tmp_path), "cuda")

    assert calls
    assert calls[0][:2] == ["git", "clone"]
    assert "--recurse-submodules" in calls[0]
    assert "https://github.com/FunAudioLLM/CosyVoice.git" in calls[0]
    install_call = next(call for call in calls if call[:2] == ["uv", "pip"])
    assert "--target" in install_call
    assert str(tmp_path / "vox-home" / "runtime" / "cosyvoice") in install_call
    assert "HyperPyYAML==1.2.3" in install_call
    assert "huggingface-hub>=0.34,<1.0" in install_call
    assert "lightning==2.2.4" in install_call
    assert "numpy==1.26.4" in install_call
    assert "tiktoken>=0.7,<1" in install_call
    assert not any(str(part).startswith("matplotlib") for part in install_call)
    assert "torch==2.3.1" not in install_call
    assert not any(str(part).startswith("openai-whisper") for part in install_call)
    assert not any(str(part).startswith("git+https://github.com/FunAudioLLM/CosyVoice") for part in install_call)

    whisper_shim = tmp_path / "vox-home" / "runtime" / "cosyvoice" / "whisper" / "__init__.py"
    assert "def log_mel_spectrogram" in whisper_shim.read_text()
    whisper_tokenizer = tmp_path / "vox-home" / "runtime" / "cosyvoice" / "whisper" / "tokenizer.py"
    assert "class Tokenizer" in whisper_tokenizer.read_text()
    matplotlib_stub = tmp_path / "vox-home" / "runtime" / "cosyvoice" / "matplotlib" / "pyplot.py"
    assert "CosyVoice inference runtime does not include matplotlib" in matplotlib_stub.read_text()
    matplotlib_init = tmp_path / "vox-home" / "runtime" / "cosyvoice" / "matplotlib" / "__init__.py"
    matplotlib_init_text = matplotlib_init.read_text()
    assert "from . import axes, colors" in matplotlib_init_text
    assert "def use" in matplotlib_init_text
    matplotlib_pylab = tmp_path / "vox-home" / "runtime" / "cosyvoice" / "matplotlib" / "pylab.py"
    assert "from .pyplot import *" in matplotlib_pylab.read_text()


def test_cosyvoice_prepare_runtime_bootstraps_without_loading_model(tmp_path):
    calls: list[list[str]] = []
    _FakeCosyVoice2.instances.clear()

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            source_root = Path(cmd[-1])
            (source_root / "cosyvoice" / "cli").mkdir(parents=True)
            (source_root / "cosyvoice" / "__init__.py").write_text("")
            (source_root / "cosyvoice" / "cli" / "__init__.py").write_text("")
            (source_root / "cosyvoice" / "cli" / "cosyvoice.py").write_text("")
            (source_root / "third_party" / "Matcha-TTS" / "matcha").mkdir(parents=True)
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
            CosyVoice2Adapter().prepare_runtime()

    assert calls
    assert _FakeCosyVoice2.instances == []
    assert calls[0][:2] == ["git", "clone"]
    install_call = next(call for call in calls if call[:2] == ["uv", "pip"])
    assert "--target" in install_call
    assert str(tmp_path / "vox-home" / "runtime" / "cosyvoice") in install_call
    assert "HyperPyYAML==1.2.3" in install_call


def test_cosyvoice_repairs_runtime_when_import_probe_is_broken(tmp_path):
    calls: list[list[str]] = []
    probe_results = [ValueError("broken runtime metadata"), _FakeCosyVoice2]

    def fake_probe():
        result = probe_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        from vox_cosyvoice.adapter import CosyVoice2Adapter

        with (
            patch("vox_cosyvoice.adapter._source_checkout_complete", return_value=True),
            patch("vox_cosyvoice.adapter._cosyvoice_class_from_runtime", side_effect=fake_probe),
            patch("vox_cosyvoice.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_cosyvoice.adapter._clear_cosyvoice_modules"),
        ):
            CosyVoice2Adapter().prepare_runtime()

    assert calls
    install_call = next(call for call in calls if call[:2] == ["uv", "pip"])
    assert "--target" in install_call
    assert str(tmp_path / "vox-home" / "runtime" / "cosyvoice") in install_call
    assert "HyperPyYAML==1.2.3" in install_call
