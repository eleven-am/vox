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


class _FakeNeuTTSAir:
    instances: list[_FakeNeuTTSAir] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.encoded_paths: list[object] = []
        self.stream_calls: list[tuple] = []
        self.infer_calls: list[tuple] = []
        self.raise_stream = False
        self.raise_stream_after = None
        _FakeNeuTTSAir.instances.append(self)

    def encode_reference(self, ref_path):
        self.encoded_paths.append(ref_path)
        return np.array([1, 2, 3], dtype=np.int64)

    def infer_stream(self, text, ref_codes, ref_text):
        self.stream_calls.append((text, ref_codes, ref_text))
        if self.raise_stream:
            raise NotImplementedError
        if self.raise_stream_after is not None:
            for _ in range(self.raise_stream_after):
                yield np.array([0.0, 0.5], dtype=np.float32)
            raise NotImplementedError
        yield np.array([0.0, 0.5], dtype=np.float32)

    def infer(self, text, ref_codes, ref_text):
        self.infer_calls.append((text, ref_codes, ref_text))
        return np.array([0.0, -0.5], dtype=np.float32)


def _install_fake_neutts_modules() -> None:
    module = ModuleType("neuttsair")
    module.NeuTTSAir = _FakeNeuTTSAir
    sys.modules["neuttsair"] = module


def test_neutts_package_import_is_light():
    sys.modules.pop("vox_neutts", None)
    sys.modules.pop("vox_neutts.adapter", None)
    sys.modules.pop("neuttsair", None)

    module = importlib.import_module("vox_neutts")

    assert module.__all__ == ["NeuTTSAirAdapter"]
    assert "neuttsair" not in sys.modules


def test_neutts_info_returns_correct_metadata():
    from vox_neutts.adapter import NeuTTSAirAdapter

    info = NeuTTSAirAdapter().info()

    assert info.name == "neutts-air-tts-torch"
    assert info.type == ModelType.TTS
    assert info.default_sample_rate == 24_000
    assert ModelFormat.PYTORCH in info.supported_formats
    assert info.supports_streaming is True
    assert info.supports_voice_cloning is True


def test_neutts_load_and_synthesize_with_reference_audio(tmp_path):
    _install_fake_neutts_modules()
    from vox_neutts.adapter import NeuTTSAirAdapter

    adapter = NeuTTSAirAdapter()
    adapter.load(str(tmp_path), "cuda", _source="neuphonic/neutts-air")

    async def run():
        chunks = []
        async for chunk in adapter.synthesize(
            "Hello",
            reference_audio=np.zeros(2400, dtype=np.float32),
            reference_text="Reference",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    instance = _FakeNeuTTSAir.instances[-1]

    assert instance.kwargs["backbone_repo"] == "neuphonic/neutts-air"
    assert instance.kwargs["backbone_device"] == "cuda"
    assert instance.kwargs["codec_repo"] == "neuphonic/neucodec"
    assert instance.stream_calls[0][0] == "Hello"
    assert instance.stream_calls[0][2] == "Reference"
    assert chunks[0].sample_rate == 24_000
    assert chunks[-1].is_final is True


def test_neutts_requires_reference_audio_or_saved_speaker(tmp_path):
    _install_fake_neutts_modules()
    from vox_neutts.adapter import NeuTTSAirAdapter

    adapter = NeuTTSAirAdapter()
    adapter.load(str(tmp_path), "cpu")

    async def run():
        async for _ in adapter.synthesize("Hello"):
            pass

    with pytest.raises(ValueError, match="reference_audio"):
        asyncio.run(run())


def test_neutts_preflight_requires_reference_audio_or_saved_speaker():
    from vox_neutts.adapter import NeuTTSAirAdapter

    with pytest.raises(InvalidConfigError, match="reference_audio"):
        NeuTTSAirAdapter().validate_synthesis_request()


def test_neutts_falls_back_when_streaming_backend_is_not_implemented(tmp_path):
    _install_fake_neutts_modules()
    from vox_neutts.adapter import NeuTTSAirAdapter

    adapter = NeuTTSAirAdapter()
    adapter.load(str(tmp_path), "cpu")
    _FakeNeuTTSAir.instances[-1].raise_stream = True

    async def run():
        chunks = []
        async for chunk in adapter.synthesize(
            "Hello",
            reference_audio=np.zeros(2400, dtype=np.float32),
            reference_text="Reference",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    instance = _FakeNeuTTSAir.instances[-1]

    assert instance.infer_calls[0][0] == "Hello"
    assert chunks[0].audio == np.array([0.0, -0.5], dtype=np.float32).tobytes()


def test_neutts_does_not_duplicate_audio_when_stream_fails_midway(tmp_path):
    _install_fake_neutts_modules()
    from vox_neutts.adapter import NeuTTSAirAdapter

    adapter = NeuTTSAirAdapter()
    adapter.load(str(tmp_path), "cpu")
    _FakeNeuTTSAir.instances[-1].raise_stream_after = 2

    async def run():
        return [
            chunk
            async for chunk in adapter.synthesize(
                "Hello",
                reference_audio=np.zeros(2400, dtype=np.float32),
                reference_text="Reference",
            )
        ]

    chunks = asyncio.run(run())
    instance = _FakeNeuTTSAir.instances[-1]

    assert instance.infer_calls == []
    audio_chunks = [c for c in chunks if c.audio]
    assert len(audio_chunks) == 2


def test_neutts_bootstraps_runtime_when_missing(tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        _install_fake_neutts_modules()
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with patch.dict(os.environ, {"VOX_HOME": str(tmp_path / "vox-home")}):
        sys.modules.pop("neuttsair", None)
        from vox_neutts.adapter import NeuTTSAirAdapter

        with (
            patch("vox_neutts.adapter.subprocess.run", side_effect=fake_run),
            patch("vox_neutts.adapter._clear_neutts_modules"),
        ):
            NeuTTSAirAdapter().load(str(tmp_path), "cpu")

    assert calls
    assert calls[0][:2] == ["uv", "pip"]
    assert "--target" in calls[0]
    assert str(tmp_path / "vox-home" / "runtime" / "neutts") in calls[0]
    assert "neutts==1.2.1" in calls[0]
