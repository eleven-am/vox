from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.registry import CATALOG
from vox.core.types import ModelFormat, ModelType


def _mock_torch(cuda_available: bool = True, mps_available: bool = False):
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = cuda_available
    torch_mock.backends.mps.is_available.return_value = mps_available
    return torch_mock


def _write_piper_bundle(root: Path) -> tuple[Path, Path]:
    model_dir = root / "en" / "en_US" / "lessac" / "medium"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "en_US-lessac-medium.onnx"
    config_file = model_dir / "en_US-lessac-medium.onnx.json"
    model_file.write_bytes(b"fake-onnx")
    config_file.write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050, "quality": "medium"},
                "espeak": {"voice": "en-us"},
                "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
                "phoneme_type": "espeak",
            }
        )
    )
    return model_file, config_file


class TestPiperAdapter:
    def test_registry_points_to_specific_voice_bundle(self):
        entry = CATALOG["piper-tts-onnx"]["en-us-lessac-medium"]

        assert entry["source"] == "rhasspy/piper-voices"
        assert entry["format"] == "onnx"
        assert entry["files"] == [
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        ]

    def test_package_import(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "piper": MagicMock()}):
            sys.modules.pop("vox_piper", None)
            sys.modules.pop("vox_piper.adapter", None)
            module = importlib.import_module("vox_piper")
            assert module.__all__ == ["PiperAdapter"]

    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "piper": MagicMock()}):
            from vox_piper.adapter import PiperAdapter

            adapter = PiperAdapter()
            info = adapter.info()

            assert info.name == "piper-tts-onnx"
            assert info.type == ModelType.TTS
            assert "piper" in info.architectures
            assert info.default_sample_rate == 22050
            assert ModelFormat.ONNX in info.supported_formats
            assert info.supports_streaming is True
            assert info.supports_voice_cloning is False
            assert info.supported_languages == ("en-us",)

    def test_load_finds_model_and_config(self, tmp_path: Path):
        torch_mock = _mock_torch()
        piper_module = MagicMock()
        voice_cls = MagicMock()
        voice_instance = MagicMock()
        voice_cls.load.return_value = voice_instance
        piper_module.PiperVoice = voice_cls
        _write_piper_bundle(tmp_path)

        with patch.dict("sys.modules", {"torch": torch_mock, "piper": piper_module}):
            from vox_piper.adapter import PiperAdapter

            adapter = PiperAdapter()
            adapter.load(str(tmp_path), "cuda", _source="rhasspy/piper-voices")

            voice_cls.load.assert_called_once()
            args, kwargs = voice_cls.load.call_args
            assert args[0].endswith("en_US-lessac-medium.onnx")
            assert args[1].endswith("en_US-lessac-medium.onnx.json")
            assert kwargs["use_cuda"] is True
            assert adapter.is_loaded is True
            assert adapter._voice is voice_instance
            assert adapter.list_voices()[0].id == "default"

    def test_load_bootstraps_runtime_when_piper_module_missing(self, tmp_path: Path):
        torch_mock = _mock_torch()
        _write_piper_bundle(tmp_path)
        calls = []
        piper_module = MagicMock()
        voice_cls = MagicMock()
        voice_instance = MagicMock()
        voice_cls.load.return_value = voice_instance
        piper_module.PiperVoice = voice_cls

        def fake_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            mock.stderr = ""
            calls.append(cmd)
            if cmd[0].endswith("uv") or cmd[:3] == [sys.executable, "-m", "pip"]:
                sys.modules["piper"] = piper_module
            return mock

        with patch.dict("sys.modules", {"torch": torch_mock}, clear=False):
            sys.modules.pop("piper", None)
            from vox_piper.adapter import PiperAdapter

            with (
                patch("vox_piper.adapter.importlib.util.find_spec", return_value=None),
                patch("vox_piper.adapter.subprocess.run", side_effect=fake_run),
            ):
                adapter = PiperAdapter()
                adapter.load(str(tmp_path), "cuda", _source="rhasspy/piper-voices")

        assert calls[0][:3] == [sys.executable, "-m", "ensurepip"]
        assert any(cmd[0].endswith("uv") or cmd[:3] == [sys.executable, "-m", "pip"] for cmd in calls[1:])
        assert adapter.is_loaded is True

    def test_synthesize_streams_audio_chunks(self, tmp_path: Path):
        torch_mock = _mock_torch()
        piper_module = MagicMock()
        voice_cls = MagicMock()
        voice_instance = MagicMock()
        syn_config = MagicMock()
        syn_config.speaker_id = None
        syn_config.length_scale = 1.0
        piper_config = MagicMock()
        piper_config.SynthesisConfig.return_value = syn_config

        def synthesize(text, syn_config=None, include_alignments=False):
            assert text == "Hello world"
            assert syn_config is not None
            assert syn_config.speaker_id is None
            assert syn_config.length_scale is not None
            chunk = MagicMock()
            chunk.sample_rate = 22050
            chunk.audio_float_array = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
            return [chunk]

        voice_instance.synthesize.side_effect = synthesize
        voice_cls.load.return_value = voice_instance
        piper_module.PiperVoice = voice_cls
        _write_piper_bundle(tmp_path)

        with patch.dict(
            "sys.modules",
            {"torch": torch_mock, "piper": piper_module, "piper.config": piper_config},
        ):
            from vox_piper.adapter import PiperAdapter

            adapter = PiperAdapter()
            adapter.load(str(tmp_path), "cpu", _source="rhasspy/piper-voices")

            async def run():
                chunks = []
                async for chunk in adapter.synthesize("Hello world"):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(run())

            assert chunks[-1].is_final is True
            assert any(chunk.audio for chunk in chunks[:-1])
            assert voice_instance.synthesize.called

    def test_synthesize_rejects_reference_audio(self, tmp_path: Path):
        torch_mock = _mock_torch()
        piper_module = MagicMock()
        voice_cls = MagicMock()
        voice_instance = MagicMock()
        voice_cls.load.return_value = voice_instance
        piper_module.PiperVoice = voice_cls
        _write_piper_bundle(tmp_path)

        with patch.dict("sys.modules", {"torch": torch_mock, "piper": piper_module}):
            from vox_piper.adapter import PiperAdapter

            adapter = PiperAdapter()
            adapter.load(str(tmp_path), "cpu", _source="rhasspy/piper-voices")

            async def run():
                async for _ in adapter.synthesize(
                    "Hello",
                    reference_audio=np.ones(22050, dtype=np.float32),
                    reference_text="hello",
                ):
                    pass

            with pytest.raises(ValueError, match="reference_audio/reference_text"):
                asyncio.run(run())

    def test_estimate_vram_bytes(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "piper": MagicMock()}):
            from vox_piper.adapter import PiperAdapter

            adapter = PiperAdapter()
            assert adapter.estimate_vram_bytes() == 220_000_000
