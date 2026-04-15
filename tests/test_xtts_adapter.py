from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


class TestXTTSAdapter:
    def test_package_import_does_not_require_runtime(self):
        module = importlib.import_module("vox_xtts")
        assert module.__all__ == ["XTTSAdapter"]

    def test_info_returns_correct_metadata(self):
        from vox_xtts.adapter import XTTSAdapter

        adapter = XTTSAdapter()
        info = adapter.info()

        assert info.name == "xtts"
        assert info.type == ModelType.TTS
        assert "xtts-v2" in info.architectures
        assert info.default_sample_rate == 24_000
        assert ModelFormat.PYTORCH in info.supported_formats
        assert info.supports_voice_cloning is True
        assert info.supports_streaming is False
        assert info.supported_languages == ()

    def test_load_uses_coqui_tts_and_puts_model_in_eval_mode(self):
        from vox_xtts.adapter import XTTSAdapter

        tts_instance = MagicMock()
        tts_instance.to.return_value = tts_instance
        tts_instance.eval.return_value = tts_instance
        torch_runtime = MagicMock()
        torch_runtime.cuda.is_available.return_value = True

        with (
            patch("vox_xtts.adapter._load_tts_runtime", return_value=MagicMock(return_value=tts_instance)) as load_rt,
            patch("vox_xtts.adapter._load_torch_runtime", return_value=torch_runtime),
        ):
            adapter = XTTSAdapter()
            adapter.load("ignored", "cuda", _source="tts_models/multilingual/multi-dataset/xtts_v2")

        load_rt.assert_called_once()
        adapter._tts.to.assert_called_once_with("cuda")
        adapter._tts.eval.assert_called_once()
        assert adapter.is_loaded is True
        assert adapter._model_id == "tts_models/multilingual/multi-dataset/xtts_v2"

    def test_synthesize_requires_reference_audio_or_voice_path(self):
        from vox_xtts.adapter import XTTSAdapter

        adapter = XTTSAdapter()
        adapter._loaded = True
        adapter._tts = MagicMock()

        with pytest.raises(ValueError, match="requires reference_audio or a voice wav path"):
            asyncio.run(self._collect(adapter.synthesize("hello")))

    def test_synthesize_uses_voice_path(self, tmp_path: Path):
        from vox_xtts.adapter import XTTSAdapter

        ref_path = tmp_path / "ref.wav"
        ref_path.write_bytes(b"RIFF0000WAVE")

        tts_instance = MagicMock()
        tts_instance.tts.return_value = np.linspace(-1.0, 1.0, 48000, dtype=np.float32)

        adapter = XTTSAdapter()
        adapter._loaded = True
        adapter._tts = tts_instance

        async def collect() -> list:
            return [chunk async for chunk in adapter.synthesize("hello world", voice=str(ref_path), language="en")]

        chunks = asyncio.run(collect())

        tts_instance.tts.assert_called_once()
        kwargs = tts_instance.tts.call_args.kwargs
        assert kwargs["speaker_wav"] == str(ref_path)
        assert kwargs["language"] == "en"
        assert len(chunks) == 2
        assert chunks[0].sample_rate == 24_000
        assert chunks[-1].is_final is True

    def test_synthesize_with_reference_audio_writes_temp_reference(self, tmp_path: Path):
        from vox_xtts.adapter import XTTSAdapter

        reference_audio = np.zeros(16000, dtype=np.float32)
        tts_instance = MagicMock()
        tts_instance.tts.return_value = np.linspace(-1.0, 1.0, 24000, dtype=np.float32)

        adapter = XTTSAdapter()
        adapter._loaded = True
        adapter._tts = tts_instance

        class DummyTmp:
            def __init__(self, path: Path) -> None:
                self.name = str(path)

            def cleanup(self) -> None:
                return None

        with (
            patch("vox_xtts.adapter.tempfile.TemporaryDirectory", return_value=DummyTmp(tmp_path)),
            patch("vox_xtts.adapter.sf.write") as write_mock,
        ):
            async def collect() -> list:
                return [
                    chunk
                    async for chunk in adapter.synthesize("hello world", reference_audio=reference_audio, language="en")
                ]

            chunks = asyncio.run(collect())

        write_mock.assert_called_once()
        kwargs = tts_instance.tts.call_args.kwargs
        assert kwargs["speaker_wav"].endswith("reference.wav")
        assert len(chunks) == 2

    def test_unload_resets_state(self):
        from vox_xtts.adapter import XTTSAdapter

        adapter = XTTSAdapter()
        adapter._loaded = True
        adapter._tts = MagicMock()

        adapter.unload()

        assert adapter.is_loaded is False
        assert adapter._tts is None

    def test_estimate_vram(self):
        from vox_xtts.adapter import XTTSAdapter

        adapter = XTTSAdapter()
        assert adapter.estimate_vram_bytes() == 4_000_000_000

    @staticmethod
    async def _collect(aiter):
        return [item async for item in aiter]


def test_registry_contains_xtts_catalog_entry():
    from vox.core.registry import CATALOG

    entry = CATALOG["xtts"]["v2"]
    assert entry["source"] == "tts_models/multilingual/multi-dataset/xtts_v2"
    assert entry["adapter"] == "xtts"
    assert entry["format"] == "pytorch"
    assert entry["parameters"]["sample_rate"] == 24_000
