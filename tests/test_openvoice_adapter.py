from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


def _clear_openvoice_modules() -> None:
    sys.modules.pop("vox_openvoice", None)
    sys.modules.pop("vox_openvoice.adapter", None)
    sys.modules.pop("openvoice", None)
    sys.modules.pop("openvoice.api", None)


class TestOpenVoiceAdapterInfo:
    def test_package_import_does_not_require_runtime_package(self):
        torch = MagicMock()
        with patch.dict("sys.modules", {"torch": torch, "librosa": MagicMock(), "soundfile": MagicMock()}):
            _clear_openvoice_modules()
            module = importlib.import_module("vox_openvoice")
            assert module.__all__ == ["OpenVoiceTTSAdapter"]

    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"torch": MagicMock(), "librosa": MagicMock(), "soundfile": MagicMock()}):
            _clear_openvoice_modules()
            from vox_openvoice.adapter import OpenVoiceTTSAdapter

            adapter = OpenVoiceTTSAdapter()
            info = adapter.info()

            assert info.name == "openvoice-tts-torch"
            assert info.type == ModelType.TTS
            assert "openvoice" in info.architectures
            assert info.default_sample_rate == 22_050
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_voice_cloning is True
            assert info.supported_languages == ("en", "zh")

    def test_list_voices_returns_both_languages(self):
        with patch.dict("sys.modules", {"torch": MagicMock(), "librosa": MagicMock(), "soundfile": MagicMock()}):
            _clear_openvoice_modules()
            from vox_openvoice.adapter import OpenVoiceTTSAdapter

            adapter = OpenVoiceTTSAdapter()
            voices = adapter.list_voices()
            ids = {v.id for v in voices}

            assert "en/default" in ids
            assert "zh/default" in ids

    def test_load_uses_openvoice_runtime_classes(self, tmp_path: Path):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.backends.mps.is_available.return_value = False
        torch.float32 = "float32"

        api = ModuleType("openvoice.api")
        base_cls = MagicMock()
        converter_cls = MagicMock()
        converter = MagicMock()
        converter.load_ckpt.return_value = None
        converter.hps.data.sampling_rate = 22_050
        converter_cls.return_value = converter
        api.BaseSpeakerTTS = base_cls
        api.ToneColorConverter = converter_cls

        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "librosa": MagicMock(),
                "soundfile": MagicMock(),
                "openvoice": ModuleType("openvoice"),
                "openvoice.api": api,
            },
        ):
            _clear_openvoice_modules()
            sys.modules["openvoice"] = ModuleType("openvoice")
            sys.modules["openvoice.api"] = api
            from vox_openvoice.adapter import OpenVoiceTTSAdapter

            adapter = OpenVoiceTTSAdapter()
            model_root = tmp_path / "openvoice"
            (model_root / "checkpoints" / "base_speakers" / "EN").mkdir(parents=True, exist_ok=True)
            (model_root / "checkpoints" / "base_speakers" / "EN" / "config.json").write_text("{}", encoding="utf-8")
            (model_root / "checkpoints" / "base_speakers" / "EN" / "checkpoint.pth").write_bytes(b"checkpoint")
            (model_root / "checkpoints" / "converter").mkdir(parents=True, exist_ok=True)
            (model_root / "checkpoints" / "converter" / "config.json").write_text("{}", encoding="utf-8")
            (model_root / "checkpoints" / "converter" / "checkpoint.pth").write_bytes(b"checkpoint")

            with patch("vox_openvoice.adapter._install_openvoice_runtime") as install_mock:
                adapter.load(str(model_root), "cuda")

            install_mock.assert_not_called()
            converter_cls.assert_called_once()
            converter.load_ckpt.assert_called_once()
            assert adapter.is_loaded is True

    def test_load_ignores_source_repo_when_resolving_local_files(self, tmp_path: Path):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False

        api = ModuleType("openvoice.api")
        base_cls = MagicMock()
        converter_cls = MagicMock()
        converter = MagicMock()
        converter.load_ckpt.return_value = None
        converter.hps.data.sampling_rate = 22_050
        converter_cls.return_value = converter
        api.BaseSpeakerTTS = base_cls
        api.ToneColorConverter = converter_cls

        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "librosa": MagicMock(),
                "soundfile": MagicMock(),
                "openvoice": ModuleType("openvoice"),
                "openvoice.api": api,
            },
        ):
            _clear_openvoice_modules()
            sys.modules["openvoice"] = ModuleType("openvoice")
            sys.modules["openvoice.api"] = api
            from vox_openvoice.adapter import OpenVoiceTTSAdapter

            adapter = OpenVoiceTTSAdapter()
            model_root = tmp_path / "openvoice"
            (model_root / "checkpoints" / "base_speakers" / "EN").mkdir(parents=True, exist_ok=True)
            (model_root / "checkpoints" / "base_speakers" / "EN" / "config.json").write_text("{}", encoding="utf-8")
            (model_root / "checkpoints" / "base_speakers" / "EN" / "checkpoint.pth").write_bytes(b"checkpoint")
            (model_root / "checkpoints" / "converter").mkdir(parents=True, exist_ok=True)
            (model_root / "checkpoints" / "converter" / "config.json").write_text("{}", encoding="utf-8")
            (model_root / "checkpoints" / "converter" / "checkpoint.pth").write_bytes(b"checkpoint")

            adapter.load(str(model_root), "cpu", _source="myshell-ai/OpenVoice")

            assert adapter._model_root == model_root

    def test_load_raises_clear_error_when_runtime_missing(self, tmp_path: Path):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": torch, "librosa": MagicMock(), "soundfile": MagicMock()}):
            _clear_openvoice_modules()
            from vox_openvoice.adapter import OpenVoiceTTSAdapter

            adapter = OpenVoiceTTSAdapter()
            model_root = tmp_path / "openvoice"
            (model_root / "checkpoints" / "converter").mkdir(parents=True, exist_ok=True)
            (model_root / "checkpoints" / "converter" / "config.json").write_text("{}", encoding="utf-8")
            (model_root / "checkpoints" / "converter" / "checkpoint.pth").write_bytes(b"checkpoint")

            with (
                patch("vox_openvoice.adapter._install_openvoice_runtime", side_effect=RuntimeError("no runtime")),
                pytest.raises(RuntimeError, match="no runtime"),
            ):
                adapter.load(str(model_root), "cpu")

    def test_install_runtime_bootstraps_pip_before_git_install(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        calls = []

        def fake_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            mock.stderr = ""
            calls.append(cmd)
            return mock

        with patch.dict("sys.modules", {"torch": torch, "librosa": MagicMock(), "soundfile": MagicMock()}):
            _clear_openvoice_modules()
            from vox_openvoice.adapter import _install_openvoice_runtime

            with (
                patch("vox_openvoice.adapter.importlib.util.find_spec", return_value=None),
                patch("vox_openvoice.adapter.subprocess.run", side_effect=fake_run),
            ):
                _install_openvoice_runtime()

        assert calls[0][0].endswith("uv")
        assert calls[0][1:5] == ["pip", "install", "--python", sys.executable]
        assert "--no-build-isolation" in calls[0]
        assert "--no-deps" in calls[0]
        assert "resampy==0.4.3" in calls[0]


class TestOpenVoiceAdapterSynthesis:
    def test_synthesize_emits_chunks(self, tmp_path: Path):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        torch.float32 = "float32"
        inference_ctx = torch.inference_mode.return_value
        inference_ctx.__enter__.return_value = None
        inference_ctx.__exit__.return_value = False

        api = ModuleType("openvoice.api")
        base_cls = MagicMock()
        base_model = MagicMock()
        base_model.tts.return_value = np.arange(44_100, dtype=np.float32)
        base_model.hps.data.sampling_rate = 22_050
        base_cls.return_value = base_model

        converter_cls = MagicMock()
        converter = MagicMock()
        converter.hps.data.sampling_rate = 22_050
        converter.convert.return_value = np.arange(22_050, dtype=np.float32)
        converter_cls.return_value = converter
        api.BaseSpeakerTTS = base_cls
        api.ToneColorConverter = converter_cls

        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "librosa": MagicMock(),
                "soundfile": MagicMock(),
                "openvoice": ModuleType("openvoice"),
                "openvoice.api": api,
            },
        ):
            _clear_openvoice_modules()
            sys.modules["openvoice"] = ModuleType("openvoice")
            sys.modules["openvoice.api"] = api
            from vox_openvoice.adapter import OpenVoiceTTSAdapter

            adapter = OpenVoiceTTSAdapter()
            adapter._loaded = True
            adapter._converter = converter
            adapter._model_root = tmp_path / "openvoice"
            adapter._BaseSpeakerTTS = base_cls  # type: ignore[attr-defined]

            base_dir = adapter._model_root / "checkpoints" / "base_speakers" / "EN"
            base_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / "config.json").write_text("{}", encoding="utf-8")
            (base_dir / "checkpoint.pth").write_bytes(b"checkpoint")

            converter_dir = adapter._model_root / "checkpoints" / "converter"
            converter_dir.mkdir(parents=True, exist_ok=True)
            (converter_dir / "config.json").write_text("{}", encoding="utf-8")
            (converter_dir / "checkpoint.pth").write_bytes(b"checkpoint")

            import asyncio

            chunks: list = []

            async def collect() -> None:
                async for chunk in adapter.synthesize("Hello OpenVoice"):
                    chunks.append(chunk)

            asyncio.run(collect())

            assert len(chunks) == 2
            assert chunks[0].audio
            assert chunks[-1].is_final is True

    def test_synthesize_uses_reference_audio_conversion(self, tmp_path: Path):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        torch.float32 = "float32"
        inference_ctx = torch.inference_mode.return_value
        inference_ctx.__enter__.return_value = None
        inference_ctx.__exit__.return_value = False

        api = ModuleType("openvoice.api")
        base_cls = MagicMock()
        base_model = MagicMock()
        base_model.tts.return_value = np.arange(44_100, dtype=np.float32)
        base_model.hps.data.sampling_rate = 22_050
        base_cls.return_value = base_model

        converter_cls = MagicMock()
        converter = MagicMock()
        converter.hps.data.sampling_rate = 22_050
        converter.convert.return_value = np.arange(22_050, dtype=np.float32)
        converter_cls.return_value = converter
        api.BaseSpeakerTTS = base_cls
        api.ToneColorConverter = converter_cls

        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "librosa": MagicMock(),
                "soundfile": MagicMock(),
                "openvoice": ModuleType("openvoice"),
                "openvoice.api": api,
            },
        ):
            _clear_openvoice_modules()
            sys.modules["openvoice"] = ModuleType("openvoice")
            sys.modules["openvoice.api"] = api
            from vox_openvoice.adapter import OpenVoiceTTSAdapter

            adapter = OpenVoiceTTSAdapter()
            adapter._loaded = True
            adapter._converter = converter
            adapter._model_root = tmp_path / "openvoice"
            adapter._BaseSpeakerTTS = base_cls  # type: ignore[attr-defined]

            base_dir = adapter._model_root / "checkpoints" / "base_speakers" / "EN"
            base_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / "config.json").write_text("{}", encoding="utf-8")
            (base_dir / "checkpoint.pth").write_bytes(b"checkpoint")

            converter_dir = adapter._model_root / "checkpoints" / "converter"
            converter_dir.mkdir(parents=True, exist_ok=True)
            (converter_dir / "config.json").write_text("{}", encoding="utf-8")
            (converter_dir / "checkpoint.pth").write_bytes(b"checkpoint")

            import asyncio

            chunks: list = []

            async def collect() -> None:
                async for chunk in adapter.synthesize(
                    "Hello OpenVoice",
                    reference_audio=np.zeros(22_050, dtype=np.float32),
                ):
                    chunks.append(chunk)

            asyncio.run(collect())

            assert converter.extract_se.call_count == 2
            assert converter.convert.called

    def test_synthesize_rejects_unsupported_language(self, tmp_path: Path):
        with patch.dict("sys.modules", {"torch": MagicMock(), "librosa": MagicMock(), "soundfile": MagicMock()}):
            _clear_openvoice_modules()
            from vox_openvoice.adapter import OpenVoiceTTSAdapter

            adapter = OpenVoiceTTSAdapter()
            adapter._loaded = True
            adapter._converter = MagicMock()
            adapter._model_root = tmp_path / "openvoice"
            adapter._BaseSpeakerTTS = MagicMock()  # type: ignore[attr-defined]

            import asyncio

            async def collect() -> None:
                async for _ in adapter.synthesize("Hello", language="fr"):
                    pass

            with pytest.raises(ValueError, match="English and Chinese"):
                asyncio.run(collect())
