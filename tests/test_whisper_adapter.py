from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from vox.core.types import ModelFormat, ModelType


def _mock_torch(cuda_available: bool = True):
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = cuda_available
    torch_mock.backends.mps.is_available.return_value = False
    torch_mock.float16 = object()
    torch_mock.float32 = object()
    torch_mock.inference_mode.return_value.__enter__.return_value = None
    torch_mock.inference_mode.return_value.__exit__.return_value = False
    return torch_mock


def _mock_faster_whisper_module():
    module = MagicMock()
    model_cls = MagicMock()
    module.WhisperModel = model_cls
    return module, model_cls


def _mock_ctranslate2(cuda_count: int = 1):
    module = MagicMock()
    module.get_cuda_device_count.return_value = cuda_count
    return module


class TestWhisperAdapterInfo:
    def test_package_import_does_not_require_runtime(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "faster_whisper": MagicMock()}):
            from vox_whisper import WhisperAdapter

            assert WhisperAdapter.__name__ == "WhisperAdapter"

    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "faster_whisper": MagicMock()}):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            info = adapter.info()

            assert info.name == "whisper-stt-ct2"
            assert info.type == ModelType.STT
            assert info.architectures == ("whisper-stt-ct2", "whisper")
            assert info.default_sample_rate == 16_000
            assert info.supported_formats == (ModelFormat.CT2,)
            assert info.supports_streaming is False
            assert info.supports_word_timestamps is True
            assert info.supports_language_detection is True

    def test_load_uses_faster_whisper_gpu_runtime(self):
        torch_mock = _mock_torch()
        fw_module, model_cls = _mock_faster_whisper_module()
        model_cls.return_value = MagicMock()

        with patch.dict("sys.modules", {"torch": torch_mock, "faster_whisper": fw_module}):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter.load("local-model", "cuda", _source="Systran/faster-whisper-large-v3", beam_size=7)

            model_cls.assert_called_once_with("Systran/faster-whisper-large-v3", device="cuda", compute_type="float16")
            assert adapter._beam_size == 7
            assert adapter.is_loaded is True

    def test_bootstraps_missing_runtime_dependencies(self, tmp_path: Path):
        fw_module, model_cls = _mock_faster_whisper_module()
        model_cls.return_value = MagicMock()
        ct2_module = _mock_ctranslate2(cuda_count=1)

        def _install_side_effect(*_args, **_kwargs):
            sys.modules["faster_whisper"] = fw_module
            sys.modules["ctranslate2"] = ct2_module
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch.dict("sys.modules", {"torch": _mock_torch()}),
            patch.dict(os.environ, {"VOX_HOME": str(tmp_path)}),
            patch("vox_whisper.adapter.importlib.util.find_spec", return_value=None),
            patch("vox_whisper.adapter.subprocess.run", side_effect=_install_side_effect) as subprocess_run,
        ):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter.load("local-model", "cuda", _source="Systran/faster-whisper-large-v3")

            subprocess_run.assert_called_once()
            assert adapter.is_loaded is True

    def test_bootstrap_uses_private_runtime_dir(self, tmp_path: Path):
        fw_module, model_cls = _mock_faster_whisper_module()
        model_cls.return_value = MagicMock()
        ct2_module = _mock_ctranslate2(cuda_count=1)

        def _install_side_effect(*_args, **_kwargs):
            sys.modules["faster_whisper"] = fw_module
            sys.modules["ctranslate2"] = ct2_module
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch.dict("sys.modules", {"torch": _mock_torch()}),
            patch.dict(os.environ, {"VOX_HOME": str(tmp_path)}),
            patch("vox_whisper.adapter.importlib.util.find_spec", return_value=None),
            patch("vox_whisper.adapter.subprocess.run", side_effect=_install_side_effect) as subprocess_run,
        ):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter.load("local-model", "cuda", _source="Systran/faster-whisper-large-v3")

            target_dir = tmp_path / "runtime" / "whisper"
            install_cmd = subprocess_run.call_args.args[0]
            assert str(target_dir) in install_cmd
            assert sys.path[0] == str(target_dir)

    def test_load_falls_back_to_cpu_when_ct2_cuda_runtime_is_missing(self):
        torch_mock = _mock_torch(cuda_available=True)
        fw_module, model_cls = _mock_faster_whisper_module()
        model_cls.side_effect = [
            RuntimeError("This CTranslate2 package was not compiled with CUDA support"),
            MagicMock(),
        ]

        with patch.dict("sys.modules", {"torch": torch_mock, "faster_whisper": fw_module}):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter.load("local-model", "cuda", _source="Systran/faster-whisper-base.en")

            assert model_cls.call_count == 2
            first_call = model_cls.call_args_list[0]
            second_call = model_cls.call_args_list[1]
            assert first_call.args == ("Systran/faster-whisper-base.en",)
            assert first_call.kwargs == {"device": "cuda", "compute_type": "float16"}
            assert second_call.args == ("Systran/faster-whisper-base.en",)
            assert second_call.kwargs == {"device": "cpu", "compute_type": "int8"}

    def test_transcribe_builds_segments_and_words(self):
        torch_mock = _mock_torch()
        fw_module, model_cls = _mock_faster_whisper_module()
        model_instance = MagicMock()
        model_cls.return_value = model_instance
        model_instance.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(
                        text="hello",
                        start=0.0,
                        end=0.5,
                        words=[SimpleNamespace(word="hello", start=0.0, end=0.5, probability=0.9)],
                    ),
                    SimpleNamespace(
                        text="world",
                        start=0.5,
                        end=1.0,
                        words=[SimpleNamespace(word="world", start=0.5, end=1.0, probability=0.8)],
                    ),
                ]
            ),
            SimpleNamespace(language="en"),
        )

        with patch.dict("sys.modules", {"torch": torch_mock, "faster_whisper": fw_module}):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter.load("local-model", "cuda", _source="Systran/faster-whisper-large-v3")
            result = adapter.transcribe(np.ones(16000, dtype=np.float32), word_timestamps=True)

            assert result.text == "hello world"
            assert result.language == "en"
            assert result.duration_ms == 1000
            assert len(result.segments) == 2
            assert result.segments[0].words[0].word == "hello"
            assert result.segments[1].words[0].word == "world"

    def test_detect_language_returns_runtime_language(self):
        torch_mock = _mock_torch()
        fw_module, model_cls = _mock_faster_whisper_module()
        model_instance = MagicMock()
        model_cls.return_value = model_instance
        model_instance.transcribe.return_value = (iter([]), SimpleNamespace(language="fr"))

        with patch.dict("sys.modules", {"torch": torch_mock, "faster_whisper": fw_module}):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter.load("local-model", "cuda", _source="Systran/faster-whisper-large-v3")

            assert adapter.detect_language(np.ones(16000, dtype=np.float32)) == "fr"

    def test_empty_audio_returns_empty_result(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "faster_whisper": MagicMock()}):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._model_id = "Systran/faster-whisper-base.en"

            result = adapter.transcribe(np.array([], dtype=np.float32))
            assert result.text == ""
            assert result.duration_ms == 0

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "faster_whisper": MagicMock()}):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._model_id = "Systran/faster-whisper-large-v3"

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None

    def test_estimate_vram_bytes(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "faster_whisper": MagicMock()}):
            from vox_whisper.adapter import WhisperAdapter

            adapter = WhisperAdapter()
            adapter._model_id = "Systran/faster-whisper-large-v3"
            assert adapter.estimate_vram_bytes() == 4_000_000_000
            assert adapter.estimate_vram_bytes(_source="Systran/faster-whisper-large-v3-turbo") == 2_500_000_000
