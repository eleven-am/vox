from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class _FakeParakeetModel:
    def __init__(self) -> None:
        self.timestamps_called = False

    def with_timestamps(self):
        self.timestamps_called = True
        return self

    def recognize(self, _path: str) -> str:
        return "hello world"


def _install_fake_modules(*, providers=None):
    fake_asr = ModuleType("onnx_asr")
    fake_asr.load_model = MagicMock(return_value=_FakeParakeetModel())
    fake_asr.adapters = SimpleNamespace(
        TextResultsAsrAdapter=object,
        TimestampedResultsAsrAdapter=object,
    )

    fake_ort = ModuleType("onnxruntime")
    fake_ort.get_available_providers = MagicMock(
        return_value=providers or ["CPUExecutionProvider"]
    )

    sys.modules["onnx_asr"] = fake_asr
    sys.modules["onnxruntime"] = fake_ort
    return fake_asr, fake_ort


def test_load_uses_source_repo_id_and_cuda_provider():
    fake_asr, _fake_ort = _install_fake_modules(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    fake_asr.load_model.assert_called_once_with(
        "nemo-parakeet-tdt-0.6b-v3",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert adapter.is_loaded is True
    assert adapter._model_id == "nemo-parakeet-tdt-0.6b-v3"


def test_load_keeps_local_model_path_unmodified(tmp_path: Path):
    fake_asr, _fake_ort = _install_fake_modules()
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    local_model_dir = tmp_path / "parakeet"
    local_model_dir.mkdir()

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()
    adapter.load(str(local_model_dir), "cpu")

    fake_asr.load_model.assert_called_once_with(
        "nemo-parakeet-tdt-0.6b-v3",
        path=str(local_model_dir),
        providers=["CPUExecutionProvider"],
    )
    assert adapter.is_loaded is True
    assert adapter._model_id == "nemo-parakeet-tdt-0.6b-v3"


def test_load_uses_source_repo_id_with_local_model_path(tmp_path: Path):
    fake_asr, _fake_ort = _install_fake_modules(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    local_model_dir = tmp_path / "parakeet"
    local_model_dir.mkdir()

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()
    adapter.load(
        str(local_model_dir),
        "cuda",
        _source="nvidia/parakeet-tdt-0.6b-v3",
    )

    fake_asr.load_model.assert_called_once_with(
        "nemo-parakeet-tdt-0.6b-v3",
        path=str(local_model_dir),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert adapter._model_id == "nemo-parakeet-tdt-0.6b-v3"


def test_load_rejects_cuda_without_provider():
    fake_asr, _fake_ort = _install_fake_modules(providers=["CPUExecutionProvider"])
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()

    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    fake_asr.load_model.assert_not_called()


def test_load_auto_falls_back_to_cpu_when_cuda_provider_is_missing():
    fake_asr, _fake_ort = _install_fake_modules(providers=["CPUExecutionProvider"])
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()
    adapter.load("ignored-local-path", "auto", _source="nvidia/parakeet-tdt-0.6b-v3")

    fake_asr.load_model.assert_called_once_with(
        "nemo-parakeet-tdt-0.6b-v3",
        providers=["CPUExecutionProvider"],
    )
    assert adapter._device == "cpu"


def test_transcribe_accepts_english_locale_without_warning(tmp_path: Path):
    _install_fake_modules(providers=["CPUExecutionProvider"])
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()
    adapter.load("ignored-local-path", "cpu")

    with patch("vox_parakeet.adapter.logger.warning") as warning:
        result = adapter.transcribe(np.zeros(16_000, dtype=np.float32), language="en-us")

    assert result.language == "en"
    warning.assert_not_called()
