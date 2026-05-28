from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
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


class _SequenceParakeetModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.recognize_calls = 0

    def with_timestamps(self):
        return self

    def recognize(self, _path: str) -> str:
        self.recognize_calls += 1
        if not self.outputs:
            return ""
        return self.outputs.pop(0)


class _ConcurrencyProbeParakeetModel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def with_timestamps(self):
        return self

    def recognize(self, _path: str) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
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


def test_transcribe_serializes_recognizer_calls():
    fake_asr, _fake_ort = _install_fake_modules(providers=["CPUExecutionProvider"])
    model = _ConcurrencyProbeParakeetModel()
    fake_asr.load_model.return_value = model
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()
    adapter.load("ignored-local-path", "cpu")
    audio = np.ones(16_000, dtype=np.float32) * 0.05

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(adapter.transcribe, audio) for _ in range(2)]
        results = [future.result() for future in futures]

    assert [result.text for result in results] == ["hello world", "hello world"]
    assert model.max_active == 1


def test_transcribe_reloads_and_retries_after_repeated_voiced_empty_results():
    fake_asr, _fake_ort = _install_fake_modules(providers=["CPUExecutionProvider"])
    poisoned_model = _SequenceParakeetModel(["", "", ""])
    recovered_model = _SequenceParakeetModel(["recovered speech"])
    fake_asr.load_model.side_effect = [poisoned_model, recovered_model]
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()
    adapter.load("ignored-local-path", "cpu")
    audio = np.ones(16_000, dtype=np.float32) * 0.05

    assert adapter.transcribe(audio).text == ""
    assert adapter.transcribe(audio).text == ""
    assert adapter.transcribe(audio).text == "recovered speech"
    assert fake_asr.load_model.call_count == 2
    assert poisoned_model.recognize_calls == 3
    assert recovered_model.recognize_calls == 1


def test_transcribe_does_not_reload_for_silent_empty_results():
    fake_asr, _fake_ort = _install_fake_modules(providers=["CPUExecutionProvider"])
    silent_model = _SequenceParakeetModel(["", "", ""])
    fake_asr.load_model.return_value = silent_model
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.adapter", None)

    from vox_parakeet.adapter import ParakeetAdapter

    adapter = ParakeetAdapter()
    adapter.load("ignored-local-path", "cpu")
    audio = np.zeros(16_000, dtype=np.float32)

    assert adapter.transcribe(audio).text == ""
    assert adapter.transcribe(audio).text == ""
    assert adapter.transcribe(audio).text == ""
    assert fake_asr.load_model.call_count == 1
