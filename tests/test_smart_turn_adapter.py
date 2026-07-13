from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vox.core.types import ModelType


def _adapter_class():
    from vox_smart_turn.adapter import SmartTurnV3Adapter

    return SmartTurnV3Adapter


def test_smart_turn_adapter_metadata():
    info = _adapter_class()().info()

    assert info.type is ModelType.TURN
    assert info.default_sample_rate == 16_000
    assert info.architectures == ("smart-turn-v3",)


def test_smart_turn_predict_uses_last_eight_seconds():
    adapter = _adapter_class()()

    extractor = MagicMock()
    extractor.return_value = SimpleNamespace(
        input_features=np.ones((1, 80, 800), dtype=np.float32)
    )
    session = MagicMock()
    session.run.return_value = [np.array([[0.82]], dtype=np.float32)]
    adapter._feature_extractor = extractor
    adapter._session = session

    probability = adapter.predict(
        np.ones(10 * 16_000, dtype=np.float32),
        sample_rate=16_000,
    )

    assert probability == pytest.approx(0.82)
    assert extractor.call_args.args[0].shape == (8 * 16_000,)
    assert session.run.call_args.args[1]["input_features"].shape == (1, 80, 800)


def test_smart_turn_load_rejects_missing_provider(monkeypatch, tmp_path: Path):
    module = __import__("vox_smart_turn.adapter", fromlist=["SmartTurnV3Adapter"])
    adapter = module.SmartTurnV3Adapter()

    monkeypatch.setattr(module, "_ensure_runtime", lambda: tmp_path)
    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)
    fake_transformers = SimpleNamespace(WhisperFeatureExtractor=MagicMock())
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        adapter.load(str(tmp_path), "cuda", provider="gpu")


def test_smart_turn_prepare_runtime_delegates(monkeypatch):
    module = __import__("vox_smart_turn.adapter", fromlist=["SmartTurnV3Adapter"])

    ensure = MagicMock()
    monkeypatch.setattr(module, "_ensure_runtime", ensure)
    module.SmartTurnV3Adapter().prepare_runtime()
    ensure.assert_called_once_with()
