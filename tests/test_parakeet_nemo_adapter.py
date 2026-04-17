from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _restore_torch_and_nemo_modules():
    poisoned_keys = ("torch", "nemo", "nemo.collections", "nemo.collections.asr",
                     "vox_parakeet", "vox_parakeet.nemo_adapter")
    saved = {k: sys.modules.get(k) for k in poisoned_keys}
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class _FakeNemoModel:
    def __init__(self, *, text: str = "hello world", with_timestamps: bool = False) -> None:
        self.text = text
        self.with_timestamps = with_timestamps
        self.to_calls: list[str] = []
        self.eval_called = False
        self.transcribe_calls: list[dict] = []
        self.cfg = SimpleNamespace(
            preprocessor=SimpleNamespace(window_stride=0.02),
        )

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def transcribe(self, paths, **kwargs):
        self.transcribe_calls.append({"paths": paths, **kwargs})
        if kwargs.get("return_hypotheses"):
            return [
                SimpleNamespace(
                    text=self.text,
                    timestamp={
                        "word": [
                            {"word": "hello", "start_offset": 0, "end_offset": 4},
                            {"word": "world", "start_offset": 5, "end_offset": 10},
                        ]
                    },
                )
            ]
        return [self.text]


def _install_fake_nemo(*, model: _FakeNemoModel | None = None):
    fake_module = ModuleType("nemo")
    fake_collections = ModuleType("nemo.collections")
    fake_asr = ModuleType("nemo.collections.asr")

    class _FakeASRModel:
        from_pretrained = MagicMock(return_value=model or _FakeNemoModel())
        restore_from = MagicMock(return_value=model or _FakeNemoModel())

    fake_asr.models = SimpleNamespace(ASRModel=_FakeASRModel)
    fake_collections.asr = fake_asr
    fake_module.collections = fake_collections

    sys.modules["nemo"] = fake_module
    sys.modules["nemo.collections"] = fake_collections
    sys.modules["nemo.collections.asr"] = fake_asr
    return _FakeASRModel


def _install_fake_torch(*, cuda_available: bool = True):
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=MagicMock(return_value=cuda_available),
        empty_cache=MagicMock(),
    )
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=MagicMock(return_value=False))
    )
    fake_torch.inference_mode = lambda: SimpleNamespace(
        __enter__=lambda self=None: None,
        __exit__=lambda self, exc_type, exc, tb: False,
    )
    sys.modules["torch"] = fake_torch
    return fake_torch


def test_info_exposes_pytorch_and_nemo_adapter_name():
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    info = ParakeetNemoAdapter().info()
    assert info.name == "parakeet-stt-nemo"
    assert info.type.value == "stt"
    assert info.supported_formats[0].value == "pytorch"
    assert info.supports_word_timestamps is True
    assert info.supports_language_detection is True


def test_load_uses_pretrained_model_name_when_source_is_provided(monkeypatch: pytest.MonkeyPatch):
    fake_model_cls = _install_fake_nemo()
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    fake_model_cls.from_pretrained.assert_called_once_with(model_name="nvidia/parakeet-tdt-0.6b-v3")
    assert adapter.is_loaded is True
    assert adapter._model_id == "nvidia/parakeet-tdt-0.6b-v3"


def test_load_uses_restore_from_for_local_nemo_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_model_cls = _install_fake_nemo()
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    checkpoint = model_dir / "parakeet-tdt-0.6b-v3.nemo"
    checkpoint.write_bytes(b"fake-nemo")

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load(str(model_dir), "cuda")

    fake_model_cls.restore_from.assert_called_once_with(restore_path=str(checkpoint))
    assert adapter.is_loaded is True
    assert adapter._model_id == str(model_dir)


def test_load_rejects_non_cuda_device(monkeypatch: pytest.MonkeyPatch):
    _install_fake_nemo()
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    with pytest.raises(RuntimeError, match="requires device='cuda' or 'auto'"):
        adapter.load("ignored-local-path", "cpu")


def test_load_rejects_missing_cuda(monkeypatch: pytest.MonkeyPatch):
    _install_fake_nemo()
    _install_fake_torch(cuda_available=False)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    with pytest.raises(RuntimeError, match="requires CUDA"):
        adapter.load("ignored-local-path", "cuda")


def test_transcribe_with_word_timestamps_builds_segment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_model = _FakeNemoModel(text="hello world")
    _install_fake_nemo(model=fake_model)
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    audio = np.zeros(16000, dtype=np.float32)
    result = adapter.transcribe(audio, word_timestamps=True)

    assert result.text == "hello world"
    assert result.duration_ms == 1000
    assert len(result.segments) == 1
    assert result.segments[0].words[0].word == "hello"
    assert result.segments[0].words[1].word == "world"
    assert fake_model.transcribe_calls[-1]["timestamps"] is True
    assert fake_model.transcribe_calls[-1]["return_hypotheses"] is True


def test_transcribe_without_word_timestamps_returns_text(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeNemoModel(text="plain text")
    _install_fake_nemo(model=fake_model)
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    audio = np.zeros(16000, dtype=np.float32)
    result = adapter.transcribe(audio, word_timestamps=False)

    assert result.text == "plain text"
    assert result.segments[0].text == "plain text"
    assert "timestamps" not in fake_model.transcribe_calls[-1]
