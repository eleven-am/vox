from __future__ import annotations

import importlib
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


def test_load_asr_model_class_bootstraps_missing_nemo_runtime(monkeypatch: pytest.MonkeyPatch):
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    fake_model_cls = _install_fake_nemo()
    sys.modules.pop("nemo", None)
    sys.modules.pop("nemo.collections", None)
    sys.modules.pop("nemo.collections.asr", None)

    imported_once = False
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        nonlocal imported_once
        if name == "nemo.collections.asr":
            if not imported_once:
                imported_once = True
                raise ImportError("missing nemo")
            return sys.modules[name]
        return real_import_module(name)

    install_mock = MagicMock(side_effect=lambda: _install_fake_nemo())
    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(module, "_install_nemo_runtime", install_mock)
    monkeypatch.setattr(module, "_clear_nemo_modules", lambda: None)

    loaded_model_cls = module._load_asr_model_class()
    assert loaded_model_cls.__name__ == fake_model_cls.__name__
    assert hasattr(loaded_model_cls, "from_pretrained")
    install_mock.assert_called_once()


def test_load_asr_model_class_prefers_global_nemo_before_local_runtime(monkeypatch: pytest.MonkeyPatch):
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    fake_model_cls = _install_fake_nemo()
    ensure_path = MagicMock()
    install_mock = MagicMock()

    monkeypatch.setattr(module, "_ensure_runtime_target_on_path", ensure_path)
    monkeypatch.setattr(module, "_install_nemo_runtime", install_mock)

    loaded_model_cls = module._load_asr_model_class()

    assert loaded_model_cls.__name__ == fake_model_cls.__name__
    ensure_path.assert_not_called()
    install_mock.assert_not_called()


def test_prime_lightning_imports_attaches_utilities_modules(monkeypatch: pytest.MonkeyPatch):
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    lightning_pytorch = ModuleType("lightning.pytorch")
    lightning_utilities = ModuleType("lightning.pytorch.utilities")
    lightning_imports = ModuleType("lightning.pytorch.utilities.imports")

    def fake_import_module(name: str):
        if name == "lightning.pytorch":
            return lightning_pytorch
        if name == "lightning.pytorch.utilities":
            return lightning_utilities
        if name == "lightning.pytorch.utilities.imports":
            return lightning_imports
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)

    module._prime_lightning_imports()

    assert lightning_pytorch.utilities is lightning_utilities
    assert lightning_utilities.imports is lightning_imports


def test_install_nemo_runtime_bootstraps_pip_when_uv_install_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        mock = MagicMock()
        mock.stderr = ""
        mock.stdout = ""
        calls.append(cmd)
        if cmd[0] == "/usr/bin/uv":
            mock.returncode = 1
            mock.stderr = "uv failed"
        else:
            mock.returncode = 0
        return mock

    monkeypatch.setattr(module, "_runtime_target_dir", lambda: tmp_path / "runtime")
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None if name == "pip" else object())
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._install_nemo_runtime()

    assert calls[0][:5] == ["/usr/bin/uv", "pip", "install", "--python", sys.executable]
    assert "--target" in calls[0]
    assert "nemo-toolkit[asr]" in calls[0]
    assert calls[1][:3] == [sys.executable, "-m", "ensurepip"]
    assert calls[2][:4] == [sys.executable, "-m", "pip", "install"]
    assert "--target" in calls[2]
    assert (tmp_path / "runtime" / ".vox-parakeet-nemo-runtime-ready").is_file()


def test_install_nemo_runtime_uses_pip_when_uv_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        mock = MagicMock()
        mock.stderr = ""
        mock.stdout = ""
        calls.append(cmd)
        mock.returncode = 0
        return mock

    monkeypatch.setattr(module, "_runtime_target_dir", lambda: tmp_path / "runtime")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._install_nemo_runtime()

    assert calls == [
        [sys.executable, "-m", "pip", "install", "--target", str(tmp_path / "runtime"), "nemo-toolkit[asr]"]
    ]
    assert (tmp_path / "runtime" / ".vox-parakeet-nemo-runtime-ready").is_file()


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
