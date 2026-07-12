from __future__ import annotations

import importlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
        self.transcribe_errors: list[Exception] = []
        self.disable_cuda_graph_calls = 0
        self.decoding = SimpleNamespace(
            decoding=SimpleNamespace(
                decoding_computer=SimpleNamespace(
                    disable_cuda_graphs=self._disable_cuda_graphs,
                    use_cuda_graph_decoder=True,
                )
            )
        )
        self.cfg = SimpleNamespace(
            preprocessor=SimpleNamespace(window_stride=0.02),
        )

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def _disable_cuda_graphs(self):
        self.disable_cuda_graph_calls += 1
        return True

    def transcribe(self, paths, **kwargs):
        self.transcribe_calls.append({"paths": paths, **kwargs})
        if self.transcribe_errors:
            raise self.transcribe_errors.pop(0)
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


class _ConcurrentDetectingNemoModel(_FakeNemoModel):
    def __init__(self) -> None:
        super().__init__(text="serialized")
        self.active_calls = 0
        self.max_active_calls = 0

    def transcribe(self, paths, **kwargs):
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.05)
            return super().transcribe(paths, **kwargs)
        finally:
            self.active_calls -= 1


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
    fake_model = _FakeNemoModel()
    fake_model_cls = _install_fake_nemo(model=fake_model)
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    fake_model_cls.from_pretrained.assert_called_once_with(model_name="nvidia/parakeet-tdt-0.6b-v3")
    assert adapter.is_loaded is True
    assert adapter._model_id == "nvidia/parakeet-tdt-0.6b-v3"
    assert fake_model.disable_cuda_graph_calls == 1
    assert fake_model.decoding.decoding.decoding_computer.use_cuda_graph_decoder is False


def test_load_prefers_pretrained_source_over_pulled_local_nemo_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_model = _FakeNemoModel()
    fake_model_cls = _install_fake_nemo(model=fake_model)
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    checkpoint = model_dir / "parakeet-tdt-0.6b-v3.nemo"
    checkpoint.write_bytes(b"fake-nemo")

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load(str(model_dir), "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    fake_model_cls.from_pretrained.assert_called_once_with(model_name="nvidia/parakeet-tdt-0.6b-v3")
    fake_model_cls.restore_from.assert_not_called()
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
    calls: list[str] = []
    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        module,
        "_ensure_runtime_target_on_path",
        lambda: calls.append("ensure_runtime"),
    )
    monkeypatch.setattr(module, "_install_nemo_runtime", install_mock)
    monkeypatch.setattr(module, "_clear_nemo_modules", lambda: calls.append("clear"))
    monkeypatch.setattr(module, "_prime_lightning_imports", lambda: calls.append("prime"))

    loaded_model_cls = module._load_asr_model_class()
    assert loaded_model_cls.__name__ == fake_model_cls.__name__
    assert hasattr(loaded_model_cls, "from_pretrained")
    install_mock.assert_called_once()
    assert calls == ["clear", "ensure_runtime", "clear", "prime"]


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


def test_install_nemo_runtime_uses_python312_compatible_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_runtime_module_available", lambda name: name == "nemo.collections.asr")

    module._install_nemo_runtime()

    assert calls[0][:5] == ["uv", "pip", "install", "--python", sys.executable]
    assert "--target" in calls[0]
    assert "nemo-toolkit[asr]==2.7.3" in calls[0]
    assert "numpy>=1.26,<2" in calls[0]
    assert "numba>=0.61,<0.67" in calls[0]
    assert "llvmlite>=0.44,<0.49" in calls[0]
    assert "matplotlib>=3.9,<3.10" in calls[0]
    assert len(calls) == 1
    assert (tmp_path / "runtime" / ".vox-parakeet-nemo-runtime-ready").is_file()


def test_install_nemo_runtime_removes_base_framework_shadows_but_keeps_runtime_triton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    runtime_dir = tmp_path / "runtime"

    def fake_run(cmd, **kwargs):
        (runtime_dir / "torch").mkdir(parents=True, exist_ok=True)
        (runtime_dir / "torch-2.9.1.dist-info").mkdir()
        (runtime_dir / "triton").mkdir()
        (runtime_dir / "triton-3.5.1.dist-info").mkdir()
        (runtime_dir / "nvidia").mkdir()
        (runtime_dir / "nvidia_cublas_cu12-12.8.4.1.dist-info").mkdir()
        mock = MagicMock()
        mock.stderr = ""
        mock.stdout = ""
        mock.returncode = 0
        return mock

    installed = False

    def fake_module_available(name: str) -> bool:
        return installed and name == "nemo.collections.asr"

    original_fake_run = fake_run

    def fake_run_and_mark_installed(cmd, **kwargs):
        nonlocal installed
        result = original_fake_run(cmd, **kwargs)
        installed = True
        return result

    monkeypatch.setattr(module, "_runtime_target_dir", lambda: runtime_dir)
    monkeypatch.setattr(module.subprocess, "run", fake_run_and_mark_installed)
    monkeypatch.setattr(module, "_runtime_module_available", fake_module_available)

    module._install_nemo_runtime()

    assert not (runtime_dir / "torch").exists()
    assert not (runtime_dir / "torch-2.9.1.dist-info").exists()
    assert (runtime_dir / "triton").exists()
    assert (runtime_dir / "triton-3.5.1.dist-info").exists()
    assert not (runtime_dir / "nvidia").exists()
    assert not (runtime_dir / "nvidia_cublas_cu12-12.8.4.1.dist-info").exists()
    assert (runtime_dir / ".vox-parakeet-nemo-runtime-ready").is_file()


def test_install_nemo_runtime_repairs_stale_framework_shadows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / ".vox-parakeet-nemo-runtime-ready").touch()
    (runtime_dir / "torch").mkdir()
    (runtime_dir / "torch-2.9.1.dist-info").mkdir()
    install = MagicMock(return_value=True)

    monkeypatch.setattr(module, "_runtime_target_dir", lambda: runtime_dir)
    monkeypatch.setattr(module, "_runtime_module_available", lambda name: name == "nemo.collections.asr")
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)

    module._install_nemo_runtime()

    assert not (runtime_dir / "torch").exists()
    assert not (runtime_dir / "torch-2.9.1.dist-info").exists()
    install.assert_not_called()


def test_install_nemo_runtime_repairs_stale_broken_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / ".vox-parakeet-nemo-runtime-ready").touch()
    install = MagicMock(return_value=True)
    probes = [False, True]

    monkeypatch.setattr(module, "_runtime_target_dir", lambda: runtime_dir)
    monkeypatch.setattr(module, "_runtime_module_available", lambda name: probes.pop(0))
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)

    module._install_nemo_runtime()

    install.assert_called_once()
    assert (runtime_dir / ".vox-parakeet-nemo-runtime-ready").is_file()


def test_install_nemo_runtime_removes_stale_matplotlib_before_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / ".vox-parakeet-nemo-runtime-ready").touch()
    (runtime_dir / "matplotlib").mkdir()
    (runtime_dir / "matplotlib-3.8.2.dist-info").mkdir()
    (runtime_dir / "matplotlib-3.9.4.dist-info").mkdir()
    (runtime_dir / "matplotlib.libs").mkdir()
    install = MagicMock(return_value=True)
    probes = [False, True]

    monkeypatch.setattr(module, "_runtime_target_dir", lambda: runtime_dir)
    monkeypatch.setattr(module, "_runtime_module_available", lambda name: probes.pop(0))
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)

    module._install_nemo_runtime()

    assert not (runtime_dir / "matplotlib").exists()
    assert not (runtime_dir / "matplotlib-3.8.2.dist-info").exists()
    assert not (runtime_dir / "matplotlib-3.9.4.dist-info").exists()
    assert not (runtime_dir / "matplotlib.libs").exists()
    install.assert_called_once()


def test_clear_nemo_modules_removes_partial_import_resolver_state():
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    sys.modules["nemo"] = ModuleType("nemo")
    sys.modules["nemo.collections.asr"] = ModuleType("nemo.collections.asr")
    sys.modules["hydra"] = ModuleType("hydra")
    sys.modules["hydra._internal"] = ModuleType("hydra._internal")
    sys.modules["hydra_plugins"] = ModuleType("hydra_plugins")
    sys.modules["omegaconf"] = ModuleType("omegaconf")
    sys.modules["omegaconf.omegaconf"] = ModuleType("omegaconf.omegaconf")
    sys.modules["matplotlib"] = ModuleType("matplotlib")
    sys.modules["fiddle"] = ModuleType("fiddle")
    sys.modules["fiddle._src.daglish"] = ModuleType("fiddle._src.daglish")
    sys.modules["transformers"] = ModuleType("transformers")
    sys.modules["transformers.models.auto.tokenization_auto"] = ModuleType(
        "transformers.models.auto.tokenization_auto"
    )
    sys.modules["nv_one_logger"] = ModuleType("nv_one_logger")
    sys.modules["nv_one_logger.training_telemetry.api.training_telemetry_provider"] = ModuleType(
        "nv_one_logger.training_telemetry.api.training_telemetry_provider"
    )

    module._clear_nemo_modules()

    assert "nemo" not in sys.modules
    assert "nemo.collections.asr" not in sys.modules
    assert "hydra" not in sys.modules
    assert "hydra._internal" not in sys.modules
    assert "hydra_plugins" not in sys.modules
    assert "omegaconf" not in sys.modules
    assert "omegaconf.omegaconf" not in sys.modules
    assert "matplotlib" not in sys.modules
    assert "fiddle" not in sys.modules
    assert "fiddle._src.daglish" not in sys.modules
    assert "transformers" not in sys.modules
    assert "transformers.models.auto.tokenization_auto" not in sys.modules
    assert "nv_one_logger" not in sys.modules
    assert "nv_one_logger.training_telemetry.api.training_telemetry_provider" not in sys.modules


def test_clear_nemo_modules_removes_modules_loaded_from_sibling_runtimes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    runtime_root = tmp_path / "runtime"
    parakeet_runtime = runtime_root / "parakeet-nemo"
    kokoro_runtime = runtime_root / "kokoro"
    parakeet_runtime.mkdir(parents=True)
    kokoro_runtime.mkdir()
    (kokoro_runtime / "packaging.py").write_text("# fake", encoding="utf-8")
    (parakeet_runtime / "nemo.py").write_text("# fake", encoding="utf-8")

    sibling_module = ModuleType("packaging")
    sibling_module.__file__ = str(kokoro_runtime / "packaging.py")
    current_module = ModuleType("nemo")
    current_module.__file__ = str(parakeet_runtime / "nemo.py")
    sys.modules["packaging"] = sibling_module
    sys.modules["nemo"] = current_module

    monkeypatch.setattr(module, "vox_runtime_root", lambda: runtime_root)
    monkeypatch.setattr(module, "_runtime_target_dir", lambda: parakeet_runtime)

    module._clear_nemo_modules()

    assert "packaging" not in sys.modules
    assert "nemo" not in sys.modules


def test_runtime_module_available_rejects_partial_nemo_asr_module():
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    sys.modules["nemo.collections.asr"] = ModuleType("nemo.collections.asr")

    assert module._runtime_module_available("nemo.collections.asr") is False


def test_install_nemo_runtime_fails_fast_when_uv_install_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        mock = MagicMock()
        mock.stderr = ""
        mock.stdout = ""
        calls.append(cmd)
        if cmd[0] == "uv":
            mock.returncode = 1
            mock.stderr = "resolution failed"
            return mock
        raise AssertionError(f"unexpected installer fallback: {cmd}")

    monkeypatch.setattr(module, "_runtime_target_dir", lambda: tmp_path / "runtime")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Failed to install Parakeet NeMo runtime dependencies"):
        module._install_nemo_runtime()

    assert calls[0][:2] == ["uv", "pip"]
    assert len(calls) == 1
    assert not (tmp_path / "runtime" / ".vox-parakeet-nemo-runtime-ready").exists()


def test_install_nemo_runtime_fails_fast_when_uv_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    import vox_parakeet.nemo_adapter as module

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "uv":
            raise FileNotFoundError("uv")
        raise AssertionError(f"unexpected installer fallback: {cmd}")

    monkeypatch.setattr(module, "_runtime_target_dir", lambda: tmp_path / "runtime")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Failed to install Parakeet NeMo runtime dependencies"):
        module._install_nemo_runtime()

    assert calls[0][:2] == ["uv", "pip"]
    assert len(calls) == 1
    assert not (tmp_path / "runtime" / ".vox-parakeet-nemo-runtime-ready").exists()


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


def test_transcribe_serializes_concurrent_nemo_model_calls(monkeypatch: pytest.MonkeyPatch):
    fake_model = _ConcurrentDetectingNemoModel()
    _install_fake_nemo(model=fake_model)
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")
    audio = np.zeros(16000, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: adapter.transcribe(audio), range(2)))

    assert [result.text for result in results] == ["serialized", "serialized"]
    assert len(fake_model.transcribe_calls) == 2
    assert fake_model.max_active_calls == 1


def test_transcribe_retries_once_after_cuda_graph_failure(monkeypatch: pytest.MonkeyPatch):
    fake_model = _FakeNemoModel(text="after retry")
    fake_model.transcribe_errors.append(
        RuntimeError("Called CUDAGraph::replay without a preceding successful capture")
    )
    _install_fake_nemo(model=fake_model)
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    result = adapter.transcribe(np.zeros(16000, dtype=np.float32), word_timestamps=False)

    assert result.text == "after retry"
    assert len(fake_model.transcribe_calls) == 2
    assert fake_model.disable_cuda_graph_calls == 2


def test_transcribe_retries_when_cleanup_error_wraps_cuda_graph_failure(monkeypatch: pytest.MonkeyPatch):
    try:
        try:
            raise RuntimeError("CUDA graph capture failed before replay")
        except RuntimeError as exc:
            raise OSError("Directory not empty") from exc
    except OSError as wrapped_error:
        fake_model = _FakeNemoModel(text="wrapped retry")
        fake_model.transcribe_errors.append(wrapped_error)

    _install_fake_nemo(model=fake_model)
    _install_fake_torch(cuda_available=True)
    sys.modules.pop("vox_parakeet", None)
    sys.modules.pop("vox_parakeet.nemo_adapter", None)

    from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    result = adapter.transcribe(np.zeros(16000, dtype=np.float32), word_timestamps=False)

    assert result.text == "wrapped retry"
    assert len(fake_model.transcribe_calls) == 2
