from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import vox_parakeet.nemo_adapter as module
from vox_parakeet.nemo_adapter import ParakeetNemoAdapter

from vox.core.errors import ModelLoadError
from vox.core.worker_host import WorkerError


def _land_expected_runtime_paths(runtime_dir: Path) -> None:
    for relative in module._RUNTIME_EXPECTED_RELATIVE_PATHS:
        target = runtime_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def _land_worker_local_gpu_stack(runtime_dir: Path) -> None:
    for relative in (
        "torch/__init__.py",
        "torch-2.13.0.dist-info/METADATA",
        "torchgen/__init__.py",
        "functorch/__init__.py",
        "torchaudio/__init__.py",
        "torchaudio-2.13.0.dist-info/METADATA",
        "torio/__init__.py",
        "torchvision/__init__.py",
        "torchvision-0.28.0.dist-info/METADATA",
        "triton/__init__.py",
        "triton-3.7.0.dist-info/METADATA",
        "nvidia/cuda_runtime/__init__.py",
        "nvidia_cuda_runtime-13.0.dist-info/METADATA",
        "cuda/__init__.py",
        "cuda_bindings-13.3.1.dist-info/METADATA",
    ):
        target = runtime_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    for relative in (
        "torchmetrics/__init__.py",
        "torchmetrics-1.9.0.dist-info/METADATA",
        "nv_one_logger_pytorch_lightning_integration-2.3.1.dist-info/METADATA",
    ):
        target = runtime_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def _startup_error(worker_error_message: str) -> ModelLoadError:
    return ModelLoadError(
        "parakeet-nemo worker failed to start: parakeet-nemo worker killed: expected ready frame, "
        "got {'error': '" + worker_error_message + "'} (exit code 1)"
    )


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "options: --excludes <EXCLUDES>", True),
        (0, "options: --exclude-newer <DATE>", False),
        (2, "options: --excludes <EXCLUDES>", False),
    ],
)
def test_uv_supports_excludes_requires_successful_help_with_option(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, expected: bool
):
    result = MagicMock(returncode=returncode, stdout=stdout)
    runner = MagicMock(return_value=result)
    monkeypatch.setattr(module, "_run_install_command", runner)

    assert module._uv_supports_excludes() is expected
    runner.assert_called_once_with(["uv", "pip", "install", "--help"], 30)


@pytest.fixture
def runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runtime = tmp_path / "runtime" / "parakeet-nemo"
    monkeypatch.setattr(module, "_runtime_target_dir", lambda: runtime)
    monkeypatch.setattr(module, "_uv_supports_excludes", lambda: True)
    return runtime


@pytest.fixture
def ready_runtime(runtime_dir: Path) -> Path:
    runtime_dir.mkdir(parents=True)
    (runtime_dir / module._RUNTIME_SENTINEL).touch()
    return runtime_dir


@pytest.fixture
def fake_host_cls(monkeypatch: pytest.MonkeyPatch):
    class _FakeWorkerHost:
        instances: list[_FakeWorkerHost] = []

        def __init__(self, argv: list[str], *, env: dict[str, str], name: str, startup_timeout: float) -> None:
            self.argv = argv
            self.env = env
            self.name = name
            self.startup_timeout = startup_timeout
            self.requests: list[dict[str, Any]] = []
            self.responses: list[dict[str, Any]] = []
            self.request_error: Exception | None = None
            self.closed = False
            self._alive = True
            type(self).instances.append(self)

        @property
        def alive(self) -> bool:
            return self._alive and not self.closed

        def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            observation: dict[str, Any] = {"payload": payload, "timeout": timeout}
            path = payload.get("path")
            if path is not None:
                observation["wav_existed"] = os.path.exists(path)
                if observation["wav_existed"]:
                    data, sample_rate = sf.read(path, dtype="float32")
                    observation["wav_samples"] = len(data)
                    observation["wav_sample_rate"] = sample_rate
            self.requests.append(observation)
            if self.request_error is not None:
                raise self.request_error
            return self.responses.pop(0)

        def close(self, grace: float = 5.0) -> None:
            self.closed = True

    monkeypatch.setattr(module, "WorkerHost", _FakeWorkerHost)
    return _FakeWorkerHost


def _loaded_adapter(fake_host_cls) -> tuple[ParakeetNemoAdapter, Any]:
    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")
    return adapter, fake_host_cls.instances[-1]


def test_info_exposes_pytorch_and_nemo_adapter_name():
    info = ParakeetNemoAdapter().info()
    assert info.name == "parakeet-stt-nemo"
    assert info.type.value == "stt"
    assert info.supported_formats[0].value == "pytorch"
    assert info.supports_word_timestamps is True
    assert info.supports_language_detection is True


def test_prepare_runtime_installs_without_importing_nemo_or_touching_sys_path(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    install = MagicMock(return_value=True)
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)
    path_snapshot = list(sys.path)

    ParakeetNemoAdapter().prepare_runtime()

    assert sys.path == path_snapshot
    assert "nemo" not in sys.modules
    install.assert_called_once()
    assert install.call_args.args[0] == runtime_dir
    assert install.call_args.args[1] == module._RUNTIME_DEPENDENCIES
    assert tuple(install.call_args.kwargs["expected_paths"]) == (
        runtime_dir / "nemo" / "__init__.py",
        runtime_dir / "nemo" / "collections" / "asr" / "__init__.py",
    )
    assert install.call_args.kwargs["extra_install_args"][0] == "--excludes"
    assert (runtime_dir / module._RUNTIME_SENTINEL).is_file()


def test_prepare_runtime_removes_worker_local_gpu_stack_after_install(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    def install(*args, **kwargs):
        _land_expected_runtime_paths(runtime_dir)
        _land_worker_local_gpu_stack(runtime_dir)
        return True

    monkeypatch.setattr(module, "install_target_runtime_requirements", install)

    ParakeetNemoAdapter().prepare_runtime()

    for pattern in module._SHARED_APP_RUNTIME_GLOBS:
        assert list(runtime_dir.glob(pattern)) == []
    assert (runtime_dir / "torchmetrics" / "__init__.py").is_file()
    assert (runtime_dir / "torchmetrics-1.9.0.dist-info" / "METADATA").is_file()
    assert (runtime_dir / "nv_one_logger_pytorch_lightning_integration-2.3.1.dist-info" / "METADATA").is_file()
    assert (runtime_dir / module._RUNTIME_SENTINEL).is_file()


def test_prepare_runtime_falls_back_when_uv_does_not_support_excludes(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[str, ...]] = []

    def install(*args, **kwargs):
        calls.append(tuple(kwargs["extra_install_args"]))
        _land_expected_runtime_paths(runtime_dir)
        _land_worker_local_gpu_stack(runtime_dir)
        return True

    monkeypatch.setattr(module, "_uv_supports_excludes", lambda: False)
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)

    ParakeetNemoAdapter().prepare_runtime()

    assert calls == [()]
    for pattern in module._SHARED_APP_RUNTIME_GLOBS:
        assert list(runtime_dir.glob(pattern)) == []
    assert (runtime_dir / "torchmetrics" / "__init__.py").is_file()
    assert (runtime_dir / module._RUNTIME_SENTINEL).is_file()


def test_prepare_runtime_skips_reinstall_when_sentinel_present(ready_runtime: Path, monkeypatch: pytest.MonkeyPatch):
    install = MagicMock(return_value=True)
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)

    ParakeetNemoAdapter().prepare_runtime()

    install.assert_not_called()


def test_prepare_runtime_repairs_sentinel_runtime_with_worker_local_gpu_stack(
    ready_runtime: Path, monkeypatch: pytest.MonkeyPatch
):
    _land_worker_local_gpu_stack(ready_runtime)
    install = MagicMock(return_value=True)
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)

    ParakeetNemoAdapter().prepare_runtime()

    install.assert_not_called()
    for pattern in module._SHARED_APP_RUNTIME_GLOBS:
        assert list(ready_runtime.glob(pattern)) == []
    assert (ready_runtime / "torchmetrics" / "__init__.py").is_file()
    assert (ready_runtime / module._RUNTIME_SENTINEL).is_file()


def test_prepare_runtime_uses_uv_target_install_with_python312_compatible_pins(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[list[str]] = []

    excluded_packages: list[str] = []

    def fake_run(cmd: list[str], timeout: int):
        calls.append(cmd)
        excludes_index = cmd.index("--excludes")
        excluded_packages.extend(Path(cmd[excludes_index + 1]).read_text(encoding="utf-8").splitlines())
        _land_expected_runtime_paths(runtime_dir)
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = ""
        mock.stderr = ""
        return mock

    monkeypatch.setattr(module, "_run_install_command", fake_run)

    ParakeetNemoAdapter().prepare_runtime()

    assert calls[0][:5] == ["uv", "pip", "install", "--python", sys.executable]
    assert "--target" in calls[0]
    assert "nemo-toolkit[asr]==2.7.3" in calls[0]
    assert "numpy>=1.26,<2" in calls[0]
    assert "numba>=0.61,<0.67" in calls[0]
    assert "llvmlite>=0.44,<0.49" in calls[0]
    assert "matplotlib>=3.9,<3.10" in calls[0]
    assert excluded_packages == list(module._SHARED_APP_RUNTIME_EXCLUDES)
    assert len(calls) == 1
    assert (runtime_dir / module._RUNTIME_SENTINEL).is_file()


def test_prepare_runtime_fails_fast_when_uv_install_fails(runtime_dir: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_run(cmd: list[str], timeout: int):
        mock = MagicMock()
        mock.returncode = 1
        mock.stdout = ""
        mock.stderr = "resolution failed"
        return mock

    monkeypatch.setattr(module, "_run_install_command", fake_run)

    with pytest.raises(RuntimeError, match="Failed to install Parakeet NeMo runtime dependencies"):
        ParakeetNemoAdapter().prepare_runtime()

    assert not (runtime_dir / module._RUNTIME_SENTINEL).exists()


def test_prepare_runtime_fails_fast_when_uv_is_missing(runtime_dir: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_run(cmd: list[str], timeout: int):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(module, "_run_install_command", fake_run)

    with pytest.raises(RuntimeError, match="Failed to install Parakeet NeMo runtime dependencies"):
        ParakeetNemoAdapter().prepare_runtime()

    assert not (runtime_dir / module._RUNTIME_SENTINEL).exists()


def test_prepare_runtime_does_not_set_sentinel_when_install_succeeds_without_landing_nemo(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    def fake_run(cmd: list[str], timeout: int):
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = ""
        mock.stderr = ""
        return mock

    monkeypatch.setattr(module, "_run_install_command", fake_run)

    with pytest.raises(RuntimeError, match="Failed to install Parakeet NeMo runtime dependencies"):
        ParakeetNemoAdapter().prepare_runtime()

    assert not (runtime_dir / module._RUNTIME_SENTINEL).exists()


def test_load_runtime_import_failure_clears_sentinel_and_next_load_reinstalls(
    ready_runtime: Path, fake_host_cls, monkeypatch: pytest.MonkeyPatch
):
    install = MagicMock(return_value=True)
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)
    error = _startup_error(
        "RuntimeError: "
        + module.RUNTIME_IMPORT_ERROR_MARKER
        + " Parakeet NeMo worker could not import nemo.collections.asr: broken llvmlite ABI"
    )

    class _BrokenRuntimeHost:
        def __init__(self, argv: list[str], *, env: dict[str, str], name: str, startup_timeout: float) -> None:
            raise error

    monkeypatch.setattr(module, "WorkerHost", _BrokenRuntimeHost)
    adapter = ParakeetNemoAdapter()

    with pytest.raises(ModelLoadError, match="failed to start"):
        adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    assert not (ready_runtime / module._RUNTIME_SENTINEL).exists()
    assert adapter.is_loaded is False
    install.assert_not_called()

    monkeypatch.setattr(module, "WorkerHost", fake_host_cls)
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    install.assert_called_once()
    assert install.call_args.args[0] == ready_runtime
    assert (ready_runtime / module._RUNTIME_SENTINEL).is_file()
    assert adapter.is_loaded is True


def test_load_non_import_startup_failure_keeps_sentinel(ready_runtime: Path, monkeypatch: pytest.MonkeyPatch):
    install = MagicMock(return_value=True)
    monkeypatch.setattr(module, "install_target_runtime_requirements", install)
    error = _startup_error("RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu")

    class _CudaLessHost:
        def __init__(self, argv: list[str], *, env: dict[str, str], name: str, startup_timeout: float) -> None:
            raise error

    monkeypatch.setattr(module, "WorkerHost", _CudaLessHost)
    adapter = ParakeetNemoAdapter()

    with pytest.raises(ModelLoadError, match="failed to start"):
        adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    assert (ready_runtime / module._RUNTIME_SENTINEL).is_file()
    assert adapter.is_loaded is False
    install.assert_not_called()


def test_load_spawns_worker_with_runtime_on_child_pythonpath(
    ready_runtime: Path, fake_host_cls, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VOX_SECRET_EXAMPLE", "leak")
    monkeypatch.setenv("TMPDIR", "/tmp/vox")

    adapter, host = _loaded_adapter(fake_host_cls)

    assert host.argv == [
        sys.executable,
        "-m",
        "vox_parakeet.nemo_worker",
        "--model-id",
        "nvidia/parakeet-tdt-0.6b-v3",
    ]
    import vox_parakeet

    import vox

    python_paths = host.env["PYTHONPATH"].split(os.pathsep)
    assert python_paths[0] == str(ready_runtime)
    assert str(Path(vox.__file__).resolve().parents[1]) in python_paths
    assert str(Path(vox_parakeet.__file__).resolve().parents[1]) in python_paths
    assert host.env[module.WORKER_DEVICE_ENV] == "cuda"
    assert host.env["TMPDIR"] == "/tmp/vox"
    assert "VOX_SECRET_EXAMPLE" not in host.env
    assert host.name == "parakeet-nemo"
    assert host.startup_timeout == module.DEFAULT_STARTUP_TIMEOUT_SECONDS
    assert host.startup_timeout >= 600.0
    assert adapter.is_loaded is True
    assert adapter._model_id == "nvidia/parakeet-tdt-0.6b-v3"


def test_load_startup_timeout_is_configurable_via_env(
    ready_runtime: Path, fake_host_cls, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(module.STARTUP_TIMEOUT_ENV, "42.5")

    _adapter, host = _loaded_adapter(fake_host_cls)

    assert host.startup_timeout == 42.5


def test_load_uses_checkpoint_argv_for_local_nemo_checkpoint(ready_runtime: Path, fake_host_cls, tmp_path: Path):
    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    checkpoint = model_dir / "parakeet-tdt-0.6b-v3.nemo"
    checkpoint.write_bytes(b"fake-nemo")

    adapter = ParakeetNemoAdapter()
    adapter.load(str(model_dir), "cuda")

    host = fake_host_cls.instances[-1]
    assert host.argv == [
        sys.executable,
        "-m",
        "vox_parakeet.nemo_worker",
        "--model-id",
        str(model_dir),
        "--checkpoint",
        str(checkpoint),
    ]
    assert adapter.is_loaded is True
    assert adapter._model_id == str(model_dir)


def test_load_prefers_pretrained_source_over_pulled_local_nemo_checkpoint(
    ready_runtime: Path, fake_host_cls, tmp_path: Path
):
    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    (model_dir / "parakeet-tdt-0.6b-v3.nemo").write_bytes(b"fake-nemo")

    adapter = ParakeetNemoAdapter()
    adapter.load(str(model_dir), "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    host = fake_host_cls.instances[-1]
    assert "--checkpoint" not in host.argv
    assert host.argv[-1] == "nvidia/parakeet-tdt-0.6b-v3"
    assert adapter._model_id == "nvidia/parakeet-tdt-0.6b-v3"


def test_load_rejects_non_cuda_device(ready_runtime: Path, fake_host_cls):
    adapter = ParakeetNemoAdapter()

    with pytest.raises(RuntimeError, match="requires device='cuda' or 'auto'"):
        adapter.load("ignored-local-path", "cpu")

    assert fake_host_cls.instances == []
    assert adapter.is_loaded is False


def test_load_is_idempotent_while_worker_alive(ready_runtime: Path, fake_host_cls):
    adapter, _host = _loaded_adapter(fake_host_cls)
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    assert len(fake_host_cls.instances) == 1


def test_load_respawns_after_worker_death(ready_runtime: Path, fake_host_cls):
    adapter, first = _loaded_adapter(fake_host_cls)
    first._alive = False
    assert adapter.is_loaded is False

    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")

    assert first.closed is True
    assert len(fake_host_cls.instances) == 2
    assert adapter.is_loaded is True


def test_transcribe_round_trips_wav_and_rebuilds_word_timestamp_result(ready_runtime: Path, fake_host_cls):
    adapter, host = _loaded_adapter(fake_host_cls)
    host.responses.append(
        {
            "text": "hello world",
            "language": "en",
            "words": [
                {"word": "hello", "start_ms": 0, "end_ms": 640},
                {"word": "world", "start_ms": 800, "end_ms": 1600},
            ],
        }
    )

    result = adapter.transcribe(np.zeros(16000, dtype=np.float32), word_timestamps=True)

    observation = host.requests[0]
    assert observation["payload"]["op"] == "transcribe"
    assert observation["payload"]["word_timestamps"] is True
    assert observation["timeout"] == module.DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert observation["wav_existed"] is True
    assert observation["wav_samples"] == 16000
    assert observation["wav_sample_rate"] == 16000
    assert not os.path.exists(observation["payload"]["path"])
    assert result.text == "hello world"
    assert result.duration_ms == 1000
    assert result.language == "en"
    assert result.model == "nvidia/parakeet-tdt-0.6b-v3"
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment.text == "hello world"
    assert segment.start_ms == 0
    assert segment.end_ms == 1000
    assert segment.language == "en"
    assert [(word.word, word.start_ms, word.end_ms) for word in segment.words] == [
        ("hello", 0, 640),
        ("world", 800, 1600),
    ]


def test_transcribe_without_word_timestamps_returns_text(ready_runtime: Path, fake_host_cls):
    adapter, host = _loaded_adapter(fake_host_cls)
    host.responses.append({"text": "plain text", "language": None, "words": []})

    result = adapter.transcribe(np.zeros(16000, dtype=np.float32), word_timestamps=False)

    assert host.requests[0]["payload"]["word_timestamps"] is False
    assert result.text == "plain text"
    assert result.segments[0].text == "plain text"
    assert result.segments[0].words == ()
    assert result.language is None


def test_transcribe_empty_audio_short_circuits_without_worker_request(ready_runtime: Path, fake_host_cls):
    adapter, host = _loaded_adapter(fake_host_cls)

    result = adapter.transcribe(np.zeros(0, dtype=np.float32), language="en")

    assert host.requests == []
    assert result.text == ""
    assert result.segments == ()
    assert result.language == "en"
    assert result.duration_ms == 0
    assert result.model == "nvidia/parakeet-tdt-0.6b-v3"


def test_transcribe_deletes_temp_wav_when_worker_request_fails(ready_runtime: Path, fake_host_cls):
    adapter, host = _loaded_adapter(fake_host_cls)
    host.request_error = WorkerError("parakeet-nemo worker killed: timed out after 600.0s (exit code -9)")

    with pytest.raises(WorkerError, match="timed out"):
        adapter.transcribe(np.zeros(16000, dtype=np.float32))

    observation = host.requests[0]
    assert observation["wav_existed"] is True
    assert not os.path.exists(observation["payload"]["path"])


def test_dead_worker_reads_as_unloaded_and_transcribe_raises(ready_runtime: Path, fake_host_cls):
    adapter, host = _loaded_adapter(fake_host_cls)
    host._alive = False

    assert adapter.is_loaded is False
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.transcribe(np.zeros(16000, dtype=np.float32))
    assert host.requests == []


def test_trim_does_not_touch_live_nemo_worker(ready_runtime: Path, fake_host_cls):
    adapter, host = _loaded_adapter(fake_host_cls)

    adapter.trim()

    assert adapter.is_loaded is True
    assert host.requests == []


def test_trim_is_noop_when_not_loaded(fake_host_cls):
    adapter = ParakeetNemoAdapter()

    adapter.trim()

    assert fake_host_cls.instances == []


def test_trim_is_noop_when_worker_dead(ready_runtime: Path, fake_host_cls):
    adapter, host = _loaded_adapter(fake_host_cls)
    host._alive = False

    adapter.trim()

    assert host.requests == []


def test_unload_closes_host_and_resets_state(ready_runtime: Path, fake_host_cls):
    adapter, host = _loaded_adapter(fake_host_cls)

    adapter.unload()

    assert host.closed is True
    assert adapter.is_loaded is False
    assert adapter._model_id == module.DEFAULT_MODEL_ID


INTEGRATION_WORKER_SOURCE = """
import wave

from vox.core.worker_host import worker_main


def handler(request):
    if request["op"] != "transcribe":
        raise RuntimeError("unexpected op: " + str(request.get("op")))
    with wave.open(request["path"], "rb") as wav:
        frames = wav.getnframes()
    return {"text": f"frames:{frames}", "language": "en", "words": []}


raise SystemExit(worker_main(handler))
"""


def test_adapter_host_wiring_end_to_end_with_fake_worker(ready_runtime: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        module,
        "_worker_argv",
        lambda model_id, checkpoint_path: [sys.executable, "-c", INTEGRATION_WORKER_SOURCE],
    )
    monkeypatch.setenv(module.STARTUP_TIMEOUT_ENV, "30")

    adapter = ParakeetNemoAdapter()
    adapter.load("ignored-local-path", "cuda", _source="nvidia/parakeet-tdt-0.6b-v3")
    try:
        assert adapter.is_loaded is True

        result = adapter.transcribe(np.zeros(16000, dtype=np.float32), word_timestamps=False)

        assert result.text == "frames:16000"
        assert result.duration_ms == 1000
        assert result.language == "en"
        assert result.segments[0].text == "frames:16000"
        assert result.model == "nvidia/parakeet-tdt-0.6b-v3"
    finally:
        adapter.unload()
    assert adapter.is_loaded is False
