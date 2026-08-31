from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import vox_parakeet.nemo_worker as worker

from vox.core.worker_host import WORKER_FD_ENV, WORKER_PARENT_PID_ENV

_NEMO_MODULE_KEYS = ("nemo", "nemo.collections", "nemo.collections.asr")


@pytest.fixture(autouse=True)
def _restore_nemo_modules():
    saved = {key: sys.modules.get(key) for key in _NEMO_MODULE_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


class _FakeNemoModel:
    def __init__(self, *, text: str = "hello world") -> None:
        self.text = text
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


def _drop_nemo_modules() -> None:
    for key in _NEMO_MODULE_KEYS:
        sys.modules.pop(key, None)


def _write_wav(tmp_path: Path, samples: int = 16000) -> Path:
    wav_path = tmp_path / "audio.wav"
    sf.write(wav_path, np.zeros(samples, dtype=np.float32), 16000)
    return wav_path


def test_load_model_uses_pretrained_and_disables_cuda_graphs():
    fake_model = _FakeNemoModel()
    fake_model_cls = _install_fake_nemo(model=fake_model)

    model = worker.load_model("nvidia/parakeet-tdt-0.6b-v3", None, "cuda")

    fake_model_cls.from_pretrained.assert_called_once_with(model_name="nvidia/parakeet-tdt-0.6b-v3")
    fake_model_cls.restore_from.assert_not_called()
    assert model is fake_model
    assert fake_model.to_calls == ["cuda"]
    assert fake_model.eval_called is True
    assert fake_model.disable_cuda_graph_calls == 1
    assert fake_model.decoding.decoding.decoding_computer.use_cuda_graph_decoder is False


def test_load_model_uses_restore_from_for_checkpoint(tmp_path: Path):
    fake_model_cls = _install_fake_nemo()
    checkpoint = tmp_path / "parakeet-tdt-0.6b-v3.nemo"
    checkpoint.write_bytes(b"fake-nemo")

    worker.load_model(str(tmp_path), str(checkpoint), "cuda")

    fake_model_cls.restore_from.assert_called_once_with(restore_path=str(checkpoint))
    fake_model_cls.from_pretrained.assert_not_called()


def test_load_model_raises_clear_error_when_nemo_is_missing():
    _drop_nemo_modules()

    with pytest.raises(RuntimeError, match="could not import nemo.collections.asr"):
        worker.load_model("nvidia/parakeet-tdt-0.6b-v3", None, "cuda")


def test_load_model_rejects_partial_nemo_asr_module():
    _drop_nemo_modules()
    sys.modules["nemo.collections.asr"] = ModuleType("nemo.collections.asr")

    with pytest.raises(RuntimeError, match="requires nemo.collections.asr.models.ASRModel"):
        worker.load_model("nvidia/parakeet-tdt-0.6b-v3", None, "cuda")


def test_handler_transcribe_with_word_timestamps_returns_response_shape(tmp_path: Path):
    fake_model = _FakeNemoModel(text="hello world")
    wav_path = _write_wav(tmp_path)
    handle = worker.build_handler(fake_model)

    with patch.object(worker, "runtime_memory_status", return_value={"rss_bytes": 123}):
        response = handle({"op": "transcribe", "path": str(wav_path), "word_timestamps": True})

    assert response == {
        "text": "hello world",
        "language": None,
        "words": [
            {"word": "hello", "start_ms": 0, "end_ms": 640},
            {"word": "world", "start_ms": 800, "end_ms": 1600},
        ],
        "memory": {"rss_bytes": 123},
    }
    assert fake_model.transcribe_calls == [
        {"paths": [str(wav_path)], "batch_size": 1, "timestamps": True, "return_hypotheses": True}
    ]


def test_handler_transcribe_without_word_timestamps_returns_text_only(tmp_path: Path):
    fake_model = _FakeNemoModel(text="plain text")
    wav_path = _write_wav(tmp_path)
    handle = worker.build_handler(fake_model)

    with patch.object(worker, "runtime_memory_status", return_value={"rss_bytes": 123}):
        response = handle({"op": "transcribe", "path": str(wav_path), "word_timestamps": False})

    assert response == {"text": "plain text", "language": None, "words": [], "memory": {"rss_bytes": 123}}
    assert fake_model.transcribe_calls == [{"paths": [str(wav_path)], "batch_size": 1}]


def test_output_health_detects_unknown_token_collapse():
    result = worker._output_health("\u2047" * 1678, 4440)

    assert result is not None
    assert result["marker"] == worker.DEGRADED_OUTPUT_MARKER
    assert result["character_count"] == 1678
    assert result["characters_per_second"] == pytest.approx(377.928)
    assert result["unknown_ratio"] == 1.0
    assert result["dominant_codepoint"] == "U+2047"


@pytest.mark.parametrize(
    ("text", "duration_ms"),
    [
        ("\u2047" * 7, 1000),
        ("This is ordinary transcription with varied characters and punctuation.", 1000),
        ("ha " * 40, 1000),
    ],
)
def test_output_health_does_not_reject_short_or_lexical_transcripts(text: str, duration_ms: int):
    assert worker._output_health(text, duration_ms) is None


def test_output_health_detects_high_rate_repetitive_punctuation_collapse():
    result = worker._output_health("?" * 100, 1000)

    assert result is not None
    assert result["dominant_codepoint"] == "U+003F"
    assert result["dominant_ratio"] == 1.0


def test_handler_suppresses_degraded_output_and_attaches_forensics(tmp_path: Path):
    fake_model = _FakeNemoModel(text="\u2047" * 400)
    wav_path = _write_wav(tmp_path, samples=32_000)
    handle = worker.build_handler(fake_model)
    tensor_health = {"tensor_count": 100, "nonfinite_tensor_count": 1}
    module_health = {"encoder": {"nonfinite_value_count": 4}, "hooked_modules": 3}

    with (
        patch.object(worker, "runtime_memory_status", return_value={"rss_bytes": 123}),
        patch.object(worker, "_model_tensor_health", return_value=tensor_health) as health,
        patch.object(worker, "_diagnostic_module_health", return_value=module_health) as module_probe,
    ):
        response = handle(
            {
                "op": "transcribe",
                "path": str(wav_path),
                "word_timestamps": False,
                "duration_ms": 2000,
            }
        )

    assert response["text"] == ""
    assert response["words"] == []
    assert response["degraded"]["marker"] == worker.DEGRADED_OUTPUT_MARKER
    assert response["degraded"]["dominant_codepoint"] == "U+2047"
    assert response["degraded"]["tensor_health"] == tensor_health
    assert response["degraded"]["module_health"] == module_health
    health.assert_called_once_with(fake_model)
    module_probe.assert_called_once_with(fake_model, str(wav_path), word_timestamps=False)


def test_model_tensor_health_reports_nonfinite_parameters_and_buffers():
    torch = pytest.importorskip("torch")

    class _TensorModel:
        def named_parameters(self):
            return (("encoder.weight", torch.tensor([1.0, float("nan")])),)

        def named_buffers(self):
            return (("decoder.state", torch.tensor([float("inf"), 2.0, 3.0])),)

    result = worker._model_tensor_health(_TensorModel())

    assert result["tensor_count"] == 2
    assert result["value_count"] == 5
    assert result["nonfinite_tensor_count"] == 2
    assert result["nonfinite_value_count"] == 2
    assert result["first_nonfinite"] == ["parameter:encoder.weight", "buffer:decoder.state"]


def test_diagnostic_module_health_captures_first_forward_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class _FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        @property
        def shape(self):
            return self.values.shape

        def detach(self):
            return self

        def numel(self):
            return self.values.size

        def amin(self):
            return self.values.min()

        def amax(self):
            return self.values.max()

    class _FakeTorch:
        @staticmethod
        def is_tensor(value):
            return isinstance(value, _FakeTensor)

        @staticmethod
        def isfinite(value):
            return np.isfinite(value.values)

    class _Handle:
        def __init__(self, hooks, hook):
            self.hooks = hooks
            self.hook = hook

        def remove(self):
            self.hooks.remove(self.hook)

    class _Module:
        def __init__(self):
            self.hooks = []

        def register_forward_hook(self, hook):
            self.hooks.append(hook)
            return _Handle(self.hooks, hook)

        def emit(self, value):
            for hook in tuple(self.hooks):
                hook(self, (), value)

    class _ProbeModel(_FakeNemoModel):
        def __init__(self):
            super().__init__(text="probe")
            self.encoder = _Module()

        def transcribe(self, paths, **kwargs):
            self.encoder.emit(_FakeTensor([1.0, 1.0, float("nan")]))
            self.encoder.emit(_FakeTensor([2.0, 3.0]))
            return super().transcribe(paths, **kwargs)

    original_import = worker.importlib.import_module
    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: _FakeTorch() if name == "torch" else original_import(name),
    )
    model = _ProbeModel()
    wav_path = _write_wav(tmp_path)

    result = worker._diagnostic_module_health(model, str(wav_path), word_timestamps=False)

    assert result["hooked_modules"] == 1
    assert result["encoder"]["tensor_count"] == 1
    assert result["encoder"]["value_count"] == 3
    assert result["encoder"]["nonfinite_value_count"] == 1
    assert result["encoder"]["shapes"] == [[3]]
    assert model.encoder.hooks == []


def test_handler_trim_returns_before_and_after_memory():
    handle = worker.build_handler(_FakeNemoModel())
    result = {
        "before": {"rss_bytes": 2_000},
        "after": {"rss_bytes": 1_000},
        "gc_collected": 3,
        "malloc_trimmed": True,
    }

    with patch.object(worker, "trim_process_memory", return_value=result) as trim:
        response = handle({"op": "trim"})

    assert response == {"memory_trim": result}
    trim.assert_called_once_with(device="cuda")


@pytest.mark.parametrize("op", ["synthesize", "status"])
def test_handler_rejects_unknown_op(op: str):
    handle = worker.build_handler(_FakeNemoModel())

    with pytest.raises(RuntimeError, match="unknown Parakeet NeMo worker op"):
        handle({"op": op})


def test_transcribe_retries_once_after_cuda_graph_failure(tmp_path: Path):
    fake_model = _FakeNemoModel(text="after retry")
    fake_model.transcribe_errors.append(RuntimeError("Called CUDAGraph::replay without a preceding successful capture"))
    wav_path = _write_wav(tmp_path)

    response = worker.transcribe(fake_model, str(wav_path), word_timestamps=False)

    assert response["text"] == "after retry"
    assert len(fake_model.transcribe_calls) == 2
    assert fake_model.disable_cuda_graph_calls == 1


def test_transcribe_retries_when_cleanup_error_wraps_cuda_graph_failure(tmp_path: Path):
    try:
        try:
            raise RuntimeError("CUDA graph capture failed before replay")
        except RuntimeError as exc:
            raise OSError("Directory not empty") from exc
    except OSError as wrapped_error:
        fake_model = _FakeNemoModel(text="wrapped retry")
        fake_model.transcribe_errors.append(wrapped_error)

    wav_path = _write_wav(tmp_path)

    response = worker.transcribe(fake_model, str(wav_path), word_timestamps=False)

    assert response["text"] == "wrapped retry"
    assert len(fake_model.transcribe_calls) == 2


def test_transcribe_does_not_retry_non_cuda_graph_errors(tmp_path: Path):
    fake_model = _FakeNemoModel()
    fake_model.transcribe_errors.append(ValueError("audio decode failed"))
    wav_path = _write_wav(tmp_path)

    with pytest.raises(ValueError, match="audio decode failed"):
        worker.transcribe(fake_model, str(wav_path), word_timestamps=False)

    assert len(fake_model.transcribe_calls) == 1


def test_main_emits_error_frame_when_nemo_import_fails(monkeypatch: pytest.MonkeyPatch):
    _drop_nemo_modules()
    parent_sock, child_sock = socket.socketpair()
    monkeypatch.setenv(WORKER_FD_ENV, str(child_sock.fileno()))

    exit_code = worker.main(["--model-id", "nvidia/parakeet-tdt-0.6b-v3"])

    child_sock.close()
    with parent_sock, parent_sock.makefile("rb") as stream:
        frame = json.loads(stream.readline())
    assert exit_code == 1
    assert frame["error"].startswith(f"RuntimeError: {worker.RUNTIME_IMPORT_ERROR_MARKER}")
    assert "nemo.collections.asr" in frame["error"]


def test_main_arms_parent_death_signal_before_loading(monkeypatch: pytest.MonkeyPatch):
    order: list[str] = []
    monkeypatch.setattr(worker, "install_parent_death_signal", lambda: order.append("armed"))

    def recording_load(*_args, **_kwargs):
        order.append("loaded")
        return _FakeNemoModel()

    monkeypatch.setattr(worker, "load_model", recording_load)
    monkeypatch.setattr(worker, "worker_main", lambda _handler: 0)

    exit_code = worker.main(["--model-id", "nvidia/parakeet-tdt-0.6b-v3"])

    assert exit_code == 0
    assert order == ["armed", "loaded"]


def test_main_exits_when_parent_pid_no_longer_matches(monkeypatch: pytest.MonkeyPatch):
    order: list[str] = []
    monkeypatch.setattr(worker, "install_parent_death_signal", lambda: order.append("armed"))
    monkeypatch.setenv(WORKER_PARENT_PID_ENV, "999999")
    monkeypatch.setattr(worker.os, "getppid", lambda: 4321)

    def fail_load(*_args, **_kwargs):
        raise AssertionError("model must not load after orphan detection")

    monkeypatch.setattr(worker, "load_model", fail_load)

    exit_code = worker.main(["--model-id", "nvidia/parakeet-tdt-0.6b-v3"])

    assert exit_code == 1
    assert order == ["armed"]


def test_main_proceeds_when_parent_is_pid_one_in_container(monkeypatch: pytest.MonkeyPatch):
    order: list[str] = []
    monkeypatch.setattr(worker, "install_parent_death_signal", lambda: order.append("armed"))
    monkeypatch.setenv(WORKER_PARENT_PID_ENV, "1")
    monkeypatch.setattr(worker.os, "getppid", lambda: 1)

    def recording_load(*_args, **_kwargs):
        order.append("loaded")
        return _FakeNemoModel()

    monkeypatch.setattr(worker, "load_model", recording_load)
    monkeypatch.setattr(worker, "worker_main", lambda _handler: 0)

    exit_code = worker.main(["--model-id", "nvidia/parakeet-tdt-0.6b-v3"])

    assert exit_code == 0
    assert order == ["armed", "loaded"]


def test_main_proceeds_when_parent_pid_env_absent(monkeypatch: pytest.MonkeyPatch):
    order: list[str] = []
    monkeypatch.setattr(worker, "install_parent_death_signal", lambda: order.append("armed"))
    monkeypatch.delenv(WORKER_PARENT_PID_ENV, raising=False)
    monkeypatch.setattr(worker.os, "getppid", lambda: 1)

    def recording_load(*_args, **_kwargs):
        order.append("loaded")
        return _FakeNemoModel()

    monkeypatch.setattr(worker, "load_model", recording_load)
    monkeypatch.setattr(worker, "worker_main", lambda _handler: 0)

    exit_code = worker.main(["--model-id", "nvidia/parakeet-tdt-0.6b-v3"])

    assert exit_code == 0
    assert order == ["armed", "loaded"]


def test_main_loads_model_before_ready_and_serves_transcribe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_model = _FakeNemoModel(text="ready path")
    fake_model_cls = _install_fake_nemo(model=fake_model)
    wav_path = _write_wav(tmp_path)
    parent_sock, child_sock = socket.socketpair()
    monkeypatch.setenv(WORKER_FD_ENV, str(child_sock.fileno()))
    saved_stdout_fd = os.dup(1)
    exit_codes: list[int] = []
    thread = threading.Thread(
        target=lambda: exit_codes.append(worker.main(["--model-id", "nvidia/parakeet-tdt-0.6b-v3"]))
    )
    try:
        thread.start()
        with parent_sock.makefile("rwb") as stream:
            ready = json.loads(stream.readline())
            assert ready == {"ready": True}
            fake_model_cls.from_pretrained.assert_called_once_with(model_name="nvidia/parakeet-tdt-0.6b-v3")
            assert fake_model.to_calls == ["cuda"]
            assert fake_model.eval_called is True

            stream.write(
                json.dumps({"op": "transcribe", "path": str(wav_path), "word_timestamps": False}).encode() + b"\n"
            )
            stream.flush()
            response = json.loads(stream.readline())
            assert response["text"] == "ready path"
            assert response["language"] is None
            assert response["words"] == []
            assert response["memory"]["pid"] == os.getpid()
    finally:
        parent_sock.close()
        child_sock.close()
        thread.join(timeout=5.0)
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
    assert not thread.is_alive()
    assert exit_codes == [0]
