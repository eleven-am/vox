from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import vox_parakeet.nemo_worker as worker

from vox.core.worker_host import WORKER_FD_ENV

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

    response = handle({"op": "transcribe", "path": str(wav_path), "word_timestamps": True})

    assert response == {
        "text": "hello world",
        "language": None,
        "words": [
            {"word": "hello", "start_ms": 0, "end_ms": 640},
            {"word": "world", "start_ms": 800, "end_ms": 1600},
        ],
    }
    assert fake_model.transcribe_calls == [
        {"paths": [str(wav_path)], "batch_size": 1, "timestamps": True, "return_hypotheses": True}
    ]


def test_handler_transcribe_without_word_timestamps_returns_text_only(tmp_path: Path):
    fake_model = _FakeNemoModel(text="plain text")
    wav_path = _write_wav(tmp_path)
    handle = worker.build_handler(fake_model)

    response = handle({"op": "transcribe", "path": str(wav_path), "word_timestamps": False})

    assert response == {"text": "plain text", "language": None, "words": []}
    assert fake_model.transcribe_calls == [{"paths": [str(wav_path)], "batch_size": 1}]


def test_handler_rejects_unknown_op():
    handle = worker.build_handler(_FakeNemoModel())

    with pytest.raises(RuntimeError, match="unknown Parakeet NeMo worker op"):
        handle({"op": "synthesize"})


def test_transcribe_retries_once_after_cuda_graph_failure(tmp_path: Path):
    fake_model = _FakeNemoModel(text="after retry")
    fake_model.transcribe_errors.append(
        RuntimeError("Called CUDAGraph::replay without a preceding successful capture")
    )
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


def test_main_loads_model_before_ready_and_serves_transcribe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
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
            assert response == {"text": "ready path", "language": None, "words": []}
    finally:
        parent_sock.close()
        child_sock.close()
        thread.join(timeout=5.0)
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
    assert not thread.is_alive()
    assert exit_codes == [0]
