from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import vox_voxtral.voxtral_tts_worker as worker

from vox.core.worker_host import WORKER_FD_ENV


class _FakeOmniRuntime:
    def __init__(self, stage_outputs: list[Any]) -> None:
        self._stage_outputs = stage_outputs
        self.generate_calls: list[dict[str, Any]] = []
        self.shutdown_calls = 0

    def generate(self, inputs: dict[str, Any], *, request_id: str, sampling_params_list: list[Any]):
        self.generate_calls.append(
            {"inputs": inputs, "request_id": request_id, "sampling_params_list": sampling_params_list}
        )
        outputs = list(self._stage_outputs)

        async def _iterate():
            for output in outputs:
                yield output

        return _iterate()

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _SpeechRequest:
    def __init__(self, *, input: str, voice: str) -> None:
        self.input = input
        self.voice = voice


def _make_tokenizer() -> Any:
    tokenizer = MagicMock()
    tokenizer.instruct_tokenizer.encode_speech_request.return_value = SimpleNamespace(tokens=[1, 2, 3])
    return tokenizer


def _make_handler(
    runtime: _FakeOmniRuntime,
    tokenizer: Any,
    runner: asyncio.Runner,
    *,
    default_voice: str = "neutral_female",
):
    return worker.build_handler(runtime, _SpeechRequest, tokenizer, ["sp0", "sp1"], default_voice, runner)


def _stage_output(audio: Any, *, finished: bool) -> SimpleNamespace:
    return SimpleNamespace(multimodal_output={"audio": audio}, finished=finished)


def test_handler_synthesize_concatenates_chunks_and_dedups_finished_tail():
    runtime = _FakeOmniRuntime(
        [
            _stage_output(np.array([0.1, 0.2], dtype=np.float32), finished=False),
            SimpleNamespace(multimodal_output=None, finished=False),
            _stage_output(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), finished=True),
        ]
    )
    tokenizer = _make_tokenizer()
    with asyncio.Runner() as runner:
        handle = _make_handler(runtime, tokenizer, runner)

        response = handle({"op": "synthesize", "text": "hello world", "voice": "casual_male"})

    expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
    assert response["sample_rate"] == 24000
    assert base64.b64decode(response["audio_b64"]) == expected
    speech_request = tokenizer.instruct_tokenizer.encode_speech_request.call_args.args[0]
    assert speech_request.input == "hello world"
    assert speech_request.voice == "casual_male"
    call = runtime.generate_calls[0]
    assert call["inputs"]["prompt_token_ids"] == [1, 2, 3]
    assert call["inputs"]["additional_information"] == {"voice": ["casual_male"]}
    assert call["sampling_params_list"] == ["sp0", "sp1"]


def test_handler_uses_default_voice_when_request_voice_is_empty():
    runtime = _FakeOmniRuntime([])
    tokenizer = _make_tokenizer()
    with asyncio.Runner() as runner:
        handle = _make_handler(runtime, tokenizer, runner, default_voice="cheerful_female")

        response = handle({"op": "synthesize", "text": "hello", "voice": ""})

    speech_request = tokenizer.instruct_tokenizer.encode_speech_request.call_args.args[0]
    assert speech_request.voice == "cheerful_female"
    assert base64.b64decode(response["audio_b64"]) == b""


def test_handler_rejects_unsupported_op():
    runtime = _FakeOmniRuntime([])
    with asyncio.Runner() as runner:
        handle = _make_handler(runtime, _make_tokenizer(), runner)

        with pytest.raises(RuntimeError, match="Unsupported op: transcribe"):
            handle({"op": "transcribe", "path": "/tmp/audio.wav"})

    assert runtime.generate_calls == []


def test_gpu_memory_utilization_defaults_and_clamps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VOX_VOXTRAL_GPU_MEMORY_UTILIZATION", raising=False)
    assert worker._gpu_memory_utilization() == 0.1

    monkeypatch.setenv("VOX_VOXTRAL_GPU_MEMORY_UTILIZATION", "0.62")
    assert worker._gpu_memory_utilization() == 0.62

    monkeypatch.setenv("VOX_VOXTRAL_GPU_MEMORY_UTILIZATION", "0.001")
    assert worker._gpu_memory_utilization() == 0.05

    monkeypatch.setenv("VOX_VOXTRAL_GPU_MEMORY_UTILIZATION", "2.0")
    assert worker._gpu_memory_utilization() == 0.95

    monkeypatch.setenv("VOX_VOXTRAL_GPU_MEMORY_UTILIZATION", "not-a-number")
    assert worker._gpu_memory_utilization() == 0.1


def test_main_emits_error_frame_when_runtime_load_fails(monkeypatch: pytest.MonkeyPatch):
    def broken_load_runtime():
        raise RuntimeError("vllm import failed: No module named 'vllm_omni'")

    monkeypatch.setattr(worker, "_load_runtime", broken_load_runtime)
    parent_sock, child_sock = socket.socketpair()
    monkeypatch.setenv(WORKER_FD_ENV, str(child_sock.fileno()))

    exit_code = worker.main(["--model-id", "mistralai/Voxtral-4B-TTS-2603", "--stage-configs-path", "/tmp/s.yaml"])

    child_sock.close()
    with parent_sock, parent_sock.makefile("rb") as stream:
        frame = json.loads(stream.readline())
    assert exit_code == 1
    assert frame["error"] == "RuntimeError: vllm import failed: No module named 'vllm_omni'"


def _install_fake_vllm_runtime(
    monkeypatch: pytest.MonkeyPatch, runtime: _FakeOmniRuntime, tokenizer: Any
) -> dict[str, Any]:
    sampling_params_cls = MagicMock(name="SamplingParams")
    tokenizer_cls = MagicMock()
    tokenizer_cls.from_hf_hub.return_value = tokenizer
    async_omni_cls = MagicMock(return_value=runtime)

    modules = {
        "vllm": ModuleType("vllm"),
        "vllm_omni": ModuleType("vllm_omni"),
        "mistral_common": ModuleType("mistral_common"),
        "mistral_common.protocol": ModuleType("mistral_common.protocol"),
        "mistral_common.protocol.speech": ModuleType("mistral_common.protocol.speech"),
        "mistral_common.protocol.speech.request": ModuleType("mistral_common.protocol.speech.request"),
        "mistral_common.tokens": ModuleType("mistral_common.tokens"),
        "mistral_common.tokens.tokenizers": ModuleType("mistral_common.tokens.tokenizers"),
        "mistral_common.tokens.tokenizers.mistral": ModuleType("mistral_common.tokens.tokenizers.mistral"),
    }
    modules["vllm"].SamplingParams = sampling_params_cls
    modules["vllm_omni"].AsyncOmni = async_omni_cls
    modules["mistral_common.protocol.speech.request"].SpeechRequest = _SpeechRequest
    modules["mistral_common.tokens.tokenizers.mistral"].MistralTokenizer = tokenizer_cls
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return {"AsyncOmni": async_omni_cls, "SamplingParams": sampling_params_cls, "MistralTokenizer": tokenizer_cls}


def test_main_loads_runtime_before_ready_and_serves_synthesize(monkeypatch: pytest.MonkeyPatch):
    audio = np.array([0.5, -0.5], dtype=np.float32)
    runtime = _FakeOmniRuntime([_stage_output(audio, finished=True)])
    tokenizer = _make_tokenizer()
    fakes = _install_fake_vllm_runtime(monkeypatch, runtime, tokenizer)
    monkeypatch.setenv("VOX_VOXTRAL_GPU_MEMORY_UTILIZATION", "0.62")

    parent_sock, child_sock = socket.socketpair()
    monkeypatch.setenv(WORKER_FD_ENV, str(child_sock.fileno()))
    saved_stdout_fd = os.dup(1)
    exit_codes: list[int] = []
    thread = threading.Thread(
        target=lambda: exit_codes.append(
            worker.main(
                [
                    "--model-id",
                    "mistralai/Voxtral-4B-TTS-2603",
                    "--stage-configs-path",
                    "/tmp/voxtral_tts.yaml",
                ]
            )
        )
    )
    try:
        thread.start()
        with parent_sock.makefile("rwb") as stream:
            ready = json.loads(stream.readline())
            assert ready == {"ready": True}
            fakes["AsyncOmni"].assert_called_once_with(
                model="mistralai/Voxtral-4B-TTS-2603",
                stage_configs_path="/tmp/voxtral_tts.yaml",
                gpu_memory_utilization=0.62,
                log_stats=False,
            )
            fakes["MistralTokenizer"].from_hf_hub.assert_called_once_with("mistralai/Voxtral-4B-TTS-2603")
            assert fakes["SamplingParams"].call_count == 2
            fakes["SamplingParams"].assert_called_with(max_tokens=2500)

            stream.write(
                json.dumps({"op": "synthesize", "text": "hello world", "voice": ""}).encode() + b"\n"
            )
            stream.flush()
            response = json.loads(stream.readline())
            assert response["sample_rate"] == 24000
            assert base64.b64decode(response["audio_b64"]) == audio.tobytes()
            speech_request = tokenizer.instruct_tokenizer.encode_speech_request.call_args.args[0]
            assert speech_request.input == "hello world"
            assert speech_request.voice == "neutral_female"
    finally:
        parent_sock.close()
        child_sock.close()
        thread.join(timeout=5.0)
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
    assert not thread.is_alive()
    assert exit_codes == [0]
    assert runtime.shutdown_calls == 1


def test_main_prefers_local_tekken_tokenizer_for_model_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
):
    runtime = _FakeOmniRuntime([])
    tokenizer = _make_tokenizer()
    fakes = _install_fake_vllm_runtime(monkeypatch, runtime, tokenizer)
    model_dir = tmp_path / "voxtral-local"
    model_dir.mkdir()
    (model_dir / "tekken.json").write_text("{}", encoding="utf-8")

    parent_sock, child_sock = socket.socketpair()
    monkeypatch.setenv(WORKER_FD_ENV, str(child_sock.fileno()))
    saved_stdout_fd = os.dup(1)
    thread = threading.Thread(
        target=lambda: worker.main(["--model-id", str(model_dir), "--stage-configs-path", "/tmp/s.yaml"])
    )
    try:
        thread.start()
        with parent_sock.makefile("rwb") as stream:
            assert json.loads(stream.readline()) == {"ready": True}
            fakes["MistralTokenizer"].from_file.assert_called_once_with(str(model_dir / "tekken.json"))
            fakes["MistralTokenizer"].from_hf_hub.assert_not_called()
    finally:
        parent_sock.close()
        child_sock.close()
        thread.join(timeout=5.0)
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
    assert not thread.is_alive()
