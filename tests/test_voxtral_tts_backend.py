from __future__ import annotations

import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import vox_voxtral
import vox_voxtral.backends as module
from vox_voxtral.backends import WorkerOmniBackend

import vox
from vox.core.types import SynthesizeChunk
from vox.core.worker_host import WorkerError


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
            self.requests.append({"payload": payload, "timeout": timeout})
            if not self.alive:
                raise WorkerError("voxtral-tts worker is not alive")
            if self.request_error is not None:
                raise self.request_error
            return self.responses.pop(0)

        def close(self, grace: float = 5.0) -> None:
            self.closed = True

    monkeypatch.setattr(module, "WorkerHost", _FakeWorkerHost)
    return _FakeWorkerHost


def _spawn(*, env: dict[str, str] | None = None) -> WorkerOmniBackend:
    return WorkerOmniBackend.spawn(
        python_executable="/opt/voxtral/bin/python",
        model_id="mistralai/Voxtral-4B-TTS-2603",
        stage_configs_path="/tmp/voxtral_tts.yaml",
        default_voice="neutral_female",
        env=env if env is not None else {},
    )


async def _collect(backend: WorkerOmniBackend, text: str, voice: str) -> list[SynthesizeChunk]:
    chunks: list[SynthesizeChunk] = []
    async for chunk in backend.generate(text, voice):
        chunks.append(chunk)
    return chunks


def test_spawn_builds_runtime_python_module_argv(fake_host_cls):
    _spawn()

    host = fake_host_cls.instances[-1]
    assert host.argv == [
        "/opt/voxtral/bin/python",
        "-m",
        "vox_voxtral.voxtral_tts_worker",
        "--model-id",
        "mistralai/Voxtral-4B-TTS-2603",
        "--stage-configs-path",
        "/tmp/voxtral_tts.yaml",
        "--default-voice",
        "neutral_female",
    ]
    assert host.name == "voxtral-tts"


def test_spawn_env_allowlist_preserves_runtime_paths_and_drops_secrets(fake_host_cls):
    runtime_env = {
        "PATH": "/usr/bin",
        "LD_LIBRARY_PATH": "/rt/torch/lib:/usr/local/cuda/lib64",
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        "VOX_VOXTRAL_GPU_MEMORY_UTILIZATION": "0.62",
        "HF_TOKEN": "hf-token",
        "CUDA_VISIBLE_DEVICES": "0",
        "AWS_SECRET_ACCESS_KEY": "leak",
        "VOX_API_KEY": "leak",
        "PYTHONPATH": "/app/site-packages",
    }

    _spawn(env=runtime_env)

    host = fake_host_cls.instances[-1]
    assert host.env["PATH"] == "/usr/bin"
    assert host.env["LD_LIBRARY_PATH"] == "/rt/torch/lib:/usr/local/cuda/lib64"
    assert host.env["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"
    assert host.env["VOX_VOXTRAL_GPU_MEMORY_UTILIZATION"] == "0.62"
    assert host.env["HF_TOKEN"] == "hf-token"
    assert host.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "AWS_SECRET_ACCESS_KEY" not in host.env
    assert "VOX_API_KEY" not in host.env
    python_paths = host.env["PYTHONPATH"].split(os.pathsep)
    assert python_paths == [
        str(Path(vox.__file__).resolve().parents[1]),
        str(Path(vox_voxtral.__file__).resolve().parents[1]),
    ]
    assert "/app/site-packages" not in python_paths


def test_spawn_startup_timeout_defaults_generously(fake_host_cls):
    _spawn()

    host = fake_host_cls.instances[-1]
    assert host.startup_timeout == module.DEFAULT_STARTUP_TIMEOUT_SECONDS
    assert host.startup_timeout >= 600.0


def test_spawn_startup_timeout_is_configurable_via_env(fake_host_cls, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(module.STARTUP_TIMEOUT_ENV, "42.5")

    _spawn()

    assert fake_host_cls.instances[-1].startup_timeout == 42.5


def test_spawn_rejects_non_positive_startup_timeout(fake_host_cls, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(module.STARTUP_TIMEOUT_ENV, "0")

    with pytest.raises(RuntimeError, match="must be positive"):
        _spawn()

    assert fake_host_cls.instances == []


def test_generate_round_trips_request_and_rebuilds_chunk_pair(fake_host_cls):
    backend = _spawn()
    host = fake_host_cls.instances[-1]
    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
    host.responses.append({"sample_rate": 24000, "audio_b64": base64.b64encode(audio).decode("ascii")})

    chunks = asyncio.run(_collect(backend, "hello world", "casual_male"))

    assert host.requests == [
        {
            "payload": {"op": "synthesize", "text": "hello world", "voice": "casual_male"},
            "timeout": module.DEFAULT_REQUEST_TIMEOUT_SECONDS,
        }
    ]
    assert len(chunks) == 2
    assert chunks[0].audio == audio
    assert chunks[0].sample_rate == 24000
    assert chunks[0].is_final is False
    assert chunks[1].audio == b""
    assert chunks[1].sample_rate == 24000
    assert chunks[1].is_final is True


def test_generate_request_timeout_is_configurable_via_env(fake_host_cls, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(module.REQUEST_TIMEOUT_ENV, "7.5")
    backend = _spawn()
    host = fake_host_cls.instances[-1]
    host.responses.append({"sample_rate": 24000, "audio_b64": ""})

    asyncio.run(_collect(backend, "hello", "neutral_female"))

    assert host.requests[0]["timeout"] == 7.5


def test_generate_worker_error_fails_loudly(fake_host_cls):
    backend = _spawn()
    host = fake_host_cls.instances[-1]
    host.request_error = WorkerError("voxtral-tts worker killed: timed out after 600.0s (exit code -9)")

    with pytest.raises(WorkerError, match="timed out"):
        asyncio.run(_collect(backend, "hello", "neutral_female"))


def test_generate_after_worker_death_fails_loudly(fake_host_cls):
    backend = _spawn()
    host = fake_host_cls.instances[-1]
    host._alive = False

    with pytest.raises(WorkerError, match="not alive"):
        asyncio.run(_collect(backend, "hello", "neutral_female"))


def test_close_closes_host(fake_host_cls):
    backend = _spawn()
    host = fake_host_cls.instances[-1]

    asyncio.run(backend.close())

    assert host.closed is True


INTEGRATION_WORKER_SOURCE = """
import base64

from vox.core.worker_host import worker_main


def handler(request):
    if request["op"] != "synthesize":
        raise RuntimeError("unexpected op: " + str(request.get("op")))
    audio = (request["text"] + ":" + request["voice"]).encode()
    return {"sample_rate": 24000, "audio_b64": base64.b64encode(audio).decode("ascii")}


raise SystemExit(worker_main(handler))
"""


def test_backend_host_wiring_end_to_end_with_fake_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        module,
        "_worker_argv",
        lambda python_executable, *, model_id, stage_configs_path, default_voice: [
            sys.executable,
            "-c",
            INTEGRATION_WORKER_SOURCE,
        ],
    )
    monkeypatch.setenv(module.STARTUP_TIMEOUT_ENV, "30")

    backend = _spawn(env={"PATH": os.environ["PATH"]})
    try:
        chunks = asyncio.run(_collect(backend, "hello", "casual_male"))

        assert [chunk.audio for chunk in chunks] == [b"hello:casual_male", b""]
        assert [chunk.is_final for chunk in chunks] == [False, True]
        assert {chunk.sample_rate for chunk in chunks} == {24000}
    finally:
        asyncio.run(backend.close())
