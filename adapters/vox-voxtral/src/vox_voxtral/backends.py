from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from vox.core.types import SynthesizeChunk
from vox.core.worker_host import WorkerHost
from vox_voxtral.protocol import (
    VOXTRAL_TTS_SAMPLE_RATE as _SAMPLE_RATE,
)
from vox_voxtral.protocol import (
    SynthesizeRequest,
    SynthesizeResponse,
    accumulate_chunk,
    extract_audio_chunk,
)

VOXTRAL_TTS_SAMPLE_RATE = _SAMPLE_RATE

DEFAULT_STARTUP_TIMEOUT_SECONDS = 1800.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
STARTUP_TIMEOUT_ENV = "VOX_VOXTRAL_STARTUP_TIMEOUT_S"
REQUEST_TIMEOUT_ENV = "VOX_VOXTRAL_REQUEST_TIMEOUT_S"
_WORKER_ENV_NAMES = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "XDG_CACHE_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
_WORKER_ENV_PREFIXES = (
    "CUDA_",
    "NVIDIA_",
    "HF_",
    "HUGGINGFACE_",
    "TORCH_",
    "PYTORCH_",
    "TORCHINDUCTOR_",
    "TRITON_",
    "NCCL_",
    "VLLM_",
    "VOX_VOXTRAL_",
)


def _worker_python_paths() -> list[str]:
    import vox

    candidates = [
        str(Path(vox.__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[1]),
    ]
    paths: list[str] = []
    for candidate in candidates:
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _worker_env(runtime_env: dict[str, str]) -> dict[str, str]:
    env = {
        name: value
        for name, value in runtime_env.items()
        if name in _WORKER_ENV_NAMES or name.startswith(_WORKER_ENV_PREFIXES)
    }
    env["PYTHONPATH"] = os.pathsep.join(_worker_python_paths())
    return env


def _worker_argv(
    python_executable: str,
    *,
    model_id: str,
    stage_configs_path: str,
    default_voice: str,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "vox_voxtral.voxtral_tts_worker",
        "--model-id",
        model_id,
        "--stage-configs-path",
        stage_configs_path,
        "--default-voice",
        default_voice,
    ]


def _env_timeout_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {raw!r}")
    return value


class OmniBackend(ABC):

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice: str,
    ) -> AsyncIterator[SynthesizeChunk]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


class InProcessOmniBackend(OmniBackend):

    def __init__(
        self,
        runtime: Any,
        tokenizer: Any,
        speech_request_cls: Any,
        sampling_params: list[Any],
    ) -> None:
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._speech_request_cls = speech_request_cls
        self._sampling_params = sampling_params

    async def generate(
        self,
        text: str,
        voice: str,
    ) -> AsyncIterator[SynthesizeChunk]:
        instruct_tokenizer = self._tokenizer.instruct_tokenizer
        tokenized = instruct_tokenizer.encode_speech_request(
            self._speech_request_cls(input=text, voice=voice)
        )
        inputs: dict[str, Any] = {
            "prompt_token_ids": tokenized.tokens,
            "additional_information": {"voice": [voice]},
        }

        accumulated_sample = 0
        chunk_idx = 0
        async for stage_output in self._runtime.generate(
            inputs,
            request_id=str(time.time_ns()),
            sampling_params_list=self._sampling_params,
        ):
            multimodal_output = getattr(stage_output, "multimodal_output", None)
            finished = bool(getattr(stage_output, "finished", False))
            if not multimodal_output or "audio" not in multimodal_output:
                continue

            audio_chunk = multimodal_output["audio"]
            audio_array = extract_audio_chunk(audio_chunk, chunk_idx)
            audio_array, accumulated_sample = accumulate_chunk(audio_array, accumulated_sample, finished)
            chunk_idx += 1

            yield SynthesizeChunk(
                audio=audio_array.astype(np.float32, copy=False).tobytes(),
                sample_rate=VOXTRAL_TTS_SAMPLE_RATE,
                is_final=False,
            )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=VOXTRAL_TTS_SAMPLE_RATE,
            is_final=True,
        )

    async def close(self) -> None:
        shutdown = getattr(self._runtime, "shutdown", None)
        if callable(shutdown):
            shutdown()


class WorkerOmniBackend(OmniBackend):

    def __init__(self, host: WorkerHost) -> None:
        self._host = host

    @classmethod
    def spawn(
        cls,
        *,
        python_executable: str,
        model_id: str,
        stage_configs_path: str,
        default_voice: str,
        env: dict[str, str],
    ) -> WorkerOmniBackend:
        host = WorkerHost(
            _worker_argv(
                python_executable,
                model_id=model_id,
                stage_configs_path=stage_configs_path,
                default_voice=default_voice,
            ),
            env=_worker_env(env),
            name="voxtral-tts",
            startup_timeout=_env_timeout_seconds(STARTUP_TIMEOUT_ENV, DEFAULT_STARTUP_TIMEOUT_SECONDS),
        )
        return cls(host)

    async def generate(
        self,
        text: str,
        voice: str,
    ) -> AsyncIterator[SynthesizeChunk]:
        frame = await asyncio.to_thread(
            self._host.request,
            SynthesizeRequest(text=text, voice=voice).payload(),
            _env_timeout_seconds(REQUEST_TIMEOUT_ENV, DEFAULT_REQUEST_TIMEOUT_SECONDS),
        )
        response = SynthesizeResponse.decode(frame)
        yield SynthesizeChunk(
            audio=response.audio_bytes(),
            sample_rate=response.sample_rate,
            is_final=False,
        )
        yield SynthesizeChunk(
            audio=b"",
            sample_rate=response.sample_rate,
            is_final=True,
        )

    async def close(self) -> None:
        await asyncio.to_thread(self._host.close)
