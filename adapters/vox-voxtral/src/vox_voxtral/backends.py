from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from vox.core.types import SynthesizeChunk
from vox_voxtral.protocol import (
    VOXTRAL_TTS_SAMPLE_RATE as _SAMPLE_RATE,
    OkResponse,
    ShutdownRequest,
    SynthesizeRequest,
    accumulate_chunk,
    decode_response,
    extract_audio_chunk,
    is_error,
    is_ok,
    is_ready,
)

VOXTRAL_TTS_SAMPLE_RATE = _SAMPLE_RATE


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


class SubprocessOmniBackend(OmniBackend):

    def __init__(self, worker_proc: subprocess.Popen[str]) -> None:
        self._worker_proc = worker_proc
        self._lock = threading.Lock()

    @classmethod
    def from_worker_script(
        cls,
        *,
        python_executable: str,
        worker_script: str,
        model_id: str,
        stage_configs_path: str,
        default_voice: str,
        env: dict[str, str] | None = None,
    ) -> "SubprocessOmniBackend":
        proc = subprocess.Popen(
            [
                python_executable,
                "-u",
                worker_script,
                "--model-id",
                model_id,
                "--stage-configs-path",
                stage_configs_path,
                "--default-voice",
                default_voice,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        startup_logs: list[str] = []
        while True:
            line = _read_line(proc, allow_empty=True)
            if not line:
                stderr = ""
                if proc.stderr is not None:
                    stderr = proc.stderr.read()
                message = stderr or "\n".join(startup_logs) or "Failed to start Voxtral TTS worker"
                raise RuntimeError(message)

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                startup_logs.append(line.strip())
                continue

            if is_ready(payload):
                return cls(proc)

            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read()
            message = (
                payload.get("error")
                or stderr
                or "\n".join(startup_logs)
                or "Failed to start Voxtral TTS worker"
            )
            raise RuntimeError(message)

    async def generate(
        self,
        text: str,
        voice: str,
    ) -> AsyncIterator[SynthesizeChunk]:
        payload = await asyncio.to_thread(self._request_sync, text=text, voice=voice)
        ok = OkResponse.decode(payload)
        yield SynthesizeChunk(
            audio=ok.audio_bytes(),
            sample_rate=ok.sample_rate,
            is_final=False,
        )
        yield SynthesizeChunk(
            audio=b"",
            sample_rate=ok.sample_rate,
            is_final=True,
        )

    def _request_sync(self, *, text: str, voice: str) -> dict[str, Any]:
        if self._worker_proc is None or self._worker_proc.stdin is None:
            raise RuntimeError("Voxtral TTS worker is not running")

        with self._lock:
            self._worker_proc.stdin.write(SynthesizeRequest.make(text=text, voice=voice).encode() + "\n")
            self._worker_proc.stdin.flush()
            request_logs: list[str] = []
            while True:
                line = _read_line(self._worker_proc)
                try:
                    payload = decode_response(line)
                except json.JSONDecodeError:
                    request_logs.append(line.strip())
                    continue

                if is_ok(payload):
                    return payload

                message = (
                    payload.get("error")
                    or "\n".join(log for log in request_logs if log)
                    or "Voxtral TTS worker request failed"
                )
                raise RuntimeError(message)

    async def close(self) -> None:
        if self._worker_proc is None:
            return
        try:
            if self._worker_proc.stdin is not None:
                self._worker_proc.stdin.write(ShutdownRequest.make().encode() + "\n")
                self._worker_proc.stdin.flush()
        except Exception:
            pass
        try:
            self._worker_proc.terminate()
            self._worker_proc.wait(timeout=10)
        except Exception:
            self._worker_proc.kill()
        self._worker_proc = None


def _read_line(proc: subprocess.Popen[str], *, allow_empty: bool = False) -> str:
    if proc.stdout is None:
        raise RuntimeError("Voxtral TTS worker stdout is not available")

    line = proc.stdout.readline()
    if line:
        return line
    if allow_empty:
        return ""

    stderr = ""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
    raise RuntimeError(stderr or "Voxtral TTS worker exited unexpectedly")
