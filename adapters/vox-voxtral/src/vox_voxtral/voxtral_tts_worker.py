from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vox.core.worker_host import WORKER_FD_ENV, worker_main
from vox_voxtral.protocol import (
    OP_SYNTHESIZE,
    VOXTRAL_TTS_SAMPLE_RATE,
    SynthesizeRequest,
    SynthesizeResponse,
    accumulate_chunk,
    extract_audio_chunk,
)

logger = logging.getLogger(__name__)


def _gpu_memory_utilization() -> float:
    raw = os.environ.get("VOX_VOXTRAL_GPU_MEMORY_UTILIZATION", "0.1")
    try:
        value = float(raw)
    except ValueError:
        return 0.1
    return min(max(value, 0.05), 0.95)


def _load_runtime() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from vllm import SamplingParams
    from vllm_omni import AsyncOmni

    return AsyncOmni, SamplingParams, SpeechRequest, MistralTokenizer


def _load_tokenizer(mistral_tokenizer_cls: Any, model_id: str) -> Any:
    candidate = Path(model_id)
    if candidate.is_dir() and (candidate / "tekken.json").is_file():
        return mistral_tokenizer_cls.from_file(str(candidate / "tekken.json"))
    return mistral_tokenizer_cls.from_hf_hub(model_id)


async def _generate_audio(
    runtime: Any,
    speech_request_cls: Any,
    tokenizer: Any,
    sampling_params: list[Any],
    *,
    text: str,
    voice: str,
) -> bytes:
    tokenized = tokenizer.instruct_tokenizer.encode_speech_request(
        speech_request_cls(input=text, voice=voice)
    )
    inputs: dict[str, Any] = {
        "prompt_token_ids": tokenized.tokens,
        "additional_information": {"voice": [voice]},
    }

    accumulated_sample = 0
    chunks: list[np.ndarray] = []
    chunk_idx = 0
    async for stage_output in runtime.generate(
        inputs,
        request_id=str(time.time_ns()),
        sampling_params_list=sampling_params,
    ):
        multimodal_output = getattr(stage_output, "multimodal_output", None)
        finished = bool(getattr(stage_output, "finished", False))
        if not multimodal_output or "audio" not in multimodal_output:
            continue

        audio_array = extract_audio_chunk(multimodal_output["audio"], chunk_idx)
        audio_array, accumulated_sample = accumulate_chunk(audio_array, accumulated_sample, finished)
        chunk_idx += 1
        chunks.append(audio_array.astype(np.float32, copy=False))

    if not chunks:
        return b""
    return np.concatenate(chunks).astype(np.float32, copy=False).tobytes()


def build_handler(
    runtime: Any,
    speech_request_cls: Any,
    tokenizer: Any,
    sampling_params: list[Any],
    default_voice: str,
    runner: asyncio.Runner,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def handle(request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op != OP_SYNTHESIZE:
            raise RuntimeError(f"Unsupported op: {op}")
        synthesize = SynthesizeRequest.decode(request)
        audio = runner.run(
            _generate_audio(
                runtime,
                speech_request_cls,
                tokenizer,
                sampling_params,
                text=synthesize.text,
                voice=synthesize.voice or default_voice or "neutral_female",
            )
        )
        return SynthesizeResponse.from_audio(audio, VOXTRAL_TTS_SAMPLE_RATE).payload()

    return handle


def _emit_startup_error(error: BaseException) -> None:
    sock = socket.socket(fileno=os.dup(int(os.environ[WORKER_FD_ENV])))
    with sock, sock.makefile("wb") as stream:
        stream.write(json.dumps({"error": f"{type(error).__name__}: {error}"}).encode() + b"\n")
        stream.flush()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(prog="vox-voxtral-tts-worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--stage-configs-path", required=True)
    parser.add_argument("--default-voice", default="neutral_female")
    args = parser.parse_args(argv)

    try:
        AsyncOmni, SamplingParams, SpeechRequest, MistralTokenizer = _load_runtime()
        tokenizer = _load_tokenizer(MistralTokenizer, args.model_id)
        runtime = AsyncOmni(
            model=args.model_id,
            stage_configs_path=args.stage_configs_path,
            gpu_memory_utilization=_gpu_memory_utilization(),
            log_stats=False,
        )
        sampling_params = [SamplingParams(max_tokens=2500), SamplingParams(max_tokens=2500)]
    except Exception as error:
        logger.exception("Voxtral TTS worker failed to load model")
        _emit_startup_error(error)
        return 1

    try:
        with asyncio.Runner() as runner:
            return worker_main(
                build_handler(runtime, SpeechRequest, tokenizer, sampling_params, args.default_voice, runner)
            )
    finally:
        shutdown = getattr(runtime, "shutdown", None)
        if callable(shutdown):
            shutdown()


if __name__ == "__main__":
    sys.exit(main())
