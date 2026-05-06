from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


VOXTRAL_TTS_SAMPLE_RATE = 24_000

OP_SYNTHESIZE = "synthesize"
OP_SHUTDOWN = "shutdown"


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


def _extract_audio_chunk(audio_chunk: Any, chunk_idx: int) -> np.ndarray:
    if isinstance(audio_chunk, list):
        if not audio_chunk:
            return np.asarray([], dtype=np.float32)
        audio_chunk = audio_chunk[chunk_idx] if chunk_idx < len(audio_chunk) else audio_chunk[-1]
    if hasattr(audio_chunk, "detach"):
        audio_chunk = audio_chunk.float().detach().cpu().numpy()
    return np.asarray(audio_chunk, dtype=np.float32)


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

        audio_array = _extract_audio_chunk(multimodal_output["audio"], chunk_idx)
        if finished and accumulated_sample and len(audio_array) > accumulated_sample:
            audio_array = audio_array[accumulated_sample:]

        accumulated_sample += len(audio_array)
        chunk_idx += 1
        chunks.append(audio_array.astype(np.float32, copy=False))

    if not chunks:
        return b""
    return np.concatenate(chunks).astype(np.float32, copy=False).tobytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--stage-configs-path", required=True)
    parser.add_argument("--default-voice", default="neutral_female")
    args = parser.parse_args()

    AsyncOmni, SamplingParams, SpeechRequest, MistralTokenizer = _load_runtime()
    tokenizer = _load_tokenizer(MistralTokenizer, args.model_id)
    runtime = AsyncOmni(
        model=args.model_id,
        stage_configs_path=args.stage_configs_path,
        gpu_memory_utilization=_gpu_memory_utilization(),
        log_stats=False,
    )
    sampling_params = [SamplingParams(max_tokens=2500), SamplingParams(max_tokens=2500)]

    import base64
    print(json.dumps({"status": "ready"}), flush=True)

    try:
        with asyncio.Runner() as runner:
            for raw_line in sys.stdin:
                if not raw_line.strip():
                    continue
                request = json.loads(raw_line)
                op = request.get("op")
                if op == OP_SHUTDOWN:
                    break
                if op != OP_SYNTHESIZE:
                    print(json.dumps({"status": "error", "error": f"Unsupported op: {op}"}), flush=True)
                    continue

                try:
                    audio = runner.run(
                        _generate_audio(
                            runtime,
                            SpeechRequest,
                            tokenizer,
                            sampling_params,
                            text=str(request.get("text", "")),
                            voice=str(request.get("voice") or args.default_voice or "neutral_female"),
                        )
                    )
                    print(
                        json.dumps(
                            {
                                "status": "ok",
                                "sample_rate": VOXTRAL_TTS_SAMPLE_RATE,
                                "audio_b64": base64.b64encode(audio).decode("ascii"),
                            }
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
    finally:
        shutdown = getattr(runtime, "shutdown", None)
        if callable(shutdown):
            shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
