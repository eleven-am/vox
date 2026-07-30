from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from vox.core.worker_host import (
    WORKER_FD_ENV,
    install_parent_death_signal,
    worker_main,
    worker_parent_lost,
)

logger = logging.getLogger(__name__)


def _load_engine(model_path: str) -> Any:
    import tts
    from model_loader import ModelSource
    from tokenizer import StepAudioTokenizer
    from tts import StepAudioTTS

    original_load_model = tts.model_loader.load_model

    def load_model(*args: Any, **kwargs: Any) -> Any:
        kwargs["attention_config"] = {"backend": "TRITON_ATTN"}
        return original_load_model(*args, **kwargs)

    tokenizer = StepAudioTokenizer(
        str(Path(model_path) / "audio_tokenizer"),
        model_source=ModelSource.LOCAL,
    )
    tts.model_loader.load_model = load_model
    try:
        return StepAudioTTS(
            model_path,
            tokenizer,
            model_source=ModelSource.LOCAL,
            quantization=None,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.1,
            max_model_len=3072,
            enforce_eager=True,
            dtype="bfloat16",
            max_num_seqs=1,
            max_num_batched_tokens=3072,
            cosyvoice_dtype="bfloat16",
            cosyvoice_cuda_graph=False,
        )
    finally:
        tts.model_loader.load_model = original_load_model


def _generate(engine: Any, token_ids: list[int], temperature: float, seed: int | None) -> Any:
    max_tokens = 3072 - len(token_ids)
    if max_tokens <= 0:
        raise RuntimeError("Step-Audio-EditX prompt exceeds the 3072-token runtime limit")
    import torch
    from vllm import SamplingParams

    params: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
    }
    if seed is not None:
        params["seed"] = seed
    outputs = engine.llm.generate(
        [{"prompt_token_ids": token_ids}],
        SamplingParams(**params),
        use_tqdm=False,
    )
    output_ids = list(outputs[0].outputs[0].token_ids)
    if output_ids and output_ids[-1] == 3:
        output_ids.pop()
    if not output_ids:
        raise RuntimeError("Step-Audio-EditX generated no audio tokens")
    return torch.tensor(output_ids, dtype=torch.long)


def _clone(engine: Any, request: dict[str, Any]) -> dict[str, Any]:
    import torch

    reference_path = str(request["reference_path"])
    reference_text = str(request["reference_text"])
    text = str(request["text"])
    output_path = str(request["output_path"])
    temperature = float(request["temperature"])
    seed = request.get("seed")

    tokens, vq02, vq06, speech_feat, _, embedding = engine.preprocess_prompt_wav(reference_path)
    prompt_tokens = engine._encode_audio_edit_clone_prompt(
        text,
        reference_text,
        "vox-reference",
        engine.audio_tokenizer.merge_vq0206_to_token_str(vq02, vq06),
    )
    generated = _generate(engine, prompt_tokens, temperature, int(seed) if seed is not None else None)
    prompt = torch.tensor([tokens], dtype=torch.long) - 65536
    audio = engine.cosy_model.token2wav_nonstream(
        generated - 65536,
        prompt,
        speech_feat.to(torch.bfloat16),
        embedding.to(torch.bfloat16),
    )
    values = np.asarray(audio.detach().cpu(), dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise RuntimeError("Step-Audio-EditX generated empty audio")
    sf.write(output_path, values, 24_000, subtype="FLOAT")
    return {
        "sample_rate": 24_000,
        "samples": int(values.size),
    }


def _handler(engine: Any):
    def handle(request: dict[str, Any]) -> dict[str, Any]:
        if request.get("op") == "clone":
            return _clone(engine, request)
        raise RuntimeError(f"unknown Step-Audio-EditX worker op: {request.get('op')}")

    return handle


def _emit_startup_error(error: BaseException) -> None:
    sock = socket.socket(fileno=os.dup(int(os.environ[WORKER_FD_ENV])))
    with sock, sock.makefile("wb") as stream:
        stream.write(json.dumps({"error": f"{type(error).__name__}: {error}"}).encode() + b"\n")
        stream.flush()


def main(argv: list[str] | None = None) -> int:
    install_parent_death_signal()
    if worker_parent_lost():
        return 1
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(prog="vox-step-audio-editx-worker")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args(argv)
    try:
        engine = _load_engine(args.model_path)
    except Exception as error:
        logger.exception("Step-Audio-EditX worker failed to load")
        _emit_startup_error(error)
        return 1
    return worker_main(_handler(engine))


if __name__ == "__main__":
    raise SystemExit(main())
