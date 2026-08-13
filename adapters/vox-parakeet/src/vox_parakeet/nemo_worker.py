from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import socket
import sys
from collections.abc import Callable
from typing import Any

from vox.core.process_memory import runtime_memory_status, trim_process_memory
from vox.core.worker_host import (
    WORKER_FD_ENV,
    install_parent_death_signal,
    worker_main,
    worker_parent_lost,
)

logger = logging.getLogger(__name__)

RUNTIME_IMPORT_ERROR_MARKER = "nemo runtime import failed:"

_CUDA_GRAPH_ERROR_MARKERS = (
    "cudagraph",
    "cuda graph",
    "graph capture",
    "preceding successful capture",
)


def _load_asr_model_class() -> Any:
    try:
        nemo_asr = importlib.import_module("nemo.collections.asr")
    except Exception as exc:
        raise RuntimeError(
            f"{RUNTIME_IMPORT_ERROR_MARKER} Parakeet NeMo worker could not import nemo.collections.asr: {exc}"
        ) from exc

    models = getattr(nemo_asr, "models", None)
    asr_model_class = getattr(models, "ASRModel", None)
    if asr_model_class is None:
        raise RuntimeError(
            f"{RUNTIME_IMPORT_ERROR_MARKER} Parakeet NeMo worker requires nemo.collections.asr.models.ASRModel"
        )
    return asr_model_class


def load_model(model_id: str, checkpoint_path: str | None, device: str) -> Any:
    asr_model_class = _load_asr_model_class()
    if checkpoint_path:
        model = asr_model_class.restore_from(restore_path=checkpoint_path)
    else:
        model = asr_model_class.from_pretrained(model_name=model_id)

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    _disable_cuda_graph_decoding(model)
    return model


def _time_stride_seconds(model: Any) -> float:
    cfg = getattr(model, "cfg", None)
    preprocessor = getattr(cfg, "preprocessor", None)
    window_stride = getattr(preprocessor, "window_stride", None)
    if window_stride is None:
        return 0.01
    return float(window_stride) * 8.0


def _extract_text(result: Any) -> str:
    text = getattr(result, "text", result)
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    return str(text or "").strip()


def _extract_timestamp_dict(result: Any) -> dict[str, Any]:
    timestamp = getattr(result, "timestamp", None)
    if isinstance(timestamp, dict):
        return timestamp

    timestep = getattr(result, "timestep", None)
    if isinstance(timestep, dict):
        return timestep

    return {}


def _extract_word_timestamps(result: Any, model: Any) -> list[dict[str, Any]]:
    timestamp_dict = _extract_timestamp_dict(result)
    entries = timestamp_dict.get("word") or []
    time_stride = _time_stride_seconds(model)
    words: list[dict[str, Any]] = []

    for entry in entries:
        if isinstance(entry, dict):
            word = entry.get("word") or entry.get("char") or entry.get("segment") or ""
            start_offset = entry.get("start_offset", entry.get("start"))
            end_offset = entry.get("end_offset", entry.get("end"))
        else:
            word = getattr(entry, "word", None) or getattr(entry, "char", None) or getattr(entry, "segment", "")
            start_offset = getattr(entry, "start_offset", getattr(entry, "start", None))
            end_offset = getattr(entry, "end_offset", getattr(entry, "end", None))

        if not word or start_offset is None or end_offset is None:
            continue

        words.append(
            {
                "word": str(word),
                "start_ms": int(float(start_offset) * time_stride * 1000),
                "end_ms": int(float(end_offset) * time_stride * 1000),
            }
        )

    return words


def _iter_decoding_objects(model: Any) -> list[Any]:
    objects: list[Any] = []
    decoding = getattr(model, "decoding", None)
    if decoding is not None:
        objects.append(decoding)
        inner = getattr(decoding, "decoding", None)
        if inner is not None:
            objects.append(inner)
            computer = getattr(inner, "decoding_computer", None)
            if computer is not None:
                objects.append(computer)
    return objects


def _disable_cuda_graph_decoding(model: Any) -> bool:
    disabled = False
    for obj in _iter_decoding_objects(model):
        disable = getattr(obj, "disable_cuda_graphs", None)
        if callable(disable):
            try:
                disabled = bool(disable()) or disabled
                logger.info("Disabled Parakeet NeMo CUDA graph decoding via %s", type(obj).__name__)
            except Exception:
                logger.warning(
                    "Failed to disable Parakeet NeMo CUDA graph decoding via %s",
                    type(obj).__name__,
                    exc_info=True,
                )

        for attr in ("use_cuda_graph_decoder", "allow_cuda_graphs", "cuda_graph_decoder"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, False)
                    disabled = True
                except Exception:
                    logger.debug(
                        "Could not set %s=False on %s",
                        attr,
                        type(obj).__name__,
                        exc_info=True,
                    )
    return disabled


def _has_cuda_graph_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if any(marker in message for marker in _CUDA_GRAPH_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def transcribe(model: Any, path: str, *, word_timestamps: bool) -> dict[str, Any]:
    transcribe_kwargs: dict[str, Any] = {"batch_size": 1}
    if word_timestamps:
        transcribe_kwargs["timestamps"] = True
        transcribe_kwargs["return_hypotheses"] = True

    try:
        result = model.transcribe([path], **transcribe_kwargs)
    except Exception as exc:
        if not _has_cuda_graph_error(exc):
            raise
        logger.warning(
            "Parakeet NeMo transcription hit CUDA graph decoding failure; disabling graphs and retrying once"
        )
        _disable_cuda_graph_decoding(model)
        result = model.transcribe([path], **transcribe_kwargs)

    if isinstance(result, tuple):
        result = result[0]
    entry = result[0] if isinstance(result, (list, tuple)) else result
    language = getattr(entry, "language", None)
    return {
        "text": _extract_text(entry),
        "language": language if isinstance(language, str) else None,
        "words": _extract_word_timestamps(entry, model) if word_timestamps else [],
    }


def build_handler(model: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def handle(request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "transcribe":
            response = transcribe(model, request["path"], word_timestamps=bool(request.get("word_timestamps")))
            response["memory"] = runtime_memory_status(device="cuda")
            return response
        if op == "trim":
            return {"memory_trim": trim_process_memory(device="cuda")}
        raise RuntimeError(f"unknown Parakeet NeMo worker op: {op}")

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
    parser = argparse.ArgumentParser(prog="vox-parakeet-nemo-worker")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)
    device = os.environ.get("VOX_PARAKEET_DEVICE", "cuda")

    try:
        model = load_model(args.model_id, args.checkpoint, device)
    except Exception as error:
        logger.exception("Parakeet NeMo worker failed to load model")
        _emit_startup_error(error)
        return 1

    return worker_main(build_handler(model))


if __name__ == "__main__":
    sys.exit(main())
