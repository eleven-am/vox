from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import os
import socket
import sys
from collections import Counter
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
DEGRADED_OUTPUT_MARKER = "parakeet_output_degraded"

_UNKNOWN_OUTPUT_CHARACTERS = frozenset(("\u2047", "\ufffd"))
_UNKNOWN_OUTPUT_MIN_CHARACTERS = 8
_UNKNOWN_OUTPUT_MIN_RATIO = 0.8
_COLLAPSE_MIN_CHARACTERS = 64
_COLLAPSE_MIN_DURATION_MS = 1_000
_COLLAPSE_MIN_CHARACTERS_PER_SECOND = 40.0
_COLLAPSE_MIN_DOMINANT_RATIO = 0.8
_COLLAPSE_MAX_ALPHANUMERIC_RATIO = 0.1

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


def _output_health(text: str, duration_ms: int) -> dict[str, Any] | None:
    characters = [character for character in text if not character.isspace()]
    character_count = len(characters)
    if character_count == 0:
        return None

    counts = Counter(characters)
    dominant_character, dominant_count = counts.most_common(1)[0]
    unknown_count = sum(counts.get(character, 0) for character in _UNKNOWN_OUTPUT_CHARACTERS)
    unknown_ratio = unknown_count / character_count
    dominant_ratio = dominant_count / character_count
    alphanumeric_ratio = sum(character.isalnum() for character in characters) / character_count
    characters_per_second = character_count / max(duration_ms / 1000.0, 0.001)
    unknown_collapse = unknown_count >= _UNKNOWN_OUTPUT_MIN_CHARACTERS and unknown_ratio >= _UNKNOWN_OUTPUT_MIN_RATIO
    repetitive_collapse = (
        duration_ms >= _COLLAPSE_MIN_DURATION_MS
        and character_count >= _COLLAPSE_MIN_CHARACTERS
        and characters_per_second >= _COLLAPSE_MIN_CHARACTERS_PER_SECOND
        and dominant_ratio >= _COLLAPSE_MIN_DOMINANT_RATIO
        and alphanumeric_ratio <= _COLLAPSE_MAX_ALPHANUMERIC_RATIO
    )
    if not unknown_collapse and not repetitive_collapse:
        return None

    return {
        "marker": DEGRADED_OUTPUT_MARKER,
        "duration_ms": duration_ms,
        "character_count": character_count,
        "characters_per_second": round(characters_per_second, 3),
        "unique_codepoints": len(counts),
        "unknown_count": unknown_count,
        "unknown_ratio": round(unknown_ratio, 6),
        "dominant_codepoint": f"U+{ord(dominant_character):04X}",
        "dominant_ratio": round(dominant_ratio, 6),
        "alphanumeric_ratio": round(alphanumeric_ratio, 6),
    }


def _model_tensor_health(model: Any) -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        return {"scan_error": f"torch_import:{type(error).__name__}"}

    tensor_count = 0
    value_count = 0
    nonfinite_tensor_count = 0
    nonfinite_value_count = 0
    first_nonfinite: list[str] = []
    scan_errors: list[str] = []

    for collection_name, accessor_name in (("parameter", "named_parameters"), ("buffer", "named_buffers")):
        accessor = getattr(model, accessor_name, None)
        if not callable(accessor):
            continue
        try:
            entries = accessor()
        except Exception as error:
            scan_errors.append(f"{collection_name}_enumeration:{type(error).__name__}")
            continue
        for name, tensor in entries:
            tensor_count += 1
            try:
                detached = tensor.detach()
                value_count += int(detached.numel())
                finite = torch.isfinite(detached)
                invalid_values = int((~finite).sum().item())
            except Exception as error:
                scan_errors.append(f"{collection_name}:{name}:{type(error).__name__}")
                continue
            if invalid_values == 0:
                continue
            nonfinite_tensor_count += 1
            nonfinite_value_count += invalid_values
            if len(first_nonfinite) < 8:
                first_nonfinite.append(f"{collection_name}:{name}")

    result: dict[str, Any] = {
        "tensor_count": tensor_count,
        "value_count": value_count,
        "nonfinite_tensor_count": nonfinite_tensor_count,
        "nonfinite_value_count": nonfinite_value_count,
        "first_nonfinite": first_nonfinite,
    }
    if scan_errors:
        result["scan_errors"] = scan_errors[:8]
        result["scan_error_count"] = len(scan_errors)
    return result


def _output_tensors(value: Any, torch: Any) -> list[Any]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [tensor for child in value.values() for tensor in _output_tensors(child, torch)]
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in _output_tensors(child, torch)]
    return []


def _tensor_output_health(value: Any, torch: Any) -> dict[str, Any]:
    tensors = _output_tensors(value, torch)
    value_count = 0
    nonfinite_value_count = 0
    constant_tensor_count = 0
    shapes: list[list[int]] = []
    scan_errors: list[str] = []
    for tensor in tensors[:8]:
        try:
            detached = tensor.detach()
            values = int(detached.numel())
            value_count += values
            shapes.append(list(detached.shape))
            finite = torch.isfinite(detached)
            nonfinite_value_count += int((~finite).sum().item())
            if values > 1 and bool(finite.all().item()):
                constant_tensor_count += int(bool((detached.amin() == detached.amax()).item()))
        except Exception as error:
            scan_errors.append(type(error).__name__)
    result: dict[str, Any] = {
        "tensor_count": len(tensors),
        "scanned_tensor_count": min(len(tensors), 8),
        "value_count": value_count,
        "nonfinite_value_count": nonfinite_value_count,
        "constant_tensor_count": constant_tensor_count,
        "shapes": shapes,
    }
    if scan_errors:
        result["scan_errors"] = scan_errors
    return result


def _diagnostic_module_health(model: Any, path: str, *, word_timestamps: bool) -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        return {"probe_error": f"torch_import:{type(error).__name__}"}

    summaries: dict[str, Any] = {}
    handles: list[Any] = []

    def register(name: str, module: Any) -> None:
        register_hook = getattr(module, "register_forward_hook", None)
        if not callable(register_hook):
            return

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if name not in summaries:
                summaries[name] = _tensor_output_health(output, torch)

        handles.append(register_hook(capture))

    for name in ("preprocessor", "encoder", "decoder", "joint"):
        module = getattr(model, name, None)
        if module is not None:
            register(name, module)

    try:
        transcribe(model, path, word_timestamps=word_timestamps)
    except Exception as error:
        summaries["probe_error"] = type(error).__name__
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.remove()
    summaries["hooked_modules"] = len(handles)
    return summaries


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
            degradation = _output_health(response["text"], int(request.get("duration_ms") or 0))
            if degradation is not None:
                degradation["tensor_health"] = _model_tensor_health(model)
                degradation["module_health"] = _diagnostic_module_health(
                    model,
                    request["path"],
                    word_timestamps=bool(request.get("word_timestamps")),
                )
                logger.error("Parakeet NeMo output degradation detected: %s", degradation)
                response["text"] = ""
                response["words"] = []
                response["degraded"] = degradation
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
