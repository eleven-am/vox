from __future__ import annotations

import logging
import time

import grpc
from dataclasses import replace

from vox.audio.pipeline import prepare_for_stt
from vox.core.adapter import STTAdapter
from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc

logger = logging.getLogger(__name__)


def _get_default_stt(registry: ModelRegistry, store: BlobStore) -> str:
    for m in store.list_models():
        if m.type.value == "stt":
            return m.full_name
    for name, tags in registry.available_models().items():
        for tag, entry in tags.items():
            if entry.get("type") == "stt":
                return f"{name}:{tag}"
    return ""


class TranscriptionServicer(vox_pb2_grpc.TranscriptionServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Transcribe(self, request, context):
        model = request.model or _get_default_stt(self._registry, self._store)
        if not model:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No model specified and no default STT model available")

        if not request.audio:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No audio data provided")

        format_hint = request.format_hint or None
        audio = prepare_for_stt(request.audio, format_hint=format_hint)

        start_time = time.perf_counter()

        try:
            async with self._scheduler.acquire(model) as adapter:
                if not isinstance(adapter, STTAdapter):
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Model '{model}' is not an STT model")

                result = adapter.transcribe(
                    audio,
                    language=request.language or None,
                    word_timestamps=request.word_timestamps,
                    temperature=request.temperature if request.temperature > 0 else 0.0,
                )
        except ModelNotFoundError as e:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except VoxError as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        except Exception:
            logger.exception(f"Transcription failed for model {model}")
            await context.abort(grpc.StatusCode.INTERNAL, "Internal transcription error")

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        result = replace(result, model=model)

        segments = []
        for s in result.segments:
            words = [
                vox_pb2.WordTimestamp(
                    word=w.word,
                    start_ms=w.start_ms,
                    end_ms=w.end_ms,
                    confidence=w.confidence,
                )
                for w in s.words
            ] if s.words else []

            segments.append(vox_pb2.TranscriptSegment(
                text=s.text,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                words=words,
            ))

        return vox_pb2.TranscribeResponse(
            model=result.model,
            text=result.text,
            language=result.language or "",
            duration_ms=result.duration_ms,
            processing_ms=processing_ms,
            segments=segments,
        )
