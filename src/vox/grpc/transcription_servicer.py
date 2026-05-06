from __future__ import annotations

import logging

import grpc

from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.operations.errors import (
    EmptyAudioError,
    NoDefaultModelError,
    OperationError,
    WrongModelTypeError,
)
from vox.operations.transcription import (
    AnnotateRequest,
    TranscriptionRequest,
    annotate_text,
    transcribe,
)

logger = logging.getLogger(__name__)


def _operation_error_status(exc: OperationError) -> tuple[grpc.StatusCode, str]:
    if isinstance(exc, (NoDefaultModelError, EmptyAudioError, WrongModelTypeError)):
        return grpc.StatusCode.INVALID_ARGUMENT, str(exc)
    return grpc.StatusCode.INTERNAL, str(exc)


class TranscriptionServicer(vox_pb2_grpc.TranscriptionServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Transcribe(self, request, context):
        op_request = TranscriptionRequest(
            audio=request.audio,
            model=request.model,
            format_hint=request.format_hint or None,
            language=request.language or None,
            word_timestamps=request.word_timestamps,
            temperature=request.temperature if request.temperature > 0 else 0.0,
            annotate_text=True,
        )
        try:
            bundle = await transcribe(
                scheduler=self._scheduler,
                registry=self._registry,
                store=self._store,
                request=op_request,
            )
        except OperationError as exc:
            code, msg = _operation_error_status(exc)
            await context.abort(code, msg)
            return
        except ModelNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            return
        except VoxError as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return
        except Exception:
            logger.exception("Transcription failed")
            await context.abort(grpc.StatusCode.INTERNAL, "Internal transcription error")
            return

        result = bundle.result
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
            processing_ms=bundle.processing_ms,
            segments=segments,
            entities=[
                vox_pb2.Entity(type=e.type, text=e.text, start_char=e.start_char, end_char=e.end_char)
                for e in bundle.entities
            ],
            topics=list(bundle.topics),
        )

    async def Annotate(self, request, context):
        result = annotate_text(AnnotateRequest(text=request.text or "", language=request.language or "en"))
        return vox_pb2.AnnotateResponse(
            entities=[
                vox_pb2.Entity(type=e.type, text=e.text, start_char=e.start_char, end_char=e.end_char)
                for e in result.entities
            ],
            topics=list(result.topics),
        )
