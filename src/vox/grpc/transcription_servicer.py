from __future__ import annotations

import logging

import grpc

from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2_grpc
from vox.grpc.operation_errors import operation_error_status
from vox.grpc.transcript_messages import annotate_response, transcribe_response
from vox.operations.errors import OperationError
from vox.operations.transcription import (
    AnnotateRequest,
    TranscriptionRequest,
    annotate_text,
    transcribe,
)

logger = logging.getLogger(__name__)


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
            code, msg = operation_error_status(exc)
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

        return transcribe_response(bundle)

    async def Annotate(self, request, context):
        result = annotate_text(AnnotateRequest(text=request.text or "", language=request.language or "en"))
        return annotate_response(result)
