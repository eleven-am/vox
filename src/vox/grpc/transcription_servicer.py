from __future__ import annotations

import logging

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2_grpc
from vox.grpc.operation_errors import map_route_errors_to_grpc
from vox.grpc.transcript_messages import annotate_response, transcribe_response
from vox.operations.transcription import (
    annotate_request_from_fields,
    annotate_text,
    transcribe,
    transcription_request_from_fields,
)

logger = logging.getLogger(__name__)


class TranscriptionServicer(vox_pb2_grpc.TranscriptionServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Transcribe(self, request, context):
        async with map_route_errors_to_grpc(
            context,
            logger=logger,
            unexpected_message="Internal transcription error",
            unexpected_log_message="Transcription failed",
        ):
            op_request = transcription_request_from_fields(
                audio=request.audio,
                model=request.model,
                format_hint=request.format_hint or None,
                language=request.language or None,
                word_timestamps=request.word_timestamps,
                temperature=request.temperature,
                annotate_text=True,
            )
            bundle = await transcribe(
                scheduler=self._scheduler,
                registry=self._registry,
                store=self._store,
                request=op_request,
            )

        return transcribe_response(bundle)

    async def Annotate(self, request, context):
        async with map_route_errors_to_grpc(
            context,
            logger=logger,
            unexpected_message="Internal annotation error",
            unexpected_log_message="Annotation failed",
        ):
            result = annotate_text(
                annotate_request_from_fields(text=request.text, language=request.language),
            )
        return annotate_response(result)
