from __future__ import annotations

import logging

import grpc

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.operation_errors import abort_operation_error, map_operation_errors_to_grpc
from vox.grpc.voice_messages import (
    create_voice_response,
    delete_voice_response,
    list_voices_response,
)
from vox.operations.errors import OperationError
from vox.operations.synthesis import synthesis_request_from_fields, synthesize_raw
from vox.operations.voices import (
    CreateVoiceRequest,
    create_voice,
    delete_voice,
    list_voices,
)

logger = logging.getLogger(__name__)


class SynthesisServicer(vox_pb2_grpc.SynthesisServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Synthesize(self, request, context):
        op_req = synthesis_request_from_fields(
            input=request.input,
            model=request.model,
            voice=request.voice or None,
            speed=request.speed,
            language=request.language or None,
            response_format="wav",
        )
        try:
            iterator = await synthesize_raw(
                scheduler=self._scheduler,
                registry=self._registry,
                store=self._store,
                request=op_req,
            )
            async for chunk in iterator:
                yield vox_pb2.AudioChunk(
                    audio=chunk.audio,
                    sample_rate=chunk.sample_rate,
                    is_final=chunk.is_final,
                )
        except OperationError as exc:
            await abort_operation_error(context, exc)
            return
        except Exception:
            logger.exception("Synthesis failed")
            await context.abort(grpc.StatusCode.INTERNAL, "Internal synthesis error")

    async def ListVoices(self, request, context):
        async with map_operation_errors_to_grpc(context):
            listed = await list_voices(
                scheduler=self._scheduler,
                store=self._store,
                model=request.model or None,
            )

        return list_voices_response(listed)

    async def CreateVoice(self, request, context):
        op_req = CreateVoiceRequest(
            name=request.name,
            audio=request.audio,
            content_type=request.format_hint or None,
            language=request.language or None,
            gender=request.gender or None,
            reference_text=request.reference_text or None,
        )
        try:
            async with map_operation_errors_to_grpc(context):
                voice = create_voice(store=self._store, request=op_req)
        except TypeError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return

        return create_voice_response(voice)

    async def DeleteVoice(self, request, context):
        async with map_operation_errors_to_grpc(context):
            delete_voice(store=self._store, voice_id=request.id)
        return delete_voice_response(request.id)
