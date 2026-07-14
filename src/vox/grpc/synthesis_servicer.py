from __future__ import annotations

import logging

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2_grpc
from vox.grpc.operation_errors import map_operation_errors_to_grpc, map_route_errors_to_grpc
from vox.grpc.synthesis_messages import audio_chunk_message, synthesis_params_from_message
from vox.grpc.voice_messages import (
    create_voice_response,
    delete_voice_response,
    list_voices_response,
)
from vox.operations.synthesis import synthesis_request_from_fields, synthesize_raw
from vox.operations.voices import (
    create_voice,
    create_voice_request_from_fields,
    delete_voice,
    delete_voice_request_from_fields,
    list_voices,
    list_voices_request_from_fields,
)

logger = logging.getLogger(__name__)


class SynthesisServicer(vox_pb2_grpc.SynthesisServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Synthesize(self, request, context):
        async with map_route_errors_to_grpc(
            context,
            logger=logger,
            unexpected_message="Internal synthesis error",
            unexpected_log_message="Synthesis failed",
        ):
            op_req = synthesis_request_from_fields(
                input=request.input,
                model=request.model,
                voice=request.voice or None,
                speed=request.speed,
                language=request.language or None,
                response_format="wav",
                params=synthesis_params_from_message(request.params),
            )
            iterator = await synthesize_raw(
                scheduler=self._scheduler,
                registry=self._registry,
                store=self._store,
                request=op_req,
            )
            async for chunk in iterator:
                yield audio_chunk_message(chunk)

    async def ListVoices(self, request, context):
        async with map_operation_errors_to_grpc(context):
            listed = await list_voices(
                scheduler=self._scheduler,
                store=self._store,
                request=list_voices_request_from_fields(model=request.model),
            )

        return list_voices_response(listed)

    async def CreateVoice(self, request, context):
        async with map_operation_errors_to_grpc(context):
            op_req = create_voice_request_from_fields(
                name=request.name,
                audio=request.audio,
                format_hint=request.format_hint,
                language=request.language or None,
                gender=request.gender or None,
                reference_text=request.reference_text or None,
            )
            voice = create_voice(store=self._store, request=op_req)

        return create_voice_response(voice)

    async def DeleteVoice(self, request, context):
        async with map_operation_errors_to_grpc(context):
            result = delete_voice(
                store=self._store,
                request=delete_voice_request_from_fields(voice_id=request.id),
            )
        return delete_voice_response(result)
