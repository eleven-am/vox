from __future__ import annotations

import logging

import grpc
from grpc_reflection.v1alpha import reflection

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.conversation_servicer import ConversationServicer
from vox.grpc.health_servicer import HealthServicer
from vox.grpc.interceptor import ApiKeyInterceptor, RequestIdInterceptor
from vox.grpc.model_servicer import ModelServicer
from vox.grpc.rtc_servicer import RtcServicer
from vox.grpc.streaming_servicer import StreamingServiceServicer
from vox.grpc.synthesis_servicer import SynthesisServicer
from vox.grpc.transcription_servicer import TranscriptionServicer
from vox.server.rtc_registry import RtcSessionRegistry
from vox.speech_context.service import SpeechContextService

logger = logging.getLogger(__name__)

_GRPC_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


async def start_grpc_server(
    store: BlobStore,
    registry: ModelRegistry,
    scheduler: Scheduler,
    rtc_registry: RtcSessionRegistry,
    speech_context_service: SpeechContextService,
    port: int = 9090,
) -> grpc.aio.Server:
    server = grpc.aio.server(
        interceptors=(RequestIdInterceptor(), ApiKeyInterceptor()),
        options=(
            ("grpc.max_receive_message_length", _GRPC_MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", _GRPC_MAX_MESSAGE_BYTES),
        ),
    )

    vox_pb2_grpc.add_HealthServiceServicer_to_server(
        HealthServicer(scheduler), server,
    )
    vox_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServicer(store, registry, scheduler), server,
    )
    vox_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServicer(store, registry, scheduler, speech_context_service), server,
    )
    vox_pb2_grpc.add_SynthesisServiceServicer_to_server(
        SynthesisServicer(store, registry, scheduler), server,
    )
    vox_pb2_grpc.add_StreamingServiceServicer_to_server(
        StreamingServiceServicer(store, registry, scheduler, speech_context_service), server,
    )
    vox_pb2_grpc.add_ConversationServiceServicer_to_server(
        ConversationServicer(store, registry, scheduler, speech_context_service), server,
    )
    vox_pb2_grpc.add_RtcServiceServicer_to_server(
        RtcServicer(
            scheduler=scheduler,
            rtc_registry=rtc_registry,
            speech_context_service=speech_context_service,
        ),
        server,
    )

    service_names = (
        vox_pb2.DESCRIPTOR.services_by_name["HealthService"].full_name,
        vox_pb2.DESCRIPTOR.services_by_name["ModelService"].full_name,
        vox_pb2.DESCRIPTOR.services_by_name["TranscriptionService"].full_name,
        vox_pb2.DESCRIPTOR.services_by_name["SynthesisService"].full_name,
        vox_pb2.DESCRIPTOR.services_by_name["StreamingService"].full_name,
        vox_pb2.DESCRIPTOR.services_by_name["ConversationService"].full_name,
        vox_pb2.DESCRIPTOR.services_by_name["RtcService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()

    logger.info("gRPC server listening on %s", listen_addr)
    return server
