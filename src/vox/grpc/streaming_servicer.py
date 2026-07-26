from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.streaming_messages import execute_stream_input_message, stream_output_message
from vox.grpc.streaming_queue import (
    GRPC_OUTPUT_QUEUE_MAX,
    iter_grpc_stream_lifecycle,
    start_grpc_event_pump,
)
from vox.operations.streaming_transcription import (
    DoneEvent,
    StreamingTranscriptionSession,
)
from vox.speech_context.service import SpeechContextService
from vox.streaming.pipeline import StreamPipelineConfig

logger = logging.getLogger(__name__)


class StreamingServiceServicer(vox_pb2_grpc.StreamingServiceServicer):
    def __init__(
        self,
        store: BlobStore,
        registry: ModelRegistry,
        scheduler: Scheduler,
        speech_context_service: SpeechContextService | None = None,
        pipeline_config: StreamPipelineConfig | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler
        self._pipeline_config = pipeline_config or StreamPipelineConfig()
        self._speech_context_service = speech_context_service

    async def StreamTranscribe(
        self,
        request_iterator: AsyncIterator[vox_pb2.StreamInput],
        context,
    ) -> AsyncIterator[vox_pb2.StreamOutput]:
        session = StreamingTranscriptionSession(
            scheduler=self._scheduler,
            registry=self._registry,
            store=self._store,
            pipeline_config=self._pipeline_config,
            speech_context_service=self._speech_context_service,
        )
        out_queue: asyncio.Queue[vox_pb2.StreamOutput | None] = asyncio.Queue(maxsize=GRPC_OUTPUT_QUEUE_MAX)
        consumer_closed = asyncio.Event()

        async def drain_client() -> None:
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break
                    if not await execute_stream_input_message(session, client_msg):
                        break
            finally:
                await session.end_of_stream()

        emit_task = start_grpc_event_pump(
            session.events(),
            out_queue,
            message=stream_output_message,
            terminal_types=(DoneEvent,),
            consumer_closed=consumer_closed,
        )
        client_task = asyncio.create_task(drain_client())
        stream = iter_grpc_stream_lifecycle(
            out_queue,
            client_task,
            emit_task,
            cleanup=session.close,
            on_consumer_close=consumer_closed.set,
        )
        try:
            async for item in stream:
                yield item
        finally:
            await stream.aclose()
