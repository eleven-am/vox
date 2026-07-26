"""gRPC ConversationService: bidi streaming agent-facing voice orchestration.

One bidi RPC per call; client messages drive a `ConversationOrchestrator`; server
messages are produced by the orchestrator's event stream.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.conversation_commands import converse_client_message_to_command
from vox.grpc.conversation_events import (
    conversation_error_pb_from_exception,
    conversation_event_to_pb,
)
from vox.grpc.streaming_queue import (
    GRPC_OUTPUT_QUEUE_MAX,
    close_grpc_output_queue,
    iter_grpc_stream_lifecycle,
    put_grpc_output_queue,
)
from vox.operations.conversation import (
    ConvDoneEvent,
    ConversationOrchestrator,
)
from vox.operations.conversation_runtime import ConversationRuntime
from vox.operations.errors import OperationError
from vox.speech_context.service import SpeechContextService

logger = logging.getLogger(__name__)


class ConversationServicer(vox_pb2_grpc.ConversationServiceServicer):
    def __init__(
        self,
        store: BlobStore,
        registry: ModelRegistry,
        scheduler: Scheduler,
        speech_context_service: SpeechContextService | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler
        self._speech_context_service = speech_context_service

    async def Converse(
        self,
        request_iterator: AsyncIterator[vox_pb2.ConverseClientMessage],
        context,
    ) -> AsyncIterator[vox_pb2.ConverseServerMessage]:
        orchestrator = ConversationOrchestrator(
            scheduler=self._scheduler,
            store=self._store,
            speech_context_service=self._speech_context_service,
        )
        runtime = ConversationRuntime(
            orchestrator,
            require_config_message="send session_update first",
            unknown_message_label="unknown message kind",
        )
        out_queue: asyncio.Queue[vox_pb2.ConverseServerMessage | None] = asyncio.Queue(maxsize=GRPC_OUTPUT_QUEUE_MAX)
        consumer_closed = asyncio.Event()

        async def emit_event(event) -> None:
            message = conversation_event_to_pb(event)
            if message is not None:
                await put_grpc_output_queue(
                    out_queue,
                    message,
                    consumer_closed=consumer_closed,
                )
            if isinstance(event, ConvDoneEvent) and not consumer_closed.is_set():
                await close_grpc_output_queue(out_queue)

        async def drain_client() -> None:
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break

                    try:
                        await runtime.dispatch(converse_client_message_to_command(client_msg))
                    except OperationError as exc:
                        await put_grpc_output_queue(
                            out_queue,
                            conversation_error_pb_from_exception(exc),
                            consumer_closed=consumer_closed,
                        )
            finally:
                await runtime.end_input()

        emit_task = runtime.start_event_pump(emit_event)
        client_task = asyncio.create_task(drain_client())
        stream = iter_grpc_stream_lifecycle(
            out_queue,
            client_task,
            emit_task,
            cleanup=runtime.close,
            on_consumer_close=consumer_closed.set,
        )
        try:
            async for item in stream:
                yield item
        finally:
            await stream.aclose()
