"""gRPC ConversationService: bidi streaming agent-facing voice orchestration.

Mirrors the WS /v1/conversation protocol (see server/routes/conversation.py).
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
from vox.core.tasks import reap_task
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.conversation_commands import execute_converse_client_message
from vox.grpc.conversation_events import (
    conversation_error_pb,
    conversation_event_to_pb,
)
from vox.operations.conversation import (
    ConvDoneEvent,
    ConversationOrchestrator,
)
from vox.operations.errors import OperationError

logger = logging.getLogger(__name__)


class ConversationServicer(vox_pb2_grpc.ConversationServiceServicer):
    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Converse(
        self,
        request_iterator: AsyncIterator[vox_pb2.ConverseClientMessage],
        context,
    ) -> AsyncIterator[vox_pb2.ConverseServerMessage]:
        orchestrator = ConversationOrchestrator(scheduler=self._scheduler)
        out_queue: asyncio.Queue[vox_pb2.ConverseServerMessage | None] = asyncio.Queue()

        async def pump_events() -> None:
            try:
                async for event in orchestrator.events():
                    pb = conversation_event_to_pb(event)
                    if pb is not None:
                        await out_queue.put(pb)
                    if isinstance(event, ConvDoneEvent):
                        break
            finally:
                await out_queue.put(None)

        async def drain_client() -> None:
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break

                    try:
                        await execute_converse_client_message(orchestrator, client_msg)
                    except OperationError as exc:
                        await out_queue.put(conversation_error_pb(str(exc)))
            finally:
                await orchestrator.end_of_stream()

        emit_task = asyncio.create_task(pump_events())
        client_task = asyncio.create_task(drain_client())
        try:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                yield item
        finally:
            client_task.cancel()
            emit_task.cancel()
            await reap_task(client_task)
            await reap_task(emit_task)
            await orchestrator.close()
