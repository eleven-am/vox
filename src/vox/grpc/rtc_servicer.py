"""gRPC RTC signaling and control transport."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from vox.core.scheduler import Scheduler
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.conversation_commands import rtc_control_message_to_command
from vox.grpc.conversation_events import conversation_event_to_pb
from vox.grpc.rtc_messages import (
    rtc_create_session_request_from_pb,
    rtc_error_pb,
    rtc_error_pb_from_exception,
    rtc_runtime_event_pb,
    rtc_session_bootstrap_pb,
)
from vox.grpc.streaming_queue import (
    GRPC_OUTPUT_QUEUE_MAX,
    close_grpc_output_queue,
    iter_grpc_stream_lifecycle,
    put_grpc_output_queue,
)
from vox.operations.conversation import ConvEvent
from vox.operations.errors import OperationError
from vox.operations.rtc_runtime import RtcRuntime
from vox.operations.rtc_signaling import create_rtc_session
from vox.server.rtc_registry import RtcSessionRegistry
from vox.speech_context.service import SpeechContextService

logger = logging.getLogger(__name__)


class RtcServicer(vox_pb2_grpc.RtcServiceServicer):
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        store: Any | None = None,
        rtc_registry: RtcSessionRegistry,
        speech_context_service: SpeechContextService | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._store = store
        self._rtc_registry = rtc_registry
        self._speech_context_service = speech_context_service

    async def CreateSession(self, request, context):
        result = create_rtc_session(
            registry=self._rtc_registry,
            request=rtc_create_session_request_from_pb(request),
        )
        return rtc_session_bootstrap_pb(result)

    async def Control(
        self,
        request_iterator: AsyncIterator[vox_pb2.RtcControlClientMessage],
        context,
    ) -> AsyncIterator[vox_pb2.RtcControlServerMessage]:
        out_queue: asyncio.Queue[vox_pb2.RtcControlServerMessage | None] = asyncio.Queue(maxsize=GRPC_OUTPUT_QUEUE_MAX)
        consumer_closed = asyncio.Event()
        runtime: RtcRuntime | None = None

        async def emit(event: dict) -> bool:
            delivered = await put_grpc_output_queue(
                out_queue,
                rtc_runtime_event_pb(event),
                consumer_closed=consumer_closed,
            )
            if event.get("type") == "rtc.session.closed" and not consumer_closed.is_set():
                await close_grpc_output_queue(out_queue)
            return delivered

        async def emit_conversation(event: ConvEvent, wire: dict) -> bool:
            conversation = conversation_event_to_pb(event)
            if conversation is None:
                return await put_grpc_output_queue(
                    out_queue,
                    rtc_runtime_event_pb(wire),
                    consumer_closed=consumer_closed,
                )
            return await put_grpc_output_queue(
                out_queue,
                vox_pb2.RtcControlServerMessage(conversation=conversation),
                consumer_closed=consumer_closed,
            )

        async def drain_client() -> None:
            nonlocal runtime
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break

                    kind = client_msg.WhichOneof("msg")
                    if runtime is None:
                        if kind != "attach":
                            await put_grpc_output_queue(
                                out_queue,
                                rtc_error_pb("send attach first"),
                                consumer_closed=consumer_closed,
                            )
                            break
                        try:
                            runtime = RtcRuntime(
                                scheduler=self._scheduler,
                                store=self._store,
                                registry=self._rtc_registry,
                                session_id=client_msg.attach.session_id,
                                transport="grpc",
                                emit=emit,
                                emit_conversation=emit_conversation,
                                speech_context_service=self._speech_context_service,
                            )
                        except OperationError as exc:
                            await put_grpc_output_queue(
                                out_queue,
                                rtc_error_pb(str(exc)),
                                consumer_closed=consumer_closed,
                            )
                            break
                        await runtime.start()
                        continue

                    command = None
                    try:
                        command = rtc_control_message_to_command(client_msg)
                        await runtime.dispatch(command)
                    except OperationError as exc:
                        await put_grpc_output_queue(
                            out_queue,
                            rtc_error_pb_from_exception(
                                exc,
                                generation=getattr(command, "generation", None),
                            ),
                            consumer_closed=consumer_closed,
                        )
            finally:
                if runtime is not None:
                    await runtime.close(reason="transport_closed")
                if not consumer_closed.is_set():
                    await close_grpc_output_queue(out_queue)

        client_task = asyncio.create_task(drain_client())
        stream = iter_grpc_stream_lifecycle(
            out_queue,
            client_task,
            on_consumer_close=consumer_closed.set,
        )
        try:
            async for item in stream:
                yield item
        finally:
            await stream.aclose()
