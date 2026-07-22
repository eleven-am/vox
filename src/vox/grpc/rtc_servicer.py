"""gRPC RTC signaling and control transport."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

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
    close_grpc_output_queue,
    iter_grpc_stream_lifecycle,
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
        rtc_registry: RtcSessionRegistry,
        speech_context_service: SpeechContextService | None = None,
    ) -> None:
        self._scheduler = scheduler
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
        out_queue: asyncio.Queue[vox_pb2.RtcControlServerMessage | None] = asyncio.Queue()
        runtime: RtcRuntime | None = None

        async def emit(event: dict) -> None:
            await out_queue.put(rtc_runtime_event_pb(event))
            if event.get("type") == "rtc.session.closed":
                await close_grpc_output_queue(out_queue)

        async def emit_conversation(event: ConvEvent, wire: dict) -> None:
            conversation = conversation_event_to_pb(event)
            if conversation is None:
                await out_queue.put(rtc_runtime_event_pb(wire))
                return
            await out_queue.put(vox_pb2.RtcControlServerMessage(conversation=conversation))

        async def drain_client() -> None:
            nonlocal runtime
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break

                    kind = client_msg.WhichOneof("msg")
                    if runtime is None:
                        if kind != "attach":
                            await out_queue.put(rtc_error_pb("send attach first"))
                            break
                        try:
                            runtime = RtcRuntime(
                                scheduler=self._scheduler,
                                registry=self._rtc_registry,
                                session_id=client_msg.attach.session_id,
                                transport="grpc",
                                emit=emit,
                                emit_conversation=emit_conversation,
                                speech_context_service=self._speech_context_service,
                            )
                        except OperationError as exc:
                            await out_queue.put(rtc_error_pb(str(exc)))
                            break
                        await runtime.start()
                        continue

                    try:
                        await runtime.dispatch(rtc_control_message_to_command(client_msg))
                    except OperationError as exc:
                        await out_queue.put(rtc_error_pb_from_exception(exc))
            finally:
                if runtime is not None:
                    await runtime.close(reason="transport_closed")
                await close_grpc_output_queue(out_queue)

        client_task = asyncio.create_task(drain_client())
        async for item in iter_grpc_stream_lifecycle(out_queue, client_task):
            yield item
