"""gRPC RTC control service for Vox-hosted WebRTC sessions.

This is one backend-facing control surface for an already-created RTC session.
Media stays on WebRTC; this stream carries only conversation control and event
signaling.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import partial

from vox.core.scheduler import Scheduler
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.conversation_commands import execute_rtc_control_message
from vox.grpc.conversation_events import (
    conversation_error_pb,
    conversation_event_to_pb,
    rtc_client_event_from_control_event,
    rtc_session_attached_pb,
)
from vox.grpc.streaming_queue import (
    close_grpc_output_queue,
    iter_grpc_stream_lifecycle,
    start_grpc_event_pump,
)
from vox.operations.conversation import ConversationOrchestrator, ConvDoneEvent, ConvEvent
from vox.operations.errors import OperationError
from vox.server.rtc_cleanup import close_rtc_runtime_resources
from vox.server.rtc_client_events import send_client_event_to_browser
from vox.server.rtc_conversation import (
    create_rtc_orchestrator_with,
    prepare_rtc_control_event,
)
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry

logger = logging.getLogger(__name__)


class RtcServicer(vox_pb2_grpc.RtcServiceServicer):
    def __init__(self, *, scheduler: Scheduler, rtc_registry: RtcSessionRegistry) -> None:
        self._scheduler = scheduler
        self._rtc_registry = rtc_registry

    async def Control(
        self,
        request_iterator: AsyncIterator[vox_pb2.RtcControlClientMessage],
        context,
    ) -> AsyncIterator[vox_pb2.ConverseServerMessage]:
        out_queue: asyncio.Queue[vox_pb2.ConverseServerMessage | None] = asyncio.Queue()
        record: RtcSessionRecord | None = None
        orchestrator: ConversationOrchestrator | None = None
        session_id = ""

        def rtc_event_message(event: ConvEvent) -> vox_pb2.ConverseServerMessage | None:
            prepared = prepare_rtc_control_event(
                record=record,
                session_id=session_id,
                event=event,
            )
            return conversation_event_to_pb(event) if prepared.wire is not None else None

        async def pump_client_events() -> None:
            assert record is not None
            if record.control_events is None:
                return
            while True:
                event = await record.control_events.get()
                if event is None:
                    return
                await out_queue.put(rtc_client_event_from_control_event(event))

        async def drain_client() -> None:
            nonlocal record, orchestrator, session_id
            emit_task: asyncio.Task | None = None
            client_event_task: asyncio.Task | None = None
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break

                    kind = client_msg.WhichOneof("msg")
                    if record is None:
                        if kind != "attach":
                            await out_queue.put(conversation_error_pb("send attach first"))
                            break
                        session_id = client_msg.attach.session_id
                        record = self._rtc_registry.attach_control(session_id)
                        if record is None:
                            await out_queue.put(
                                conversation_error_pb(
                                    "unknown, expired, or already attached RTC session",
                                )
                            )
                            break
                        orchestrator = create_rtc_orchestrator_with(
                            scheduler=self._scheduler,
                            record=record,
                            orchestrator_cls=ConversationOrchestrator,
                        )
                        await out_queue.put(rtc_session_attached_pb(session_id))
                        emit_task = start_grpc_event_pump(
                            orchestrator.events(),
                            out_queue,
                            message=rtc_event_message,
                            terminal_types=(ConvDoneEvent,),
                        )
                        client_event_task = asyncio.create_task(pump_client_events())
                        continue

                    assert orchestrator is not None

                    try:
                        await execute_rtc_control_message(
                            orchestrator,
                            client_msg,
                            client_event_handler=partial(send_client_event_to_browser, record),
                        )
                    except OperationError as exc:
                        await out_queue.put(conversation_error_pb(str(exc)))
            finally:
                if (
                    record is not None
                    and orchestrator is not None
                    and emit_task is not None
                    and client_event_task is not None
                ):
                    await close_rtc_runtime_resources(
                        session_id=session_id,
                        registry=self._rtc_registry,
                        record=record,
                        orchestrator=orchestrator,
                        emit_task=emit_task,
                        client_event_task=client_event_task,
                    )
                elif client_event_task is None:
                    await close_grpc_output_queue(out_queue)

        client_task = asyncio.create_task(drain_client())
        async for item in iter_grpc_stream_lifecycle(out_queue, client_task):
            yield item
