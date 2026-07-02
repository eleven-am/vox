"""gRPC RTC control service for Vox-hosted WebRTC sessions.

This is the backend-facing counterpart to `/v1/rtc/sessions/{session_id}/control`.
Media stays on WebRTC; the gRPC stream carries only conversation control and
event signaling for an already-created RTC session.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from functools import partial

from vox.core.scheduler import Scheduler
from vox.core.tasks import drain_task, reap_task
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.conversation_commands import rtc_control_message_to_command
from vox.grpc.conversation_events import conversation_error_pb, conversation_event_to_pb
from vox.operations.conversation import (
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConversationOrchestrator,
    execute_conversation_command,
    execute_conversation_session_update,
    serialize_conversation_event,
)
from vox.operations.errors import OperationError
from vox.server.rtc_client_events import control_event_as_client_event, send_client_event_to_browser
from vox.server.rtc_conversation import (
    clear_rtc_audio_if_needed,
    create_rtc_orchestrator_with,
    forward_wire_event_to_browser,
)
from vox.server.rtc_media import cancel_and_drain_media_tasks
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

        async def pump_events() -> None:
            assert orchestrator is not None
            try:
                async for event in orchestrator.events():
                    clear_rtc_audio_if_needed(record, event)
                    forward_wire_event_to_browser(record, serialize_conversation_event(event))
                    if (
                        record is not None
                        and isinstance(event, ConvAudioDeltaEvent)
                        and record.audio_output_track is not None
                    ):
                        continue
                    pb = conversation_event_to_pb(event)
                    if pb is not None:
                        await out_queue.put(pb)
                    if isinstance(event, ConvDoneEvent):
                        break
            finally:
                await out_queue.put(None)

        async def pump_client_events() -> None:
            assert record is not None
            if record.control_events is None:
                return
            while True:
                event = await record.control_events.get()
                if event is None:
                    return
                event_name, payload = control_event_as_client_event(event)
                await out_queue.put(
                    vox_pb2.ConverseServerMessage(
                        client_event=vox_pb2.RtcClientEvent(
                            event=event_name,
                            payload_json=json.dumps(payload),
                        ),
                    )
                )

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
                        await out_queue.put(
                            vox_pb2.ConverseServerMessage(
                                rtc_session_attached=vox_pb2.RtcSessionAttached(
                                    session_id=session_id,
                                    provider="webrtc",
                                ),
                            )
                        )
                        emit_task = asyncio.create_task(pump_events())
                        client_event_task = asyncio.create_task(pump_client_events())
                        continue

                    assert orchestrator is not None

                    try:
                        command = rtc_control_message_to_command(client_msg)
                        if command.kind == "session_update":
                            assert command.config is not None
                            await execute_conversation_session_update(orchestrator, command.config)
                        else:
                            assert command.message is not None
                            await execute_conversation_command(
                                orchestrator,
                                command.message,
                                allow_input_audio=False,
                                client_event_handler=partial(send_client_event_to_browser, record),
                                require_config_message="send session_update first",
                                unknown_message_label="unknown control message kind",
                            )
                    except OperationError as exc:
                        await out_queue.put(conversation_error_pb(str(exc)))
            finally:
                if orchestrator is not None:
                    await orchestrator.end_of_stream(flush_response=False)
                if emit_task is not None:
                    await drain_task(emit_task)
                if record is not None and record.control_events is not None:
                    await record.control_events.put(None)
                if client_event_task is not None:
                    await drain_task(client_event_task)
                else:
                    await out_queue.put(None)

        client_task = asyncio.create_task(drain_client())
        try:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                yield item
        finally:
            await reap_task(client_task)
            if orchestrator is not None:
                await orchestrator.close()
            if record is not None:
                record.orchestrator = None
                if record.audio_output is not None:
                    await record.audio_output.put(None)
                if record.media_events is not None:
                    await record.media_events.put(None)
                await cancel_and_drain_media_tasks(record)
                if record.rtc_peer is not None:
                    with suppress(Exception):
                        await record.rtc_peer.close()
                self._rtc_registry.detach_control(session_id)
                self._rtc_registry.close(session_id)
