"""gRPC RTC control service for Vox-hosted WebRTC sessions.

This is the backend-facing counterpart to `/v1/rtc/sessions/{session_id}/control`.
Media stays on WebRTC; the gRPC stream carries only conversation control and
event signaling for an already-created RTC session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress

from vox.core.scheduler import Scheduler
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.conversation_servicer import _error_pb, _event_to_pb, _pb_to_config
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConversationOrchestrator,
)
from vox.operations.errors import OperationError, SessionAlreadyConfiguredError
from vox.server.rtc_client_events import control_event_as_client_event, send_client_event_to_browser
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry

logger = logging.getLogger(__name__)


async def _cancel_media_tasks(record: RtcSessionRecord) -> None:
    tasks = list(record.media_tasks)
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    record.media_tasks.clear()


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

        async def send_rtc_audio(event: ConvAudioDeltaEvent) -> None:
            if record is not None and record.audio_output_track is not None:
                await record.audio_output_track.enqueue(base64.b64decode(event.audio_b64), event.sample_rate)

        async def wait_for_rtc_playout() -> None:
            if record is not None and record.audio_output_track is not None:
                await record.audio_output_track.wait_until_drained()

        async def pump_events() -> None:
            assert orchestrator is not None
            try:
                async for event in orchestrator.events():
                    if (
                        record is not None
                        and isinstance(event, ConvAudioClearEvent)
                        and record.audio_output_track is not None
                    ):
                        record.audio_output_track.clear()
                    if (
                        record is not None
                        and isinstance(event, ConvAudioDeltaEvent)
                        and record.audio_output_track is not None
                    ):
                        continue
                    pb = _event_to_pb(event)
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
                            await out_queue.put(_error_pb("send attach first"))
                            break
                        session_id = client_msg.attach.session_id
                        record = self._rtc_registry.attach_control(session_id)
                        if record is None:
                            await out_queue.put(
                                _error_pb(
                                    "unknown, expired, or already attached RTC session",
                                )
                            )
                            break
                        orchestrator = ConversationOrchestrator(
                            scheduler=self._scheduler,
                            pace_response_done_to_audio=True,
                            audio_sink=send_rtc_audio,
                            wait_for_output_playout=wait_for_rtc_playout,
                        )
                        record.orchestrator = orchestrator
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

                    if kind == "session_update":
                        try:
                            config = _pb_to_config(client_msg.session_update)
                            await orchestrator.start_session(config)
                        except SessionAlreadyConfiguredError:
                            await out_queue.put(_error_pb("session already configured"))
                        except OperationError as exc:
                            await out_queue.put(_error_pb(str(exc)))
                        continue

                    if kind == "client_event":
                        event_name = client_msg.client_event.event.strip()
                        if not event_name:
                            await out_queue.put(_error_pb("client_event requires a non-empty event"))
                            continue
                        try:
                            payload = json.loads(client_msg.client_event.payload_json or "null")
                        except json.JSONDecodeError as exc:
                            await out_queue.put(_error_pb(f"client_event requires valid payload JSON: {exc}"))
                            continue
                        send_client_event_to_browser(record, event_name, payload)
                        continue

                    if orchestrator.config is None:
                        await out_queue.put(_error_pb("send session_update first"))
                        continue

                    if kind == "response_start":
                        await orchestrator.start_response()
                    elif kind == "response_delta":
                        await orchestrator.append_response_text(client_msg.response_delta.delta)
                    elif kind == "response_commit":
                        await orchestrator.commit_response()
                    elif kind == "response_cancel":
                        await orchestrator.cancel_response()
                    else:
                        await out_queue.put(_error_pb(f"unknown control message kind: {kind!r}"))
            finally:
                if orchestrator is not None:
                    await orchestrator.end_of_stream(flush_response=False)
                if emit_task is not None:
                    with suppress(asyncio.CancelledError):
                        await asyncio.wait_for(emit_task, timeout=5.0)
                    if not emit_task.done():
                        emit_task.cancel()
                        with suppress(Exception):
                            await emit_task
                if record is not None and record.control_events is not None:
                    await record.control_events.put(None)
                if client_event_task is not None:
                    with suppress(asyncio.CancelledError):
                        await asyncio.wait_for(client_event_task, timeout=5.0)
                    if not client_event_task.done():
                        client_event_task.cancel()
                        with suppress(Exception):
                            await client_event_task
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
            client_task.cancel()
            with suppress(Exception):
                await client_task
            if orchestrator is not None:
                await orchestrator.close()
            if record is not None:
                record.orchestrator = None
                if record.audio_output is not None:
                    await record.audio_output.put(None)
                if record.media_events is not None:
                    await record.media_events.put(None)
                await _cancel_media_tasks(record)
                if record.rtc_peer is not None:
                    with suppress(Exception):
                        await record.rtc_peer.close()
                self._rtc_registry.detach_control(session_id)
                self._rtc_registry.close(session_id)
