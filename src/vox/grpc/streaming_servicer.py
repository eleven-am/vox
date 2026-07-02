from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.core.tasks import reap_task
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.transcript_messages import stream_transcript_result
from vox.operations.errors import (
    OperationError,
    SessionNotConfiguredError,
)
from vox.operations.streaming_transcription import (
    DoneEvent,
    ErrorEvent,
    SessionReadyEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    StreamingTranscriptionConfig,
    StreamingTranscriptionSession,
    TranscriptEvent,
)
from vox.streaming.opus import OPUS_SAMPLE_RATE
from vox.streaming.pipeline import StreamPipelineConfig

logger = logging.getLogger(__name__)


def _event_to_pb(event) -> vox_pb2.StreamOutput | None:
    if isinstance(event, SessionReadyEvent):
        return vox_pb2.StreamOutput(ready=vox_pb2.StreamReady())
    if isinstance(event, SpeechStartedEvent):
        return vox_pb2.StreamOutput(
            speech_started=vox_pb2.StreamSpeechStarted(timestamp_ms=event.timestamp_ms),
        )
    if isinstance(event, SpeechStoppedEvent):
        return vox_pb2.StreamOutput(
            speech_stopped=vox_pb2.StreamSpeechStopped(timestamp_ms=event.timestamp_ms),
        )
    if isinstance(event, TranscriptEvent):
        return vox_pb2.StreamOutput(transcript=stream_transcript_result(event.transcript))
    if isinstance(event, ErrorEvent):
        return vox_pb2.StreamOutput(error=vox_pb2.StreamErrorMessage(message=event.message))
    return None


def _config_from_pb(cfg) -> StreamingTranscriptionConfig:
    return StreamingTranscriptionConfig(
        model=cfg.model or "",
        language=cfg.language or "en",
        sample_rate=cfg.sample_rate or 16_000,
        partials=bool(cfg.partials),
        partial_window_ms=cfg.partial_window_ms or 1500,
        partial_stride_ms=cfg.partial_stride_ms or 700,
        include_word_timestamps=bool(cfg.include_word_timestamps),
        temperature=float(cfg.temperature),
    )


class StreamingServiceServicer(vox_pb2_grpc.StreamingServiceServicer):

    def __init__(
        self,
        store: BlobStore,
        registry: ModelRegistry,
        scheduler: Scheduler,
        pipeline_config: StreamPipelineConfig | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler
        self._pipeline_config = pipeline_config or StreamPipelineConfig()

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
        )
        out_queue: asyncio.Queue[vox_pb2.StreamOutput | None] = asyncio.Queue()

        async def pump_events() -> None:
            try:
                async for event in session.events():
                    pb = _event_to_pb(event)
                    if pb is not None:
                        await out_queue.put(pb)
                    if isinstance(event, DoneEvent):
                        break
            finally:
                await out_queue.put(None)

        async def drain_client() -> None:
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break
                    msg_type = client_msg.WhichOneof("msg")
                    if msg_type == "config":
                        try:
                            await session.configure(_config_from_pb(client_msg.config))
                        except OperationError as exc:
                            await session.report_error(str(exc))
                        continue
                    if msg_type == "audio":
                        try:
                            await session.submit_pcm16(
                                client_msg.audio.pcm16,
                                sample_rate=client_msg.audio.sample_rate or None,
                            )
                        except SessionNotConfiguredError as exc:
                            await session.report_error(str(exc))
                    elif msg_type == "opus_frame":
                        try:
                            await session.submit_opus(
                                client_msg.opus_frame.data,
                                sample_rate=client_msg.opus_frame.sample_rate or OPUS_SAMPLE_RATE,
                                channels=client_msg.opus_frame.channels or 1,
                            )
                        except SessionNotConfiguredError as exc:
                            await session.report_error(str(exc))
                    elif msg_type == "encoded_audio":
                        try:
                            await session.submit_encoded(
                                client_msg.encoded_audio.data,
                                format_hint=client_msg.encoded_audio.format or None,
                            )
                        except SessionNotConfiguredError as exc:
                            await session.report_error(str(exc))
                    elif msg_type == "end_of_stream":
                        break
            finally:
                await session.end_of_stream()

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
            await session.close()
