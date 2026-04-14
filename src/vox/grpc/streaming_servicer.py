from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.core.registry import ModelRegistry
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.streaming.codecs import pcm16_to_float32, resample_audio
from vox.streaming.opus import OPUS_SAMPLE_RATE, OpusStreamDecoder
from vox.streaming.partials import PartialTranscriptService
from vox.streaming.pipeline import StreamPipeline, StreamPipelineConfig
from vox.streaming.session import SpeechSession
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    SpeechStarted,
    SpeechStopped,
    StreamSessionConfig,
    StreamTranscript,
    samples_to_ms,
)

logger = logging.getLogger(__name__)


def _get_default_stt(registry: ModelRegistry, store: BlobStore) -> str:
    for m in store.list_models():
        if m.type.value == "stt":
            return m.full_name
    for name, tags in registry.available_models().items():
        for tag, entry in tags.items():
            if entry.get("type") == "stt":
                return f"{name}:{tag}"
    return ""


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
        pipeline: StreamPipeline | None = None
        session_config: StreamSessionConfig | None = None
        session = SpeechSession()
        opus_decoder: OpusStreamDecoder | None = None
        partial_service: PartialTranscriptService | None = None

        async for client_msg in request_iterator:
            if context.cancelled():
                break

            msg_type = client_msg.WhichOneof("msg")

            if msg_type == "config":
                if session_config is not None:
                    yield vox_pb2.StreamOutput(
                        error=vox_pb2.StreamErrorMessage(message="Session already configured")
                    )
                    continue

                cfg = client_msg.config
                model = cfg.model or _get_default_stt(self._registry, self._store)

                session_config = StreamSessionConfig(
                    language=cfg.language or "en",
                    sample_rate=cfg.sample_rate or TARGET_SAMPLE_RATE,
                    model=model,
                    partials=cfg.partials,
                    partial_window_ms=cfg.partial_window_ms or 1500,
                    partial_stride_ms=cfg.partial_stride_ms or 700,
                    include_word_timestamps=cfg.include_word_timestamps,
                    temperature=cfg.temperature,
                )

                pipeline = StreamPipeline(
                    scheduler=self._scheduler,
                    config=self._pipeline_config,
                )
                pipeline.configure(session_config)

                partial_service = PartialTranscriptService(
                    transcribe_async_fn=pipeline.transcribe_async,
                )

                logger.info("Stream session configured: model=%s, language=%s", model, session_config.language)
                yield vox_pb2.StreamOutput(ready=vox_pb2.StreamReady())
                continue

            if pipeline is None or session_config is None or partial_service is None:
                yield vox_pb2.StreamOutput(
                    error=vox_pb2.StreamErrorMessage(message="Session not configured")
                )
                continue

            if msg_type == "audio":
                audio = pcm16_to_float32(client_msg.audio.pcm16)
                src_rate = client_msg.audio.sample_rate or session_config.sample_rate
                if src_rate != TARGET_SAMPLE_RATE:
                    audio = resample_audio(audio, src_rate, TARGET_SAMPLE_RATE)

                session.append_audio(audio)
                async for event in pipeline.process_audio(audio):
                    yield self._map_event(event, session)

                for msg in await self._handle_partials(session, partial_service, session_config):
                    yield msg

            elif msg_type == "opus_frame":
                opus_frame = client_msg.opus_frame
                if opus_decoder is None:
                    sample_rate = opus_frame.sample_rate or OPUS_SAMPLE_RATE
                    channels = opus_frame.channels or 1
                    opus_decoder = OpusStreamDecoder(sample_rate=sample_rate, channels=channels)

                try:
                    audio = opus_decoder.decode_frame(opus_frame.data)
                    audio = resample_audio(audio, OPUS_SAMPLE_RATE, TARGET_SAMPLE_RATE)

                    session.append_audio(audio)
                    async for event in pipeline.process_audio(audio):
                        yield self._map_event(event, session)

                    for msg in await self._handle_partials(session, partial_service, session_config):
                        yield msg
                except Exception as e:
                    yield vox_pb2.StreamOutput(
                        error=vox_pb2.StreamErrorMessage(message=str(e))
                    )

            elif msg_type == "encoded_audio":
                try:
                    from vox.audio.pipeline import prepare_for_stt

                    encoded = client_msg.encoded_audio
                    audio = prepare_for_stt(encoded.data, format_hint=encoded.format or None)

                    session.append_audio(audio)
                    async for event in pipeline.process_audio(audio):
                        yield self._map_event(event, session)

                    for msg in await self._handle_partials(session, partial_service, session_config):
                        yield msg
                except Exception as e:
                    yield vox_pb2.StreamOutput(
                        error=vox_pb2.StreamErrorMessage(message=str(e))
                    )

            elif msg_type == "end_of_stream":
                break

        if pipeline is not None and partial_service is not None and session_config is not None:
            if opus_decoder is not None:
                try:
                    audio = opus_decoder.flush()
                    if audio.size > 0:
                        audio = resample_audio(audio, OPUS_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                        async for event in pipeline.process_audio(audio):
                            yield self._map_event(event, session)
                except Exception as e:
                    yield vox_pb2.StreamOutput(
                        error=vox_pb2.StreamErrorMessage(message=str(e))
                    )

            remaining = partial_service.flush_remaining_audio(session)
            if remaining is not None and len(remaining) > 0:
                try:
                    duration_ms = samples_to_ms(len(remaining))
                    if duration_ms > 0:
                        transcript = await pipeline.transcribe_async(audio=remaining)
                        if transcript.text.strip():
                            yield self._map_transcript(transcript, is_partial=False)
                except Exception as e:
                    yield vox_pb2.StreamOutput(
                        error=vox_pb2.StreamErrorMessage(message=str(e))
                    )

            pipeline.reset()
            pipeline.shutdown()

    def _map_event(self, event, session: SpeechSession) -> vox_pb2.StreamOutput:
        if isinstance(event, SpeechStarted):
            session.start_speech()
            return vox_pb2.StreamOutput(
                speech_started=vox_pb2.StreamSpeechStarted(timestamp_ms=event.timestamp_ms)
            )
        elif isinstance(event, SpeechStopped):
            session.stop_speech()
            return vox_pb2.StreamOutput(
                speech_stopped=vox_pb2.StreamSpeechStopped(timestamp_ms=event.timestamp_ms)
            )
        elif isinstance(event, StreamTranscript):
            return self._map_transcript(event, is_partial=False)
        return vox_pb2.StreamOutput(
            error=vox_pb2.StreamErrorMessage(message="Unknown event")
        )

    def _map_transcript(self, transcript: StreamTranscript, is_partial: bool) -> vox_pb2.StreamOutput:
        kwargs = {
            "text": transcript.text,
            "is_partial": is_partial or transcript.is_partial,
            "start_ms": transcript.start_ms,
            "end_ms": transcript.end_ms,
            "audio_duration_ms": transcript.audio_duration_ms,
            "processing_duration_ms": transcript.processing_duration_ms,
            "model": transcript.model or "",
        }
        if transcript.eou_probability is not None:
            kwargs["eou_probability"] = transcript.eou_probability

        return vox_pb2.StreamOutput(
            transcript=vox_pb2.StreamTranscriptResult(**kwargs)
        )

    async def _handle_partials(
        self,
        session: SpeechSession,
        partial_service: PartialTranscriptService,
        config: StreamSessionConfig,
    ) -> list[vox_pb2.StreamOutput]:
        messages = []
        if not session.is_active() or not config.partials:
            return messages

        try:
            transcript = await partial_service.generate_partial_async(session, config)
            if transcript:
                messages.append(self._map_transcript(transcript, is_partial=True))
        except Exception as e:
            messages.append(vox_pb2.StreamOutput(
                error=vox_pb2.StreamErrorMessage(message=str(e))
            ))
        return messages
