from __future__ import annotations

from vox.grpc import vox_pb2
from vox.grpc.transcript_messages import stream_transcript_result
from vox.operations.streaming_transcription import (
    ErrorEvent,
    SessionEvent,
    SessionReadyEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    StreamingTranscriptionConfig,
    TranscriptEvent,
    streaming_transcription_config_from_fields,
)
from vox.streaming.opus import OPUS_SAMPLE_RATE


async def execute_stream_input_message(
    session,
    client_msg: vox_pb2.StreamInput,
) -> bool:
    msg_type = client_msg.WhichOneof("msg")
    if msg_type == "config":
        await session.configure_or_report(stream_config_request(client_msg.config))
        return True
    if msg_type == "audio":
        await session.submit_pcm16_or_report(
            client_msg.audio.pcm16,
            sample_rate=client_msg.audio.sample_rate or None,
        )
        return True
    if msg_type == "opus_frame":
        await session.submit_opus_or_report(
            client_msg.opus_frame.data,
            sample_rate=client_msg.opus_frame.sample_rate or OPUS_SAMPLE_RATE,
            channels=client_msg.opus_frame.channels or 1,
        )
        return True
    if msg_type == "encoded_audio":
        await session.submit_encoded_or_report(
            client_msg.encoded_audio.data,
            format_hint=client_msg.encoded_audio.format or None,
        )
        return True
    if msg_type == "end_of_stream":
        return False
    return True


def stream_config_request(message: vox_pb2.StreamConfig) -> StreamingTranscriptionConfig:
    return streaming_transcription_config_from_fields(
        model=message.model or "",
        language=message.language or "en",
        sample_rate=message.sample_rate or 16_000,
        partials=bool(message.partials),
        partial_window_ms=message.partial_window_ms or 1500,
        partial_stride_ms=message.partial_stride_ms or 700,
        include_word_timestamps=bool(message.include_word_timestamps),
        temperature=float(message.temperature),
    )


def stream_output_message(event: SessionEvent) -> vox_pb2.StreamOutput | None:
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
