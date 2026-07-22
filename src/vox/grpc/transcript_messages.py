from __future__ import annotations

from typing import Any

from vox.grpc import vox_pb2
from vox.operations.transcription import (
    AnnotateResult,
    TranscriptionResultBundle,
    entity_payload,
    entity_payloads,
    transcript_segment_payload,
    transcript_segment_payloads,
    word_timestamp_payload,
    word_timestamp_payloads,
)
from vox.speech_context.types import speech_context_payload
from vox.streaming.types import StreamTranscript


def entity_message(entity: Any) -> vox_pb2.Entity:
    return vox_pb2.Entity(**entity_payload(entity))


def entity_messages(entities: Any) -> list[vox_pb2.Entity]:
    return [vox_pb2.Entity(**payload) for payload in entity_payloads(entities)]


def word_timestamp_message(word: Any) -> vox_pb2.WordTimestamp:
    return vox_pb2.WordTimestamp(**word_timestamp_payload(word))


def word_timestamp_messages(words: Any) -> list[vox_pb2.WordTimestamp]:
    return [vox_pb2.WordTimestamp(**payload) for payload in word_timestamp_payloads(words)]


def transcript_segment_message(segment: Any) -> vox_pb2.TranscriptSegment:
    payload = transcript_segment_payload(segment)
    return vox_pb2.TranscriptSegment(
        text=payload["text"],
        start_ms=payload["start_ms"],
        end_ms=payload["end_ms"],
        words=[vox_pb2.WordTimestamp(**word) for word in payload["words"]],
    )


def transcript_segment_messages(segments: Any) -> list[vox_pb2.TranscriptSegment]:
    return [transcript_segment_message(payload) for payload in transcript_segment_payloads(segments)]


def transcribe_response(bundle: TranscriptionResultBundle) -> vox_pb2.TranscribeResponse:
    result = bundle.result
    message = vox_pb2.TranscribeResponse(
        model=result.model,
        text=result.text,
        language=result.language or "",
        duration_ms=result.duration_ms,
        processing_ms=bundle.processing_ms,
        segments=transcript_segment_messages(result.segments),
        entities=entity_messages(bundle.entities),
        topics=list(bundle.topics),
    )
    if bundle.speech_context is not None:
        message.speech_context.update(speech_context_payload(bundle.speech_context))
    return message


def annotate_response(result: AnnotateResult) -> vox_pb2.AnnotateResponse:
    return vox_pb2.AnnotateResponse(
        entities=entity_messages(result.entities),
        topics=list(result.topics),
    )


def stream_transcript_result(transcript: StreamTranscript) -> vox_pb2.StreamTranscriptResult:
    kwargs = {
        "text": transcript.text,
        "is_partial": transcript.is_partial,
        "start_ms": transcript.start_ms,
        "end_ms": transcript.end_ms,
        "audio_duration_ms": transcript.audio_duration_ms,
        "processing_duration_ms": transcript.processing_duration_ms,
        "model": transcript.model or "",
    }
    if transcript.eou_probability is not None:
        kwargs["eou_probability"] = transcript.eou_probability
    message = vox_pb2.StreamTranscriptResult(**kwargs)
    message.entities.extend(entity_messages(transcript.entities))
    if transcript.topics:
        message.topics.extend(transcript.topics)
    message.words.extend(word_timestamp_messages(transcript.words))
    message.segments.extend(transcript_segment_messages(transcript.segments))
    if transcript.speech_context is not None and not transcript.is_partial:
        message.speech_context.update(speech_context_payload(transcript.speech_context))
    return message
