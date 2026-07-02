from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from vox.grpc import vox_pb2
from vox.operations.transcription import AnnotateResult, TranscriptionResultBundle
from vox.streaming.types import StreamTranscript


def entity_message(entity: Any) -> vox_pb2.Entity:
    return vox_pb2.Entity(
        type=str(_field(entity, "type", "")),
        text=str(_field(entity, "text", "")),
        start_char=_int_field(entity, "start_char"),
        end_char=_int_field(entity, "end_char"),
    )


def entity_messages(entities: Iterable[Any] | None) -> list[vox_pb2.Entity]:
    return [entity_message(entity) for entity in entities or ()]


def word_timestamp_message(word: Any) -> vox_pb2.WordTimestamp:
    kwargs = {
        "word": str(_field(word, "word", "")),
        "start_ms": _int_field(word, "start_ms"),
        "end_ms": _int_field(word, "end_ms"),
    }
    confidence = _field(word, "confidence", None)
    if confidence is not None:
        kwargs["confidence"] = float(confidence)
    return vox_pb2.WordTimestamp(**kwargs)


def word_timestamp_messages(words: Iterable[Any] | None) -> list[vox_pb2.WordTimestamp]:
    return [word_timestamp_message(word) for word in words or ()]


def transcript_segment_message(segment: Any) -> vox_pb2.TranscriptSegment:
    return vox_pb2.TranscriptSegment(
        text=str(_field(segment, "text", "")),
        start_ms=_int_field(segment, "start_ms"),
        end_ms=_int_field(segment, "end_ms"),
        words=word_timestamp_messages(_field(segment, "words", ())),
    )


def transcript_segment_messages(segments: Iterable[Any] | None) -> list[vox_pb2.TranscriptSegment]:
    return [transcript_segment_message(segment) for segment in segments or ()]


def transcribe_response(bundle: TranscriptionResultBundle) -> vox_pb2.TranscribeResponse:
    result = bundle.result
    return vox_pb2.TranscribeResponse(
        model=result.model,
        text=result.text,
        language=result.language or "",
        duration_ms=result.duration_ms,
        processing_ms=bundle.processing_ms,
        segments=transcript_segment_messages(result.segments),
        entities=entity_messages(bundle.entities),
        topics=list(bundle.topics),
    )


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
    return message


def _field(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _int_field(source: Any, key: str) -> int:
    value = _field(source, key, None)
    return int(value) if value is not None else 0
