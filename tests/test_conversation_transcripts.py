from __future__ import annotations

import logging

from vox.conversation.transcripts import (
    WIRE_TRANSCRIPT_DONE,
    PendingTranscriptFinalizer,
    coalesce_transcript_payload,
    is_transcript_revision,
    transcript_done_payload,
)
from vox.streaming.types import StreamTranscript


def test_transcript_done_payload_preserves_optional_fields_only_when_present():
    payload = transcript_done_payload(
        StreamTranscript(
            text="hello",
            start_ms=10,
            end_ms=400,
            eou_probability=0.7,
            topics=["greeting"],
            entities=[{"type": "person", "text": "Roy"}],
            words=[{"word": "hello", "start_ms": 10, "end_ms": 400}],
        ),
        language="en",
    )

    assert payload == {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": "hello",
        "language": "en",
        "start_ms": 10,
        "end_ms": 400,
        "eou_probability": 0.7,
        "topics": ["greeting"],
        "entities": [{"type": "person", "text": "Roy"}],
        "words": [{"word": "hello", "start_ms": 10, "end_ms": 400}],
    }

    minimal = transcript_done_payload(StreamTranscript(text="hello"), language="en")
    assert "eou_probability" not in minimal
    assert "topics" not in minimal
    assert "entities" not in minimal
    assert "words" not in minimal


def test_transcript_revision_detection_treats_larger_overlap_as_replacement():
    assert is_transcript_revision("Can you tell me?", "What can you tell me?")
    assert is_transcript_revision("hello there", "hello there now")
    assert not is_transcript_revision("first part", "second part")


def test_coalesce_transcript_payload_keeps_best_revision_instead_of_appending():
    previous = {"type": WIRE_TRANSCRIPT_DONE, "transcript": "Can you tell me?", "start_ms": 0, "end_ms": 1200}
    current = {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": "What can you tell me?",
        "start_ms": 0,
        "end_ms": 1800,
    }

    assert coalesce_transcript_payload(previous, current) == current
    assert coalesce_transcript_payload(current, previous) == current


def test_coalesce_transcript_payload_appends_continuations_and_metadata():
    previous = {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": "first part",
        "language": "en",
        "start_ms": 100,
        "end_ms": 1000,
        "eou_probability": 0.6,
        "topics": ["first"],
        "entities": [{"type": "topic", "text": "first"}],
        "words": [{"word": "first", "start_ms": 100, "end_ms": 300}],
    }
    current = {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": "second part",
        "language": "en",
        "start_ms": 1300,
        "end_ms": 2200,
        "topics": ["first", "second"],
        "entities": [{"type": "topic", "text": "second"}],
        "words": [{"word": "second", "start_ms": 1300, "end_ms": 1600}],
    }

    merged = coalesce_transcript_payload(previous, current)

    assert merged["transcript"] == "first part second part"
    assert merged["start_ms"] == 100
    assert merged["end_ms"] == 2200
    assert merged["eou_probability"] == 0.6
    assert merged["topics"] == ["first", "second"]
    assert merged["entities"] == [
        {"type": "topic", "text": "first"},
        {"type": "topic", "text": "second"},
    ]
    assert merged["words"] == [
        {"word": "first", "start_ms": 100, "end_ms": 300},
        {"word": "second", "start_ms": 1300, "end_ms": 1600},
    ]


def test_pending_transcript_finalizer_remembers_pops_clears_and_logs(caplog):
    finalizer = PendingTranscriptFinalizer(language="en")
    finalizer.remember(StreamTranscript(text="first", start_ms=0, end_ms=100))
    finalizer.remember(StreamTranscript(text="second", start_ms=150, end_ms=300))

    assert finalizer.pending_text() == "first second"
    payload = finalizer.pop()
    assert payload is not None
    assert payload["transcript"] == "first second"
    assert finalizer.pop() is None

    finalizer.remember(StreamTranscript(text="discard me"))
    finalizer.clear()
    assert finalizer.pending_text("fallback") == "fallback"

    caplog.set_level(logging.INFO, logger="vox.conversation.session")
    finalizer.log({"transcript": "logged", "start_ms": 1, "end_ms": 2})
    assert any(
        record.name == "vox.conversation.session"
        and "conversation final transcript emitted" in record.message
        and "logged" in record.message
        for record in caplog.records
    )
