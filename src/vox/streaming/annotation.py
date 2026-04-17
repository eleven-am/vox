from __future__ import annotations

from vox.core.ner import annotate, entity_to_dict
from vox.streaming.types import StreamTranscript


def enrich_transcript(transcript: StreamTranscript, language: str) -> StreamTranscript:
    if not transcript.text or not transcript.text.strip():
        return transcript
    entities, topics = annotate(transcript.text, language or "en")
    transcript.entities = [entity_to_dict(e) for e in entities] or None
    transcript.topics = topics or None
    return transcript
