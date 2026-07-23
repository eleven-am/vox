from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SpeechContextStatus = Literal["complete", "partial", "failed"]
SpeechContextTrack = Literal["speaker", "sounds"]


@dataclass(frozen=True, slots=True)
class SpeechContextSpan:
    label: str
    start_ms: int
    end_ms: int
    score: float | None = None


@dataclass(frozen=True, slots=True)
class SpeechContext:
    status: SpeechContextStatus
    emotions: tuple[SpeechContextSpan, ...] | None = None
    vocal: tuple[SpeechContextSpan, ...] | None = None
    sounds: tuple[SpeechContextSpan, ...] | None = None
    unavailable: tuple[SpeechContextTrack, ...] = ()
    schema_version: int = 2


def spans_from_payload(payload: Any, field: str) -> tuple[SpeechContextSpan, ...]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field} result must be an object")
    raw_spans = payload.get(field)
    if not isinstance(raw_spans, list):
        raise ValueError(f"{field} must be a list")

    spans: list[SpeechContextSpan] = []
    for index, raw_span in enumerate(raw_spans):
        if not isinstance(raw_span, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        label = raw_span.get("label")
        start_ms = raw_span.get("start_ms")
        end_ms = raw_span.get("end_ms")
        score = raw_span.get("score")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{field}[{index}].label must be non-empty")
        if isinstance(start_ms, bool) or not isinstance(start_ms, int) or start_ms < 0:
            raise ValueError(f"{field}[{index}].start_ms must be a non-negative integer")
        if isinstance(end_ms, bool) or not isinstance(end_ms, int) or end_ms <= start_ms:
            raise ValueError(f"{field}[{index}].end_ms must be an integer greater than start_ms")
        if field == "sounds":
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{field}[{index}].score must be a number")
            score = float(score)
            if not 0 <= score <= 1:
                raise ValueError(f"{field}[{index}].score must be between zero and one")
        elif score is not None:
            raise ValueError(f"{field}[{index}].score is only valid for sounds")
        spans.append(
            SpeechContextSpan(
                label=label,
                start_ms=start_ms,
                end_ms=end_ms,
                score=score,
            )
        )
    return tuple(spans)


def speech_context_payload(context: SpeechContext) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": context.schema_version,
        "status": context.status,
    }
    for field in ("emotions", "vocal", "sounds"):
        spans = getattr(context, field)
        if spans is not None:
            serialized = []
            for span in spans:
                item = {
                    "label": span.label,
                    "start_ms": span.start_ms,
                    "end_ms": span.end_ms,
                }
                if span.score is not None:
                    item["score"] = span.score
                serialized.append(item)
            payload[field] = serialized
    if context.unavailable:
        payload["unavailable"] = list(context.unavailable)
    return payload
