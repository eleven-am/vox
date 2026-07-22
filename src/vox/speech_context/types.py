from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SpeechContextStatus = Literal["complete", "partial", "failed"]
SpeechContextTrack = Literal["prosody", "audio_events"]


@dataclass(frozen=True, slots=True)
class Pitch:
    mean_st: float | None
    median_st: float | None
    range_st: float | None
    variation: float | None


@dataclass(frozen=True, slots=True)
class Energy:
    mean: float | None
    range: float | None
    peaks_per_second: float | None


@dataclass(frozen=True, slots=True)
class VoiceQuality:
    hnr_db: float | None
    jitter: float | None
    shimmer_db: float | None


@dataclass(frozen=True, slots=True)
class Delivery:
    voiced_segments_per_second: float | None
    mean_voiced_ms: float | None
    mean_unvoiced_ms: float | None


@dataclass(frozen=True, slots=True)
class Prosody:
    pitch: Pitch
    energy: Energy
    voice_quality: VoiceQuality
    spectral_variation: float | None
    delivery: Delivery


@dataclass(frozen=True, slots=True)
class AudioEventSpan:
    start_ms: int
    end_ms: int
    score: float


@dataclass(frozen=True, slots=True)
class AudioEventCandidate:
    label: str
    spans: tuple[AudioEventSpan, ...]


@dataclass(frozen=True, slots=True)
class AudioEvents:
    candidates: tuple[AudioEventCandidate, ...]


@dataclass(frozen=True, slots=True)
class SpeechContext:
    status: SpeechContextStatus
    prosody: Prosody | None = None
    audio_events: AudioEvents | None = None
    unavailable: tuple[SpeechContextTrack, ...] = ()
    schema_version: int = 1


def _optional_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number or null")
    return float(value)


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def prosody_from_payload(payload: Any) -> Prosody:
    value = _object(payload, "prosody")
    pitch = _object(value.get("pitch"), "prosody.pitch")
    energy = _object(value.get("energy"), "prosody.energy")
    voice_quality = _object(value.get("voice_quality"), "prosody.voice_quality")
    delivery = _object(value.get("delivery"), "prosody.delivery")
    return Prosody(
        pitch=Pitch(
            mean_st=_optional_number(pitch.get("mean_st"), "prosody.pitch.mean_st"),
            median_st=_optional_number(pitch.get("median_st"), "prosody.pitch.median_st"),
            range_st=_optional_number(pitch.get("range_st"), "prosody.pitch.range_st"),
            variation=_optional_number(pitch.get("variation"), "prosody.pitch.variation"),
        ),
        energy=Energy(
            mean=_optional_number(energy.get("mean"), "prosody.energy.mean"),
            range=_optional_number(energy.get("range"), "prosody.energy.range"),
            peaks_per_second=_optional_number(
                energy.get("peaks_per_second"),
                "prosody.energy.peaks_per_second",
            ),
        ),
        voice_quality=VoiceQuality(
            hnr_db=_optional_number(voice_quality.get("hnr_db"), "prosody.voice_quality.hnr_db"),
            jitter=_optional_number(voice_quality.get("jitter"), "prosody.voice_quality.jitter"),
            shimmer_db=_optional_number(
                voice_quality.get("shimmer_db"),
                "prosody.voice_quality.shimmer_db",
            ),
        ),
        spectral_variation=_optional_number(
            value.get("spectral_variation"),
            "prosody.spectral_variation",
        ),
        delivery=Delivery(
            voiced_segments_per_second=_optional_number(
                delivery.get("voiced_segments_per_second"),
                "prosody.delivery.voiced_segments_per_second",
            ),
            mean_voiced_ms=_optional_number(
                delivery.get("mean_voiced_ms"),
                "prosody.delivery.mean_voiced_ms",
            ),
            mean_unvoiced_ms=_optional_number(
                delivery.get("mean_unvoiced_ms"),
                "prosody.delivery.mean_unvoiced_ms",
            ),
        ),
    )


def audio_events_from_payload(payload: Any) -> AudioEvents:
    value = _object(payload, "audio_events")
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("audio_events.candidates must be a list")
    candidates: list[AudioEventCandidate] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        candidate = _object(raw_candidate, f"audio_events.candidates[{candidate_index}]")
        label = candidate.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"audio_events.candidates[{candidate_index}].label must be non-empty")
        raw_spans = candidate.get("spans")
        if not isinstance(raw_spans, list):
            raise ValueError(f"audio_events.candidates[{candidate_index}].spans must be a list")
        spans: list[AudioEventSpan] = []
        for span_index, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, list) or len(raw_span) != 3:
                raise ValueError(
                    f"audio_events.candidates[{candidate_index}].spans[{span_index}] must have three values"
                )
            start_ms, end_ms, score = raw_span
            if isinstance(start_ms, bool) or not isinstance(start_ms, int):
                raise ValueError("audio event span start_ms must be an integer")
            if isinstance(end_ms, bool) or not isinstance(end_ms, int) or end_ms <= start_ms:
                raise ValueError("audio event span end_ms must be an integer greater than start_ms")
            score_value = _optional_number(score, "audio event span score")
            if score_value is None or not 0 <= score_value <= 1:
                raise ValueError("audio event span score must be between zero and one")
            spans.append(AudioEventSpan(start_ms=start_ms, end_ms=end_ms, score=score_value))
        candidates.append(AudioEventCandidate(label=label, spans=tuple(spans)))
    return AudioEvents(candidates=tuple(candidates))


def speech_context_payload(context: SpeechContext) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": context.schema_version,
        "status": context.status,
    }
    if context.prosody is not None:
        payload["prosody"] = {
            "pitch": {
                "mean_st": context.prosody.pitch.mean_st,
                "median_st": context.prosody.pitch.median_st,
                "range_st": context.prosody.pitch.range_st,
                "variation": context.prosody.pitch.variation,
            },
            "energy": {
                "mean": context.prosody.energy.mean,
                "range": context.prosody.energy.range,
                "peaks_per_second": context.prosody.energy.peaks_per_second,
            },
            "voice_quality": {
                "hnr_db": context.prosody.voice_quality.hnr_db,
                "jitter": context.prosody.voice_quality.jitter,
                "shimmer_db": context.prosody.voice_quality.shimmer_db,
            },
            "spectral_variation": context.prosody.spectral_variation,
            "delivery": {
                "voiced_segments_per_second": context.prosody.delivery.voiced_segments_per_second,
                "mean_voiced_ms": context.prosody.delivery.mean_voiced_ms,
                "mean_unvoiced_ms": context.prosody.delivery.mean_unvoiced_ms,
            },
        }
    if context.audio_events is not None:
        payload["audio_events"] = {
            "candidates": [
                {
                    "label": candidate.label,
                    "spans": [
                        [span.start_ms, span.end_ms, span.score]
                        for span in candidate.spans
                    ],
                }
                for candidate in context.audio_events.candidates
            ]
        }
    if context.unavailable:
        payload["unavailable"] = list(context.unavailable)
    return payload
