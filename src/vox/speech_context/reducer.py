from __future__ import annotations

import math
from typing import Any

MIN_EVENT_SCORE = 0.05
STRONG_EVENT_SCORE = 0.2
MIN_EVENT_WINDOWS = 2
MAX_EVENT_CLASSES_PER_FRAME = 3
ANCESTOR_EXPLANATION_RATIO = 0.8
EVENT_SCORE_DECIMALS = 2
DIAGNOSTIC_SCORE_DECIMALS = 4
DIAGNOSTIC_CLASSES_PER_FRAME = 12
DIAGNOSTIC_CLASS_MAXIMA = 24
DIAGNOSTIC_MAX_FRAMES = 64
FUNCTIONAL_DECIMALS = 3
NONFINITE_FUNCTIONALS = frozenset({"NaN", "Infinity", "-Infinity"})

PROSODY_COLUMNS = {
    "pitch_mean": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "pitch_median": "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "pitch_range": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "pitch_variation": "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "energy_mean": "loudness_sma3_amean",
    "energy_range": "loudness_sma3_pctlrange0-2",
    "energy_peaks": "loudnessPeaksPerSec",
    "hnr": "HNRdBACF_sma3nz_amean",
    "jitter": "jitterLocal_sma3nz_amean",
    "shimmer": "shimmerLocaldB_sma3nz_amean",
    "spectral_variation": "spectralFluxV_sma3nz_amean",
    "voiced_segments": "VoicedSegmentsPerSec",
    "mean_voiced": "MeanVoicedSegmentLengthSec",
    "mean_unvoiced": "MeanUnvoicedSegmentLength",
}


class SpeechContextReductionError(ValueError):
    pass


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeechContextReductionError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise SpeechContextReductionError(f"{name} must be finite")
    return number


def _class_catalog(raw: dict[str, Any]) -> list[dict[str, Any]]:
    classes = raw.get("classes")
    if not isinstance(classes, list):
        raise SpeechContextReductionError("audio event classes must be a list")
    catalog: list[dict[str, Any]] = []
    for position, item in enumerate(classes):
        if not isinstance(item, dict):
            raise SpeechContextReductionError(f"audio event class {position} must be an object")
        if item.get("index") != position:
            raise SpeechContextReductionError(f"audio event class index at position {position} must equal {position}")
        class_id = item.get("id")
        label = item.get("label")
        if not isinstance(class_id, str) or not class_id:
            raise SpeechContextReductionError(f"audio event class {position} id must be non-empty")
        if not isinstance(label, str) or not label:
            raise SpeechContextReductionError(f"audio event class {position} label must be non-empty")
        ancestor_ids = item.get("ancestor_ids", [])
        if not isinstance(ancestor_ids, list) or not all(
            isinstance(ancestor_id, str) and ancestor_id for ancestor_id in ancestor_ids
        ):
            raise SpeechContextReductionError(f"audio event class {position} ancestor_ids must be non-empty strings")
        if len(set(ancestor_ids)) != len(ancestor_ids) or class_id in ancestor_ids:
            raise SpeechContextReductionError(f"audio event class {position} has invalid ancestor_ids")
        catalog.append({"id": class_id, "label": label, "ancestor_ids": ancestor_ids})
    return catalog


def _validated_score_frames(raw: dict[str, Any], class_count: int) -> list[dict[str, Any]]:
    frames = raw.get("scores")
    if not isinstance(frames, list):
        raise SpeechContextReductionError("audio event scores must be a list")
    validated: list[dict[str, Any]] = []
    previous_start = -1.0
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise SpeechContextReductionError(f"audio event score frame {frame_index} must be an object")
        start_ms = _finite_number(frame.get("start_ms"), f"score frame {frame_index} start_ms")
        end_ms = _finite_number(frame.get("end_ms"), f"score frame {frame_index} end_ms")
        if start_ms < 0 or end_ms <= start_ms:
            raise SpeechContextReductionError(
                f"score frame {frame_index} must have a non-negative, increasing interval"
            )
        if start_ms < previous_start:
            raise SpeechContextReductionError("audio event score frames must be time ordered")
        values = frame.get("values")
        if not isinstance(values, list) or len(values) != class_count:
            actual = len(values) if isinstance(values, list) else "non-list"
            raise SpeechContextReductionError(
                f"score vector length at frame {frame_index} must be {class_count}, got {actual}"
            )
        scores = [
            _finite_number(value, f"score frame {frame_index} class {class_index}")
            for class_index, value in enumerate(values)
        ]
        if any(score < 0 or score > 1 for score in scores):
            raise SpeechContextReductionError(f"score frame {frame_index} values must be between zero and one")
        validated.append({"start_ms": start_ms, "end_ms": end_ms, "scores": scores})
        previous_start = start_ms
    return validated


def _score_frames(raw: dict[str, Any], class_count: int) -> list[dict[str, Any]]:
    validated = _validated_score_frames(raw, class_count)
    selected_frames: list[dict[str, Any]] = []
    for frame in validated:
        scores = frame["scores"]
        selected = {
            class_index: scores[class_index]
            for class_index in sorted(
                (index for index, score in enumerate(scores) if score >= MIN_EVENT_SCORE),
                key=lambda index: (-scores[index], index),
            )[:MAX_EVENT_CLASSES_PER_FRAME]
        }
        selected_frames.append({
            "start_ms": frame["start_ms"],
            "end_ms": frame["end_ms"],
            "selected": selected,
        })
    return selected_frames


def summarize_audio_event_scores(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, unthresholded view of model scores for diagnostics."""
    catalog = _class_catalog(raw)
    frames = _validated_score_frames(raw, len(catalog))
    frame_summaries = []
    maxima = [0.0] * len(catalog)
    for frame in frames:
        scores = frame["scores"]
        for index, score in enumerate(scores):
            maxima[index] = max(maxima[index], score)
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        frame_summaries.append({
            "start_ms": frame["start_ms"],
            "end_ms": frame["end_ms"],
            "candidates": [
                {
                    "label": catalog[index]["label"],
                    "score": round(scores[index], DIAGNOSTIC_SCORE_DECIMALS),
                }
                for index in ranked[:DIAGNOSTIC_CLASSES_PER_FRAME]
            ],
        })

    omitted_frames = max(0, len(frame_summaries) - DIAGNOSTIC_MAX_FRAMES)
    if omitted_frames:
        head_count = DIAGNOSTIC_MAX_FRAMES // 2
        tail_count = DIAGNOSTIC_MAX_FRAMES - head_count
        frame_summaries = [
            *frame_summaries[:head_count],
            *frame_summaries[-tail_count:],
        ]

    ranked_maxima = sorted(range(len(maxima)), key=lambda index: (-maxima[index], index))
    return {
        "frame_count": len(frames),
        "omitted_frame_count": omitted_frames,
        "frames": frame_summaries,
        "class_maxima": [
            {
                "label": catalog[index]["label"],
                "score": round(maxima[index], DIAGNOSTIC_SCORE_DECIMALS),
            }
            for index in ranked_maxima[:DIAGNOSTIC_CLASS_MAXIMA]
        ],
    }


def _finish_event(
    catalog_item: dict[str, Any],
    start_ms: float,
    end_ms: float,
    maximum: float,
    windows: int,
    duration_ms: float,
) -> dict[str, Any] | None:
    if windows < MIN_EVENT_WINDOWS and maximum < STRONG_EVENT_SCORE:
        return None
    bounded_start = max(0.0, min(start_ms, duration_ms))
    bounded_end = max(bounded_start, min(end_ms, duration_ms))
    if bounded_end <= bounded_start:
        return None
    return {
        "id": catalog_item["id"],
        "label": catalog_item["label"],
        "start_ms": int(round(bounded_start)),
        "end_ms": int(round(bounded_end)),
        "score": round(maximum, EVENT_SCORE_DECIMALS),
        "_ancestor_ids": catalog_item["ancestor_ids"],
    }


def _ancestor_is_explained(ancestor: dict[str, Any], descendant: dict[str, Any]) -> bool:
    if ancestor["id"] not in descendant["_ancestor_ids"]:
        return False
    overlap = min(ancestor["end_ms"], descendant["end_ms"]) - max(ancestor["start_ms"], descendant["start_ms"])
    duration = ancestor["end_ms"] - ancestor["start_ms"]
    return overlap > 0 and overlap / duration >= ANCESTOR_EXPLANATION_RATIO


def _remove_explained_ancestors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained = [
        event
        for event in events
        if not any(event is not candidate and _ancestor_is_explained(event, candidate) for candidate in events)
    ]
    for event in retained:
        del event["_ancestor_ids"]
    return retained


def _group_event_spans(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        class_id = event["id"]
        candidate = grouped.get(class_id)
        if candidate is None:
            candidate = {"label": event["label"], "spans": []}
            grouped[class_id] = candidate
        candidate["spans"].append(
            [
                event["start_ms"],
                event["end_ms"],
                event["score"],
            ]
        )
    return list(grouped.values())


def reduce_audio_events(raw: dict[str, Any], *, duration_ms: float) -> dict[str, Any]:
    duration = _finite_number(duration_ms, "audio duration_ms")
    if duration <= 0:
        raise SpeechContextReductionError("audio duration_ms must be greater than zero")
    catalog = _class_catalog(raw)
    frames = _score_frames(raw, len(catalog))
    events: list[dict[str, Any]] = []
    for class_index, catalog_item in enumerate(catalog):
        active = [
            (frame["start_ms"], frame["end_ms"], frame["selected"][class_index])
            for frame in frames
            if frame["start_ms"] < duration and class_index in frame["selected"]
        ]
        if not active:
            continue
        run_start, run_end, run_maximum = active[0]
        run_windows = 1
        for start_ms, end_ms, score in active[1:]:
            if start_ms <= run_end:
                run_end = max(run_end, end_ms)
                run_maximum = max(run_maximum, score)
                run_windows += 1
                continue
            event = _finish_event(
                catalog_item,
                run_start,
                run_end,
                run_maximum,
                run_windows,
                duration,
            )
            if event is not None:
                events.append(event)
            run_start, run_end, run_maximum, run_windows = start_ms, end_ms, score, 1
        event = _finish_event(
            catalog_item,
            run_start,
            run_end,
            run_maximum,
            run_windows,
            duration,
        )
        if event is not None:
            events.append(event)
    events = _remove_explained_ancestors(events)
    events.sort(
        key=lambda event: (
            event["start_ms"],
            event["end_ms"],
            -event["score"],
            event["id"],
        )
    )
    return {"candidates": _group_event_spans(events)}


def _functional_value(value: Any, column: str) -> float | None:
    if isinstance(value, str):
        if value not in NONFINITE_FUNCTIONALS:
            raise SpeechContextReductionError(f"prosody functional {column} has unsupported string value {value!r}")
        return None
    return round(_finite_number(value, f"prosody functional {column}"), FUNCTIONAL_DECIMALS)


def _scaled_functional(value: Any, column: str, scale: float = 1.0) -> float | None:
    if isinstance(value, str):
        return _functional_value(value, column)
    return round(_finite_number(value, f"prosody functional {column}") * scale, FUNCTIONAL_DECIMALS)


def reduce_prosody(raw: dict[str, Any]) -> dict[str, Any]:
    functionals = raw.get("functionals")
    if not isinstance(functionals, dict):
        raise SpeechContextReductionError("prosody functionals must be an object")
    columns = functionals.get("columns")
    frames = functionals.get("frames")
    if not isinstance(columns, list) or not all(isinstance(column, str) and column for column in columns):
        raise SpeechContextReductionError("prosody functional columns must be non-empty strings")
    if len(set(columns)) != len(columns):
        raise SpeechContextReductionError("prosody functional columns must be unique")
    if not isinstance(frames, list) or len(frames) != 1 or not isinstance(frames[0], dict):
        raise SpeechContextReductionError("prosody functionals must contain exactly one frame")
    values = frames[0].get("values")
    if not isinstance(values, list) or len(values) != len(columns):
        actual = len(values) if isinstance(values, list) else "non-list"
        raise SpeechContextReductionError(f"prosody functional value count must be {len(columns)}, got {actual}")
    validated: dict[str, Any] = {}
    for column, value in zip(columns, values, strict=True):
        _functional_value(value, column)
        validated[column] = value

    def required(name: str, scale: float = 1.0) -> float | None:
        column = PROSODY_COLUMNS[name]
        if column not in validated:
            raise SpeechContextReductionError(f"required prosody functional {column} is missing")
        return _scaled_functional(validated[column], column, scale)

    return {
        "pitch": {
            "mean_st": required("pitch_mean"),
            "median_st": required("pitch_median"),
            "range_st": required("pitch_range"),
            "variation": required("pitch_variation"),
        },
        "energy": {
            "mean": required("energy_mean"),
            "range": required("energy_range"),
            "peaks_per_second": required("energy_peaks"),
        },
        "voice_quality": {
            "hnr_db": required("hnr"),
            "jitter": required("jitter"),
            "shimmer_db": required("shimmer"),
        },
        "spectral_variation": required("spectral_variation"),
        "delivery": {
            "voiced_segments_per_second": required("voiced_segments"),
            "mean_voiced_ms": required("mean_voiced", 1_000.0),
            "mean_unvoiced_ms": required("mean_unvoiced", 1_000.0),
        },
    }


def offset_audio_events(payload: dict[str, Any], *, offset_ms: int) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SpeechContextReductionError("audio event candidates must be a list")
    shifted: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise SpeechContextReductionError(f"audio event candidate {candidate_index} must be an object")
        label = candidate.get("label")
        spans = candidate.get("spans")
        if not isinstance(label, str) or not label:
            raise SpeechContextReductionError(f"audio event candidate {candidate_index} label must be non-empty")
        if not isinstance(spans, list):
            raise SpeechContextReductionError(f"audio event candidate {candidate_index} spans must be a list")
        shifted_spans: list[list[int | float]] = []
        for span_index, span in enumerate(spans):
            if not isinstance(span, list) or len(span) != 3:
                raise SpeechContextReductionError(
                    f"audio event candidate {candidate_index} span {span_index} must have three values"
                )
            start_ms, end_ms, score = span
            start = int(_finite_number(start_ms, "audio event span start_ms")) + offset_ms
            end = int(_finite_number(end_ms, "audio event span end_ms")) + offset_ms
            confidence = _finite_number(score, "audio event span score")
            if start < 0 or end <= start or not 0 <= confidence <= 1:
                raise SpeechContextReductionError("audio event span is invalid")
            shifted_spans.append([start, end, round(confidence, EVENT_SCORE_DECIMALS)])
        shifted.append({"label": label, "spans": shifted_spans})
    return {"candidates": shifted}


def merge_audio_event_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[list[int | float]]] = {}
    for chunk in chunks:
        candidates = chunk.get("candidates")
        if not isinstance(candidates, list):
            raise SpeechContextReductionError("audio event candidates must be a list")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise SpeechContextReductionError("audio event candidate must be an object")
            label = candidate.get("label")
            spans = candidate.get("spans")
            if not isinstance(label, str) or not label or not isinstance(spans, list):
                raise SpeechContextReductionError("audio event candidate is invalid")
            target = grouped.setdefault(label, [])
            for span in spans:
                if not isinstance(span, list) or len(span) != 3:
                    raise SpeechContextReductionError("audio event span must have three values")
                start = int(_finite_number(span[0], "audio event span start_ms"))
                end = int(_finite_number(span[1], "audio event span end_ms"))
                score = round(_finite_number(span[2], "audio event span score"), EVENT_SCORE_DECIMALS)
                if start < 0 or end <= start or not 0 <= score <= 1:
                    raise SpeechContextReductionError("audio event span is invalid")
                if target and start <= int(target[-1][1]):
                    target[-1][1] = max(int(target[-1][1]), end)
                    target[-1][2] = max(float(target[-1][2]), score)
                else:
                    target.append([start, end, score])
    ordered = sorted(
        (
            {"label": label, "spans": spans}
            for label, spans in grouped.items()
            if spans
        ),
        key=lambda candidate: (candidate["spans"][0][0], candidate["label"]),
    )
    return {"candidates": ordered}


def reduce_speech_context(
    results: dict[str, Any],
    *,
    duration_ms: float,
) -> dict[str, Any]:
    reduced: dict[str, Any] = {"schema_version": 1}
    unavailable: list[str] = []
    prosody = results.get("prosody")
    if isinstance(prosody, dict) and prosody.get("status") == "ok":
        raw = prosody.get("raw")
        if not isinstance(raw, dict):
            raise SpeechContextReductionError("successful prosody result must contain raw output")
        reduced["prosody"] = reduce_prosody(raw)
    else:
        unavailable.append("prosody")
    audio_events = results.get("audio_events")
    if isinstance(audio_events, dict) and audio_events.get("status") == "ok":
        raw = audio_events.get("raw")
        if not isinstance(raw, dict):
            raise SpeechContextReductionError("successful audio event result must contain raw output")
        reduced["audio_events"] = reduce_audio_events(raw, duration_ms=duration_ms)
    else:
        unavailable.append("audio_events")
    if unavailable:
        reduced["status"] = "failed" if len(unavailable) == 2 else "partial"
        reduced["unavailable"] = unavailable
    else:
        reduced["status"] = "complete"
    return reduced
