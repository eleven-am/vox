from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

MIN_SOUND_SCORE = 0.05
STRONG_SOUND_SCORE = 0.2
MIN_SOUND_WINDOWS = 2
MAX_SOUND_CLASSES_PER_FRAME = 3
ANCESTOR_EXPLANATION_RATIO = 0.8
DIAGNOSTIC_SCORE_DECIMALS = 4
DIAGNOSTIC_CLASSES_PER_FRAME = 12
DIAGNOSTIC_CLASS_MAXIMA = 24
DIAGNOSTIC_MAX_FRAMES = 64

AUDIOSET_HUMAN_VOICE_ID = "/m/09l8g"
AUDIOSET_RESPIRATORY_SOUNDS_ID = "/m/09hlz4"
AUDIOSET_SILENCE_ID = "/m/028v0c"

EMOTION_LABELS = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "FEARFUL": "fearful",
    "DISGUSTED": "disgusted",
    "SURPRISED": "surprised",
}

VOCAL_LABELS = {
    "LAUGHTER": "laughter",
    "CRY": "crying",
    "CRYING": "crying",
    "SNEEZE": "sneezing",
    "COUGH": "coughing",
    "BREATH": "breathing",
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


def _tag_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpeechContextReductionError(f"{name} must be a non-empty string")
    if value.startswith("<|") and value.endswith("|>"):
        return value[2:-2]
    return value


def _span(
    label: str,
    start_ms: float,
    end_ms: float,
    *,
    score: float | None = None,
) -> dict[str, Any]:
    span = {
        "label": label.strip().casefold(),
        "start_ms": int(round(start_ms)),
        "end_ms": int(round(end_ms)),
    }
    if score is not None:
        span["score"] = round(score, DIAGNOSTIC_SCORE_DECIMALS)
    return span


def _merge_spans(spans: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        grouped.setdefault(span["label"], []).append(dict(span))

    merged: list[dict[str, Any]] = []
    for occurrences in grouped.values():
        occurrences.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
        current = occurrences[0]
        for occurrence in occurrences[1:]:
            if occurrence["start_ms"] <= current["end_ms"]:
                current["end_ms"] = max(current["end_ms"], occurrence["end_ms"])
                if "score" in occurrence:
                    current["score"] = max(current.get("score", 0.0), occurrence["score"])
            else:
                merged.append(current)
                current = occurrence
        merged.append(current)
    return sorted(merged, key=lambda item: (item["start_ms"], item["end_ms"], item["label"]))


def reduce_speaker_context(raw: dict[str, Any]) -> dict[str, Any]:
    windows = raw.get("windows")
    if not isinstance(windows, list):
        raise SpeechContextReductionError("speaker windows must be a list")

    emotions: list[dict[str, Any]] = []
    vocal: list[dict[str, Any]] = []
    previous_start = -1.0
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise SpeechContextReductionError(f"speaker window {index} must be an object")
        start_ms = _finite_number(window.get("start_ms"), f"speaker window {index} start_ms")
        end_ms = _finite_number(window.get("end_ms"), f"speaker window {index} end_ms")
        if start_ms < 0 or end_ms <= start_ms:
            raise SpeechContextReductionError(f"speaker window {index} must have a non-negative, increasing interval")
        if start_ms < previous_start:
            raise SpeechContextReductionError("speaker windows must be time ordered")
        previous_start = start_ms

        emotion = EMOTION_LABELS.get(_tag_value(window.get("emotion"), f"speaker window {index} emotion").upper())
        if emotion is not None:
            emotions.append(_span(emotion, start_ms, end_ms))

        event = VOCAL_LABELS.get(_tag_value(window.get("event"), f"speaker window {index} event").upper())
        if event is not None:
            vocal.append(_span(event, start_ms, end_ms))

    return {
        "emotions": _merge_spans(emotions),
        "vocal": _merge_spans(vocal),
    }


def _class_catalog(raw: dict[str, Any]) -> list[dict[str, Any]]:
    classes = raw.get("classes")
    if not isinstance(classes, list):
        raise SpeechContextReductionError("sound classes must be a list")
    catalog: list[dict[str, Any]] = []
    for position, item in enumerate(classes):
        if not isinstance(item, dict):
            raise SpeechContextReductionError(f"sound class {position} must be an object")
        if item.get("index") != position:
            raise SpeechContextReductionError(f"sound class index at position {position} must equal {position}")
        class_id = item.get("id")
        label = item.get("label")
        ancestor_ids = item.get("ancestor_ids", [])
        if not isinstance(class_id, str) or not class_id:
            raise SpeechContextReductionError(f"sound class {position} id must be non-empty")
        if not isinstance(label, str) or not label:
            raise SpeechContextReductionError(f"sound class {position} label must be non-empty")
        if not isinstance(ancestor_ids, list) or not all(
            isinstance(ancestor_id, str) and ancestor_id for ancestor_id in ancestor_ids
        ):
            raise SpeechContextReductionError(f"sound class {position} ancestor_ids must be non-empty strings")
        if len(set(ancestor_ids)) != len(ancestor_ids) or class_id in ancestor_ids:
            raise SpeechContextReductionError(f"sound class {position} has invalid ancestor_ids")
        catalog.append(
            {
                "id": class_id,
                "label": label,
                "ancestor_ids": tuple(ancestor_ids),
            }
        )
    return catalog


def _score_frames(raw: dict[str, Any], class_count: int) -> list[dict[str, Any]]:
    frames = raw.get("scores")
    if not isinstance(frames, list):
        raise SpeechContextReductionError("sound scores must be a list")
    validated: list[dict[str, Any]] = []
    previous_start = -1.0
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise SpeechContextReductionError(f"sound score frame {frame_index} must be an object")
        start_ms = _finite_number(
            frame.get("start_ms"),
            f"sound score frame {frame_index} start_ms",
        )
        end_ms = _finite_number(
            frame.get("end_ms"),
            f"sound score frame {frame_index} end_ms",
        )
        if start_ms < 0 or end_ms <= start_ms:
            raise SpeechContextReductionError(
                f"sound score frame {frame_index} must have a non-negative, increasing interval"
            )
        if start_ms < previous_start:
            raise SpeechContextReductionError("sound score frames must be time ordered")
        values = frame.get("values")
        if not isinstance(values, list) or len(values) != class_count:
            actual = len(values) if isinstance(values, list) else "non-list"
            raise SpeechContextReductionError(
                f"sound score vector length at frame {frame_index} must be {class_count}, got {actual}"
            )
        scores = [
            _finite_number(value, f"sound score frame {frame_index} class {class_index}")
            for class_index, value in enumerate(values)
        ]
        if any(score < 0 or score > 1 for score in scores):
            raise SpeechContextReductionError(f"sound score frame {frame_index} values must be between zero and one")
        validated.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "scores": scores,
            }
        )
        previous_start = start_ms
    return validated


def _is_public_sound(item: dict[str, Any]) -> bool:
    class_id = item["id"]
    ancestors = item["ancestor_ids"]
    return (
        class_id != AUDIOSET_SILENCE_ID
        and class_id not in {AUDIOSET_HUMAN_VOICE_ID, AUDIOSET_RESPIRATORY_SOUNDS_ID}
        and AUDIOSET_HUMAN_VOICE_ID not in ancestors
        and AUDIOSET_RESPIRATORY_SOUNDS_ID not in ancestors
    )


def summarize_sound_scores(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, unthresholded model view for diagnostics only."""
    catalog = _class_catalog(raw)
    frames = _score_frames(raw, len(catalog))
    frame_summaries = []
    maxima = [0.0] * len(catalog)
    for frame in frames:
        scores = frame["scores"]
        for index, score in enumerate(scores):
            maxima[index] = max(maxima[index], score)
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        frame_summaries.append(
            {
                "start_ms": frame["start_ms"],
                "end_ms": frame["end_ms"],
                "candidates": [
                    {
                        "label": catalog[index]["label"],
                        "score": round(scores[index], DIAGNOSTIC_SCORE_DECIMALS),
                    }
                    for index in ranked[:DIAGNOSTIC_CLASSES_PER_FRAME]
                ],
            }
        )

    omitted_frames = max(0, len(frame_summaries) - DIAGNOSTIC_MAX_FRAMES)
    if omitted_frames:
        head_count = DIAGNOSTIC_MAX_FRAMES // 2
        frame_summaries = [
            *frame_summaries[:head_count],
            *frame_summaries[-(DIAGNOSTIC_MAX_FRAMES - head_count) :],
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


def _ancestor_is_explained(ancestor: dict[str, Any], descendant: dict[str, Any]) -> bool:
    if ancestor["id"] not in descendant["ancestor_ids"]:
        return False
    overlap = min(ancestor["end_ms"], descendant["end_ms"]) - max(
        ancestor["start_ms"],
        descendant["start_ms"],
    )
    duration = ancestor["end_ms"] - ancestor["start_ms"]
    return overlap > 0 and overlap / duration >= ANCESTOR_EXPLANATION_RATIO


def reduce_sound_events(raw: dict[str, Any], *, duration_ms: float) -> dict[str, Any]:
    duration = _finite_number(duration_ms, "audio duration_ms")
    if duration <= 0:
        raise SpeechContextReductionError("audio duration_ms must be greater than zero")

    catalog = _class_catalog(raw)
    frames = _score_frames(raw, len(catalog))
    eligible = {index for index, item in enumerate(catalog) if _is_public_sound(item)}
    selected_frames: list[dict[str, Any]] = []
    for frame in frames:
        ranked = sorted(
            (index for index in eligible if frame["scores"][index] >= MIN_SOUND_SCORE),
            key=lambda index: (-frame["scores"][index], index),
        )[:MAX_SOUND_CLASSES_PER_FRAME]
        selected_frames.append(
            {
                "start_ms": frame["start_ms"],
                "end_ms": frame["end_ms"],
                "selected": {index: frame["scores"][index] for index in ranked},
            }
        )

    events: list[dict[str, Any]] = []
    for class_index in sorted(eligible):
        active = [
            (
                frame["start_ms"],
                frame["end_ms"],
                frame["selected"][class_index],
            )
            for frame in selected_frames
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
            if run_windows >= MIN_SOUND_WINDOWS or run_maximum >= STRONG_SOUND_SCORE:
                events.append(
                    {
                        **catalog[class_index],
                        "start_ms": max(0, int(round(run_start))),
                        "end_ms": min(int(round(run_end)), int(round(duration))),
                        "score": run_maximum,
                    }
                )
            run_start, run_end, run_maximum, run_windows = start_ms, end_ms, score, 1
        if run_windows >= MIN_SOUND_WINDOWS or run_maximum >= STRONG_SOUND_SCORE:
            events.append(
                {
                    **catalog[class_index],
                    "start_ms": max(0, int(round(run_start))),
                    "end_ms": min(int(round(run_end)), int(round(duration))),
                    "score": run_maximum,
                }
            )

    retained = [
        event
        for event in events
        if event["end_ms"] > event["start_ms"]
        and not any(event is not candidate and _ancestor_is_explained(event, candidate) for candidate in events)
    ]
    return {
        "sounds": _merge_spans(
            _span(
                event["label"],
                event["start_ms"],
                event["end_ms"],
                score=event["score"],
            )
            for event in retained
        )
    }


def offset_context_spans(payload: dict[str, Any], *, offset_ms: int) -> dict[str, Any]:
    shifted: dict[str, Any] = {}
    for field in ("emotions", "vocal", "sounds"):
        if field not in payload:
            continue
        shifted[field] = [
            {
                **span,
                "start_ms": span["start_ms"] + offset_ms,
                "end_ms": span["end_ms"] + offset_ms,
            }
            for span in payload[field]
        ]
    return shifted


def merge_context_chunks(
    chunks: Iterable[dict[str, Any]],
    *,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    merged = {field: [] for field in fields}
    for chunk in chunks:
        for field in fields:
            merged[field].extend(chunk.get(field, []))
    return {field: _merge_spans(spans) for field, spans in merged.items()}


def reduce_speech_context(
    results: dict[str, dict[str, Any]],
    *,
    duration_ms: float,
) -> dict[str, Any]:
    unavailable: list[str] = []
    payload: dict[str, Any] = {
        "schema_version": 2,
    }

    speaker = results.get("speaker")
    if speaker is None or speaker.get("status") != "ok":
        unavailable.append("speaker")
    else:
        reduced_speaker = reduce_speaker_context(speaker.get("raw", {}))
        payload.update(reduced_speaker)

    sounds = results.get("sounds")
    if sounds is None or sounds.get("status") != "ok":
        unavailable.append("sounds")
    else:
        payload.update(
            reduce_sound_events(
                sounds.get("raw", {}),
                duration_ms=duration_ms,
            )
        )

    payload["status"] = "complete" if not unavailable else "failed" if len(unavailable) == 2 else "partial"
    if unavailable:
        payload["unavailable"] = unavailable
    return payload
