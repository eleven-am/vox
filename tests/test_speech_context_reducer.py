from __future__ import annotations

import math
from typing import Any

import pytest

from vox.speech_context.audioset import enrich_audioset_classes
from vox.speech_context.reducer import (
    AUDIOSET_HUMAN_VOICE_ID,
    AUDIOSET_RESPIRATORY_SOUNDS_ID,
    SpeechContextReductionError,
    merge_context_chunks,
    offset_context_spans,
    reduce_sound_events,
    reduce_speaker_context,
    reduce_speech_context,
    summarize_sound_scores,
)


def _classes(*items: tuple[str, str, list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "id": class_id,
            "label": label,
            "ancestor_ids": ancestors,
        }
        for index, (class_id, label, ancestors) in enumerate(items)
    ]


def _frame(start_ms: float, *values: float) -> dict[str, Any]:
    return {
        "start_ms": start_ms,
        "end_ms": start_ms + 960.0,
        "values": list(values),
    }


def test_speaker_reduction_omits_unknown_and_merges_overlapping_model_windows() -> None:
    reduced = reduce_speaker_context(
        {
            "windows": [
                {
                    "start_ms": 0,
                    "end_ms": 2500,
                    "emotion": "<|SURPRISED|>",
                    "event": "<|Speech|>",
                },
                {
                    "start_ms": 1000,
                    "end_ms": 3500,
                    "emotion": "<|SURPRISED|>",
                    "event": "<|Laughter|>",
                },
                {
                    "start_ms": 2000,
                    "end_ms": 4500,
                    "emotion": "<|EMO_UNKNOWN|>",
                    "event": "<|Laughter|>",
                },
            ]
        }
    )

    assert reduced == {
        "emotions": [
            {"label": "surprised", "start_ms": 0, "end_ms": 3500},
        ],
        "vocal": [
            {"label": "laughter", "start_ms": 1000, "end_ms": 4500},
        ],
    }


def test_speaker_reduction_keeps_separate_occurrences_and_known_vocal_taxonomy() -> None:
    reduced = reduce_speaker_context(
        {
            "windows": [
                {"start_ms": 0, "end_ms": 1000, "emotion": "SAD", "event": "Cry"},
                {
                    "start_ms": 2000,
                    "end_ms": 3000,
                    "emotion": "HAPPY",
                    "event": "Cough",
                },
            ]
        }
    )

    assert reduced == {
        "emotions": [
            {"label": "sad", "start_ms": 0, "end_ms": 1000},
            {"label": "happy", "start_ms": 2000, "end_ms": 3000},
        ],
        "vocal": [
            {"label": "crying", "start_ms": 0, "end_ms": 1000},
            {"label": "coughing", "start_ms": 2000, "end_ms": 3000},
        ],
    }


@pytest.mark.parametrize(
    ("windows", "message"),
    [
        ([{"start_ms": 0, "end_ms": 0, "emotion": "SAD", "event": "Speech"}], "interval"),
        (
            [
                {"start_ms": 100, "end_ms": 200, "emotion": "SAD", "event": "Speech"},
                {"start_ms": 0, "end_ms": 100, "emotion": "SAD", "event": "Speech"},
            ],
            "time ordered",
        ),
        ([{"start_ms": 0, "end_ms": 100, "emotion": None, "event": "Speech"}], "emotion"),
    ],
)
def test_speaker_reduction_rejects_malformed_worker_output(
    windows: list[dict[str, Any]],
    message: str,
) -> None:
    with pytest.raises(SpeechContextReductionError, match=message):
        reduce_speaker_context({"windows": windows})


def test_sound_reduction_filters_voice_and_respiratory_taxonomies_without_allowlisting_sounds() -> None:
    raw = {
        "classes": _classes(
            ("/m/speech", "Speech", [AUDIOSET_HUMAN_VOICE_ID]),
            ("/m/cough", "Cough", [AUDIOSET_RESPIRATORY_SOUNDS_ID]),
            ("/m/dog", "Dog", ["/m/animal"]),
            ("/m/glass", "Breaking glass", ["/m/sounds"]),
        ),
        "scores": [
            _frame(0, 0.99, 0.98, 0.72, 0.64),
            _frame(480, 0.99, 0.98, 0.69, 0.61),
        ],
    }

    reduced = reduce_sound_events(raw, duration_ms=1440)

    assert reduced == {
        "sounds": [
            {"label": "breaking glass", "start_ms": 0, "end_ms": 1440, "score": 0.64},
            {"label": "dog", "start_ms": 0, "end_ms": 1440, "score": 0.72},
        ]
    }


def test_sound_reduction_keeps_non_vocal_human_sounds() -> None:
    raw = {
        "classes": _classes(
            ("/m/applause", "Applause", ["/m/human-group-actions"]),
            ("/m/footsteps", "Walk, footsteps", ["/m/human-locomotion"]),
        ),
        "scores": [_frame(0, 0.5, 0.4)],
    }

    assert reduce_sound_events(raw, duration_ms=960) == {
        "sounds": [
            {"label": "applause", "start_ms": 0, "end_ms": 960, "score": 0.5},
            {"label": "walk, footsteps", "start_ms": 0, "end_ms": 960, "score": 0.4},
        ]
    }


def test_sound_reduction_removes_explained_ancestor() -> None:
    raw = {
        "classes": _classes(
            ("/m/animal", "Animal", []),
            ("/m/dog", "Dog", ["/m/animal"]),
            ("/m/bark", "Bark", ["/m/animal", "/m/dog"]),
        ),
        "scores": [_frame(0, 0.7, 0.8, 0.9)],
    }

    assert reduce_sound_events(raw, duration_ms=960) == {
        "sounds": [{"label": "bark", "start_ms": 0, "end_ms": 960, "score": 0.9}]
    }


def test_sound_reduction_requires_repetition_or_one_strong_window() -> None:
    raw = {
        "classes": _classes(
            ("/m/repeated", "Repeated", []),
            ("/m/weak", "Weak", []),
            ("/m/strong", "Strong", []),
        ),
        "scores": [
            _frame(0, 0.06, 0.07, 0),
            _frame(480, 0.08, 0, 0.24),
        ],
    }

    assert reduce_sound_events(raw, duration_ms=1440) == {
        "sounds": [
            {"label": "repeated", "start_ms": 0, "end_ms": 1440, "score": 0.08},
            {"label": "strong", "start_ms": 480, "end_ms": 1440, "score": 0.24},
        ]
    }


def test_sound_diagnostic_preserves_pre_reduction_scores() -> None:
    raw = {
        "classes": _classes(
            ("/m/speech", "Speech", [AUDIOSET_HUMAN_VOICE_ID]),
            ("/m/cry", "Crying", [AUDIOSET_HUMAN_VOICE_ID]),
        ),
        "scores": [_frame(0, 0.6, 0.0412)],
    }

    assert summarize_sound_scores(raw) == {
        "frame_count": 1,
        "omitted_frame_count": 0,
        "frames": [
            {
                "start_ms": 0.0,
                "end_ms": 960.0,
                "candidates": [
                    {"label": "Speech", "score": 0.6},
                    {"label": "Crying", "score": 0.0412},
                ],
            }
        ],
        "class_maxima": [
            {"label": "Speech", "score": 0.6},
            {"label": "Crying", "score": 0.0412},
        ],
    }


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {
                "classes": _classes(("/m/one", "One", []), ("/m/two", "Two", [])),
                "scores": [_frame(0, 0.4)],
            },
            "vector length",
        ),
        (
            {
                "classes": _classes(("/m/one", "One", [])),
                "scores": [_frame(0, math.nan)],
            },
            "finite",
        ),
    ],
)
def test_sound_reduction_rejects_malformed_model_output(
    raw: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(SpeechContextReductionError, match=message):
        reduce_sound_events(raw, duration_ms=1000)


def test_offset_and_chunk_merge_keep_public_shape_compact() -> None:
    first = {"sounds": [{"label": "dog", "start_ms": 0, "end_ms": 960, "score": 0.6}]}
    second = offset_context_spans(
        {"sounds": [{"label": "dog", "start_ms": 0, "end_ms": 960, "score": 0.8}]},
        offset_ms=900,
    )

    assert merge_context_chunks([first, second], fields=("sounds",)) == {
        "sounds": [{"label": "dog", "start_ms": 0, "end_ms": 1860, "score": 0.8}]
    }


def test_combined_reduction_reports_partial_tracks_without_hiding_success() -> None:
    result = reduce_speech_context(
        {
            "speaker": {
                "status": "ok",
                "raw": {
                    "windows": [
                        {
                            "start_ms": 0,
                            "end_ms": 1000,
                            "emotion": "HAPPY",
                            "event": "Speech",
                        }
                    ]
                },
            },
            "sounds": {"status": "failed", "error": {"message": "offline"}},
        },
        duration_ms=1000,
    )

    assert result == {
        "schema_version": 2,
        "status": "partial",
        "emotions": [{"label": "happy", "start_ms": 0, "end_ms": 1000}],
        "vocal": [],
        "unavailable": ["sounds"],
    }


def test_audioset_enrichment_adds_transitive_ancestors() -> None:
    classes = [
        {"index": 0, "id": "/m/animal", "label": "Animal"},
        {"index": 1, "id": "/m/dog", "label": "Dog"},
        {"index": 2, "id": "/m/bark", "label": "Bark"},
    ]
    ontology = [
        {"id": "/m/animal", "child_ids": ["/m/dog"]},
        {"id": "/m/dog", "child_ids": ["/m/bark"]},
        {"id": "/m/bark", "child_ids": []},
    ]

    assert enrich_audioset_classes(classes, ontology)[2]["ancestor_ids"] == [
        "/m/animal",
        "/m/dog",
    ]
