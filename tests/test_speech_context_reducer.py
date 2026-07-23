from __future__ import annotations

import json
import math
from typing import Any

import pytest

from vox.speech_context.audioset import enrich_audioset_classes
from vox.speech_context.reducer import (
    SpeechContextReductionError,
    reduce_audio_events,
    reduce_prosody,
    reduce_speech_context,
    summarize_audio_event_scores,
)

PROSODY_COLUMNS = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness_sma3_amean",
    "loudness_sma3_pctlrange0-2",
    "loudnessPeaksPerSec",
    "HNRdBACF_sma3nz_amean",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "spectralFluxV_sma3nz_amean",
    "VoicedSegmentsPerSec",
    "MeanVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLength",
]


def _classes(*labels: str) -> list[dict[str, Any]]:
    return [{"index": index, "id": f"/m/test-{index}", "label": label} for index, label in enumerate(labels)]


def _score_frame(start_ms: float, *values: float) -> dict[str, Any]:
    return {
        "start_ms": start_ms,
        "end_ms": start_ms + 960.0,
        "values": list(values),
    }


def _prosody_raw(columns: list[str], values: list[float | str]) -> dict[str, Any]:
    return {
        "low_level_descriptors": {
            "columns": ["unused"],
            "frames": [{"start_ms": 0.0, "end_ms": 20.0, "values": [1.0]}],
        },
        "functionals": {
            "columns": columns,
            "frames": [{"start_ms": 0.0, "end_ms": 2_000.0, "values": values}],
        },
    }


def _result(raw: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "raw": raw}


def test_audio_event_reduction_keeps_arbitrary_simultaneous_classes() -> None:
    raw = {
        "classes": _classes("Speech", "Bark", "Child speech", "Breaking glass"),
        "scores": [
            _score_frame(0.0, 0.82, 0.31, 0.16, 0.01),
            _score_frame(480.0, 0.75, 0.32, 0.14, 0.22),
        ],
        "embeddings": [{"values": [0.1] * 1_024}],
        "log_mel_spectrogram": [{"values": [-1.0] * 64}],
    }

    reduced = reduce_audio_events(raw, duration_ms=2_000.0)

    assert [event["label"] for event in reduced["candidates"]] == [
        "Speech",
        "Bark",
        "Breaking glass",
    ]
    assert reduced["candidates"][1] == {
        "label": "Bark",
        "spans": [[0, 1440, 0.32]],
    }
    assert "embeddings" not in reduced
    assert "log_mel_spectrogram" not in reduced


def test_audio_event_reduction_does_not_special_case_child_speech() -> None:
    raw = {
        "classes": _classes("Child speech", "Unrelated"),
        "scores": [
            _score_frame(0.0, 0.12, 0.01),
            _score_frame(480.0, 0.14, 0.01),
        ],
    }

    reduced = reduce_audio_events(raw, duration_ms=1_440.0)

    assert [event["label"] for event in reduced["candidates"]] == ["Child speech"]


def test_audio_event_reduction_keeps_only_three_strongest_classes_per_window() -> None:
    raw = {
        "classes": _classes("First", "Second", "Third", "Fourth", "Fifth"),
        "scores": [_score_frame(0.0, 0.9, 0.8, 0.7, 0.6, 0.5)],
    }

    reduced = reduce_audio_events(raw, duration_ms=960.0)

    assert [event["label"] for event in reduced["candidates"]] == [
        "First",
        "Second",
        "Third",
    ]


def test_audio_event_diagnostic_preserves_pre_reducer_rankings_and_weak_scores() -> None:
    raw = {
        "classes": _classes("Speech", "Crying", "Sneeze", "Throat clearing"),
        "scores": [
            _score_frame(0.0, 0.6, 0.0494, 0.55, 0.4),
            _score_frame(480.0, 0.7, 0.1932, 0.2, 0.1),
        ],
    }

    diagnostic = summarize_audio_event_scores(raw)

    assert diagnostic == {
        "frame_count": 2,
        "omitted_frame_count": 0,
        "frames": [
            {
                "start_ms": 0.0,
                "end_ms": 960.0,
                "candidates": [
                    {"label": "Speech", "score": 0.6},
                    {"label": "Sneeze", "score": 0.55},
                    {"label": "Throat clearing", "score": 0.4},
                    {"label": "Crying", "score": 0.0494},
                ],
            },
            {
                "start_ms": 480.0,
                "end_ms": 1440.0,
                "candidates": [
                    {"label": "Speech", "score": 0.7},
                    {"label": "Sneeze", "score": 0.2},
                    {"label": "Crying", "score": 0.1932},
                    {"label": "Throat clearing", "score": 0.1},
                ],
            },
        ],
        "class_maxima": [
            {"label": "Speech", "score": 0.7},
            {"label": "Sneeze", "score": 0.55},
            {"label": "Throat clearing", "score": 0.4},
            {"label": "Crying", "score": 0.1932},
        ],
    }


def test_audio_event_reduction_removes_explained_ancestors_but_keeps_unrelated_events() -> None:
    raw = {
        "classes": [
            {
                "index": 0,
                "id": "/m/speech",
                "label": "Speech",
                "ancestor_ids": [],
            },
            {
                "index": 1,
                "id": "/m/dog",
                "label": "Dog",
                "ancestor_ids": ["/m/animal"],
            },
            {
                "index": 2,
                "id": "/m/bark",
                "label": "Bark",
                "ancestor_ids": ["/m/animal", "/m/dog"],
            },
        ],
        "scores": [_score_frame(0.0, 0.95, 0.72, 0.81)],
    }

    reduced = reduce_audio_events(raw, duration_ms=960.0)

    assert [event["label"] for event in reduced["candidates"]] == [
        "Speech",
        "Bark",
    ]


def test_audio_event_reduction_keeps_strong_transient_laughter() -> None:
    raw = {
        "classes": _classes("Laughter", "Speech"),
        "scores": [_score_frame(0.0, 0.222, 0.9)],
    }

    reduced = reduce_audio_events(raw, duration_ms=960.0)

    assert [event["label"] for event in reduced["candidates"]] == [
        "Speech",
        "Laughter",
    ]


def test_audio_event_reduction_groups_disjoint_spans_and_omits_internal_ids() -> None:
    raw = {
        "classes": _classes("Laughter"),
        "scores": [
            _score_frame(0.0, 0.246),
            _score_frame(2_000.0, 0.314),
        ],
    }

    reduced = reduce_audio_events(raw, duration_ms=3_000.0)

    assert reduced == {
        "candidates": [
            {
                "label": "Laughter",
                "spans": [
                    [0, 960, 0.25],
                    [2000, 2960, 0.31],
                ],
            }
        ]
    }


def test_audio_event_reduction_merges_supported_windows_and_drops_weak_isolation() -> None:
    raw = {
        "classes": _classes("Repeated", "Weak", "Strong"),
        "scores": [
            _score_frame(0.0, 0.06, 0.06, 0.0),
            _score_frame(480.0, 0.09, 0.0, 0.0),
            _score_frame(960.0, 0.07, 0.0, 0.24),
        ],
    }

    reduced = reduce_audio_events(raw, duration_ms=1_700.0)

    assert reduced == {
        "candidates": [
            {
                "label": "Repeated",
                "spans": [[0, 1700, 0.09]],
            },
            {
                "label": "Strong",
                "spans": [[960, 1700, 0.24]],
            },
        ]
    }


@pytest.mark.parametrize(
    "raw, message",
    [
        (
            {"classes": _classes("One", "Two"), "scores": [_score_frame(0.0, 0.4)]},
            "score vector length",
        ),
        (
            {"classes": _classes("One"), "scores": [_score_frame(0.0, math.nan)]},
            "finite",
        ),
        (
            {
                "classes": [{"index": 1, "id": "/m/wrong", "label": "Wrong"}],
                "scores": [_score_frame(0.0, 0.4)],
            },
            "class index",
        ),
    ],
)
def test_audio_event_reduction_rejects_malformed_model_output(
    raw: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(SpeechContextReductionError, match=message):
        reduce_audio_events(raw, duration_ms=1_000.0)


def test_prosody_reduction_emits_fourteen_conversational_measurements() -> None:
    columns = [*PROSODY_COLUMNS, *(f"unused_{index}" for index in range(74))]
    values: list[float | str] = [index + 0.123456 for index in range(88)]

    reduced = reduce_prosody(_prosody_raw(columns, values))

    assert reduced == {
        "pitch": {
            "mean_st": 0.123,
            "median_st": 1.123,
            "range_st": 2.123,
            "variation": 3.123,
        },
        "energy": {
            "mean": 4.123,
            "range": 5.123,
            "peaks_per_second": 6.123,
        },
        "voice_quality": {
            "hnr_db": 7.123,
            "jitter": 8.123,
            "shimmer_db": 9.123,
        },
        "spectral_variation": 10.123,
        "delivery": {
            "voiced_segments_per_second": 11.123,
            "mean_voiced_ms": 12123.456,
            "mean_unvoiced_ms": 13123.456,
        },
    }
    assert "low_level_descriptors" not in reduced
    assert "frames" not in reduced


def test_prosody_reduction_rejects_duplicate_columns() -> None:
    with pytest.raises(SpeechContextReductionError, match="unique"):
        reduce_prosody(_prosody_raw(["pitch", "pitch"], [1.0, 2.0]))


def test_prosody_reduction_rejects_missing_required_functional() -> None:
    with pytest.raises(SpeechContextReductionError, match="required prosody functional"):
        reduce_prosody(
            _prosody_raw(
                PROSODY_COLUMNS[:-1],
                [float(index) for index in range(len(PROSODY_COLUMNS) - 1)],
            )
        )


def test_speech_context_reduction_is_deterministic_partial_and_compact() -> None:
    classes = _classes("Speech", "Bark", *(f"Class {index}" for index in range(519)))
    scores = [
        _score_frame(
            frame * 480.0,
            *(0.8 if index == 0 else 0.12 if index == 1 else 0.001 for index in range(521)),
        )
        for frame in range(20)
    ]
    audio_events = {
        "classes": classes,
        "scores": scores,
        "embeddings": [{"start_ms": frame * 480.0, "values": [0.1] * 1_024} for frame in range(20)],
        "log_mel_spectrogram": [{"start_ms": frame * 10.0, "values": [-1.0] * 64} for frame in range(200)],
    }
    prosody = _prosody_raw(
        [*PROSODY_COLUMNS, *(f"unused_{index}" for index in range(74))],
        [float(index) for index in range(88)],
    )
    results = {
        "transcription": {"status": "failed", "error": {"message": "offline"}},
        "prosody": _result(prosody),
        "audio_events": _result(audio_events),
    }

    first = reduce_speech_context(results, duration_ms=10_000.0)
    second = reduce_speech_context(results, duration_ms=10_000.0)

    assert first == second
    assert first["status"] == "complete"
    assert "transcription" not in first
    assert "unavailable" not in first
    assert len(json.dumps(first["prosody"])) < 500
    assert [event["label"] for event in first["audio_events"]["candidates"]] == [
        "Speech",
        "Bark",
    ]
    raw_bytes = len(json.dumps({"prosody": prosody, "audio_events": audio_events}))
    compact_bytes = len(json.dumps(first))
    assert compact_bytes < raw_bytes * 0.01

    partial = reduce_speech_context(
        {
            "prosody": {"status": "failed", "error": {"message": "unavailable"}},
            "audio_events": _result(audio_events),
        },
        duration_ms=10_000.0,
    )
    assert partial["status"] == "partial"
    assert partial["unavailable"] == ["prosody"]
    assert "prosody" not in partial
    assert "audio_events" in partial


def test_audioset_enrichment_adds_complete_transitive_ancestors() -> None:
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

    enriched = enrich_audioset_classes(classes, ontology)

    assert enriched == [
        {"index": 0, "id": "/m/animal", "label": "Animal", "ancestor_ids": []},
        {
            "index": 1,
            "id": "/m/dog",
            "label": "Dog",
            "ancestor_ids": ["/m/animal"],
        },
        {
            "index": 2,
            "id": "/m/bark",
            "label": "Bark",
            "ancestor_ids": ["/m/animal", "/m/dog"],
        },
    ]


def test_audioset_enrichment_rejects_missing_class_and_cycles() -> None:
    with pytest.raises(SpeechContextReductionError, match="missing from the AudioSet ontology"):
        enrich_audioset_classes(
            [{"index": 0, "id": "/m/missing", "label": "Missing"}],
            [{"id": "/m/root", "child_ids": []}],
        )

    with pytest.raises(SpeechContextReductionError, match="cycle"):
        enrich_audioset_classes(
            [{"index": 0, "id": "/m/one", "label": "One"}],
            [
                {"id": "/m/one", "child_ids": ["/m/two"]},
                {"id": "/m/two", "child_ids": ["/m/one"]},
            ],
        )
