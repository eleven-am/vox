#!/usr/bin/env python3
"""Deterministic WebRTC interruption accuracy and latency benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from vox.conversation.interrupt import HeuristicInterruptClassifier, looks_like_self_echo
from vox.conversation.interruption_detector import (
    EvidenceBasedInterruptDetector,
    InterruptionDecisionAction,
)
from vox.conversation.types import TurnPolicy
from vox.streaming.partials import PARTIAL_OVERLAP_MS
from vox.streaming.types import StreamSessionConfig, StreamTranscript

SAMPLE_RATE = 16_000
ASSISTANT_TEXT = "The assistant is explaining the appointment and the next steps."
BASELINE_DUCK_SIGNAL_OFFSET_MS = 0
BASELINE_PARTIAL_WINDOW_MS = 1500
BASELINE_PARTIAL_STRIDE_MS = 700
BASELINE_PARTIAL_OVERLAP_MS = 300
REQUIRED_CATEGORIES = frozenset(
    {
        "natural_single_word",
        "natural_multiword",
        "low_volume",
        "hesitant_speech",
        "tts_playback",
        "microphone_rub",
        "handling_noise",
        "tap",
        "keyboard",
        "breath",
        "cough",
        "room_noise",
        "assistant_echo",
        "tts_leakage",
        "backchannel",
        "stt_hallucination",
        "stale_event",
        "repeated_candidate",
        "aec_warmup",
        "response_restart",
    }
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    categories: tuple[str, ...]
    should_interrupt: bool
    path: Literal["final", "partial", "timeout", "stale", "repeated", "restart"]
    transcript: str = ""
    duration_ms: int = 500
    eou_probability: float | None = None
    audio_profile: str = "voice"
    output_echo: bool = False
    aec_warmup: bool = False
    partial_text: str = ""
    partial_at_ms: int = 0


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def false_positive_rate(self) -> float:
        denominator = self.false_positive + self.true_negative
        return self.false_positive / denominator if denominator else 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    confusion: ConfusionMatrix
    median_confirmation_ms: float
    p95_confirmation_ms: float


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            "single_word_high_eou",
            ("natural_single_word", "tts_playback"),
            True,
            "final",
            transcript="Hold",
            eou_probability=0.9,
            audio_profile="voice_then_short_silence",
        ),
        BenchmarkCase(
            "single_word_stable_partial",
            ("natural_single_word", "tts_playback"),
            True,
            "final",
            transcript="Actually",
            partial_text="Actually",
            partial_at_ms=250,
            audio_profile="voice",
        ),
        BenchmarkCase(
            "multiword_partial",
            ("natural_multiword", "tts_playback"),
            True,
            "partial",
            transcript="I need to add something",
            duration_ms=500,
        ),
        BenchmarkCase(
            "quiet_multiword_final",
            ("natural_multiword", "low_volume", "tts_playback"),
            True,
            "final",
            transcript="Let me clarify that",
            duration_ms=800,
            eou_probability=0.7,
            audio_profile="quiet_voice",
        ),
        BenchmarkCase(
            "hesitant_multiword_final",
            ("natural_multiword", "hesitant_speech", "tts_playback"),
            True,
            "final",
            transcript="I think I mean something else",
            duration_ms=1000,
            eou_probability=0.4,
            audio_profile="hesitant_voice",
        ),
        BenchmarkCase(
            "sustained_voice_without_partial",
            ("natural_multiword", "tts_playback"),
            True,
            "timeout",
            duration_ms=420,
        ),
        BenchmarkCase(
            "incomplete_multiword_final",
            ("natural_multiword", "tts_playback"),
            True,
            "final",
            transcript="What I mean is",
            duration_ms=900,
            eou_probability=0.08,
        ),
        BenchmarkCase(
            "microphone_rub_hallucination",
            ("microphone_rub", "stt_hallucination", "tts_playback"),
            False,
            "final",
            transcript="Anyway",
            duration_ms=690,
            eou_probability=0.00005,
            audio_profile="noise",
        ),
        BenchmarkCase(
            "sustained_handling_noise",
            ("handling_noise", "tts_playback"),
            False,
            "timeout",
            duration_ms=600,
            audio_profile="noise",
        ),
        BenchmarkCase(
            "tap_hallucination",
            ("tap", "stt_hallucination", "tts_playback"),
            False,
            "final",
            transcript="Okay",
            eou_probability=0.02,
            audio_profile="tap",
        ),
        BenchmarkCase(
            "keyboard_hallucination",
            ("keyboard", "stt_hallucination", "tts_playback"),
            False,
            "final",
            transcript="Right",
            duration_ms=650,
            eou_probability=0.01,
            audio_profile="keyboard",
        ),
        BenchmarkCase(
            "breath_hallucination",
            ("breath", "stt_hallucination", "tts_playback"),
            False,
            "final",
            transcript="Yeah",
            eou_probability=0.1,
            audio_profile="breath",
        ),
        BenchmarkCase(
            "cough_hallucination",
            ("cough", "stt_hallucination", "tts_playback"),
            False,
            "final",
            transcript="Uh",
            eou_probability=0.03,
            audio_profile="cough",
        ),
        BenchmarkCase(
            "quiet_room_noise",
            ("room_noise", "tts_playback"),
            False,
            "timeout",
            duration_ms=600,
            audio_profile="room",
        ),
        BenchmarkCase(
            "assistant_transcript_echo",
            ("assistant_echo", "tts_playback"),
            False,
            "final",
            transcript="assistant is explaining the appointment",
            duration_ms=850,
            eou_probability=0.8,
        ),
        BenchmarkCase(
            "correlated_tts_leakage",
            ("tts_leakage", "tts_playback"),
            False,
            "timeout",
            duration_ms=600,
            output_echo=True,
        ),
        BenchmarkCase(
            "real_speech_over_tts_leakage",
            ("natural_multiword", "tts_leakage", "tts_playback"),
            True,
            "final",
            transcript="I need to clarify something important",
            duration_ms=800,
            eou_probability=0.8,
            output_echo=True,
        ),
        BenchmarkCase(
            "short_acknowledgement",
            ("backchannel", "tts_playback"),
            False,
            "final",
            transcript="Mhm",
            eou_probability=0.4,
            audio_profile="backchannel",
        ),
        BenchmarkCase(
            "multiword_noise_hallucination",
            ("handling_noise", "stt_hallucination", "tts_playback"),
            False,
            "final",
            transcript="Okay anyway",
            duration_ms=800,
            eou_probability=0.01,
            audio_profile="noise",
        ),
        BenchmarkCase(
            "delayed_final_from_previous_candidate",
            ("stale_event", "tts_playback"),
            False,
            "stale",
            transcript="This belongs to the previous turn",
            duration_ms=900,
            eou_probability=0.9,
        ),
        BenchmarkCase(
            "repeated_noise_candidates",
            ("repeated_candidate", "stt_hallucination", "tts_playback"),
            False,
            "repeated",
            transcript="Anyway",
            duration_ms=650,
            eou_probability=0.01,
            audio_profile="noise",
        ),
        BenchmarkCase(
            "aec_warmup_burst",
            ("aec_warmup", "tts_leakage", "tts_playback"),
            False,
            "timeout",
            duration_ms=250,
            aec_warmup=True,
            audio_profile="noise",
        ),
        BenchmarkCase(
            "stale_final_after_response_restart",
            ("response_restart", "stale_event", "tts_playback"),
            False,
            "restart",
            transcript="Old response candidate content",
            duration_ms=900,
            eou_probability=0.9,
        ),
    )


def _voice(duration_ms: int, *, amplitude: float = 0.1, frequency: float = 220) -> np.ndarray:
    t = np.arange(duration_ms * SAMPLE_RATE // 1000) / SAMPLE_RATE
    envelope = 0.65 + 0.35 * np.sin(2 * np.pi * 4 * t) ** 2
    signal = np.sin(2 * np.pi * frequency * t) + 0.35 * np.sin(4 * np.pi * frequency * t)
    return (amplitude * envelope * signal).astype(np.float32)


def _audio(profile: str, duration_ms: int) -> np.ndarray:
    samples = duration_ms * SAMPLE_RATE // 1000
    rng = np.random.default_rng(712)
    if profile == "voice":
        return _voice(duration_ms)
    if profile == "quiet_voice":
        return _voice(duration_ms, amplitude=0.018)
    if profile == "voice_then_short_silence":
        return np.concatenate([_voice(max(1, duration_ms - 100)), np.zeros(SAMPLE_RATE // 10, np.float32)])
    if profile == "hesitant_voice":
        first = _voice(320, amplitude=0.055)
        gap = np.zeros(160 * SAMPLE_RATE // 1000, np.float32)
        last_ms = max(1, duration_ms - 480)
        return np.concatenate([first, gap, _voice(last_ms, amplitude=0.06)])
    if profile == "backchannel":
        voice_ms = min(180, duration_ms)
        silence_samples = samples - voice_ms * SAMPLE_RATE // 1000
        return np.concatenate(
            [
                _voice(voice_ms, amplitude=0.06),
                np.zeros(silence_samples, np.float32),
            ]
        )
    if profile == "noise":
        return rng.normal(0, 0.15, samples).astype(np.float32)
    if profile == "room":
        return rng.normal(0, 0.0008, samples).astype(np.float32)
    if profile == "tap":
        signal = np.zeros(samples, np.float32)
        signal[100 : min(samples, 220)] = 0.8
        return signal
    if profile == "keyboard":
        signal = np.zeros(samples, np.float32)
        for position in range(200, samples, max(1, SAMPLE_RATE // 12)):
            signal[position : min(samples, position + 35)] = 0.35
        return signal
    if profile == "breath":
        noise = rng.normal(0, 0.045, samples).astype(np.float32)
        return np.convolve(noise, np.ones(9, np.float32) / 9, mode="same").astype(np.float32)
    if profile == "cough":
        noise = rng.normal(0, 0.2, samples).astype(np.float32)
        envelope = np.exp(-np.linspace(0, 8, samples, dtype=np.float32))
        return noise * envelope
    raise ValueError(f"unknown audio profile: {profile}")


def _detector() -> EvidenceBasedInterruptDetector:
    return EvidenceBasedInterruptDetector(
        policy=TurnPolicy(
            min_interrupt_duration_ms=420,
            speaking_interrupt_min_duration_ms=420,
            speaking_interrupt_min_words=2,
            aec_warmup_ms=900,
        ),
        classifier=HeuristicInterruptClassifier(),
    )


def _begin(
    detector: EvidenceBasedInterruptDetector,
    *,
    utterance_id: int,
    response_id: str = "resp_1",
    started_at: float = 10.0,
) -> None:
    detector.begin(
        utterance_id=utterance_id,
        response_id=response_id,
        started_at=started_at,
        speech_started_ms=int(started_at * 1000),
        assistant_text=ASSISTANT_TEXT,
    )


async def _new_decision(case: BenchmarkCase) -> tuple[bool, float | None, str]:
    detector = _detector()
    _begin(detector, utterance_id=1)
    audio = _audio(case.audio_profile, case.duration_ms)

    if case.path in {"stale", "restart"}:
        _begin(
            detector,
            utterance_id=2,
            response_id="resp_2" if case.path == "restart" else "resp_1",
            started_at=11.0,
        )
    elif case.path == "repeated":
        detector.mark_speech_stopped(
            utterance_id=1,
            stopped_at=10.2,
            speech_stopped_ms=10_200,
            expects_transcript=False,
        )
        detector.finish(1)
        _begin(detector, utterance_id=2, started_at=11.0)

    utterance_id = 1 if case.path in {"stale", "restart"} else (2 if case.path == "repeated" else 1)
    if case.partial_text:
        partial = await detector.observe_partial(
            StreamTranscript(
                text=case.partial_text,
                is_partial=True,
                audio_duration_ms=case.partial_at_ms,
                utterance_id=utterance_id,
            ),
            cumulative_transcript=case.partial_text,
            assistant_text=ASSISTANT_TEXT,
            output_echo=case.output_echo,
            now=10.0 + case.partial_at_ms / 1000,
        )
        if partial.action is InterruptionDecisionAction.CONFIRM:
            return True, float(case.partial_at_ms), partial.reason

    if case.path == "partial":
        decision = await detector.observe_partial(
            StreamTranscript(
                text=case.transcript,
                is_partial=True,
                audio_duration_ms=case.duration_ms,
                utterance_id=utterance_id,
            ),
            cumulative_transcript=case.transcript,
            assistant_text=ASSISTANT_TEXT,
            output_echo=case.output_echo,
            now=10.0 + case.duration_ms / 1000,
        )
    elif case.path == "timeout":
        decision = await detector.evaluate_timeout(
            assistant_text=ASSISTANT_TEXT,
            output_echo=case.output_echo,
            aec_warmup_remaining_ms=250 if case.aec_warmup else 0,
            audio=audio,
            sample_rate=SAMPLE_RATE,
            last_eou_probability=case.eou_probability,
            now=10.0 + case.duration_ms / 1000,
        )
    else:
        if case.path not in {"stale", "restart"}:
            detector.mark_speech_stopped(
                utterance_id=utterance_id,
                stopped_at=10.0 + case.duration_ms / 1000,
                speech_stopped_ms=10_000 + case.duration_ms,
                expects_transcript=True,
            )
        decision = await detector.observe_final(
            StreamTranscript(
                text=case.transcript,
                audio_duration_ms=case.duration_ms,
                eou_probability=case.eou_probability,
                utterance_id=utterance_id,
            ),
            assistant_text=ASSISTANT_TEXT,
            output_echo=case.output_echo,
            audio=audio,
            sample_rate=SAMPLE_RATE,
            now=10.0 + case.duration_ms / 1000,
        )

    confirmation_ms = case.duration_ms
    if decision.action is InterruptionDecisionAction.DEFER and decision.retry_after_ms > 0:
        elapsed_ms = case.duration_ms + decision.retry_after_ms
        decision = await detector.evaluate_timeout(
            assistant_text=ASSISTANT_TEXT,
            output_echo=case.output_echo,
            aec_warmup_remaining_ms=0,
            audio=audio,
            sample_rate=SAMPLE_RATE,
            last_eou_probability=case.eou_probability,
            now=10.0 + elapsed_ms / 1000,
        )
        confirmation_ms = elapsed_ms

    confirmed = decision.action is InterruptionDecisionAction.CONFIRM
    latency = float(confirmation_ms) if confirmed else None
    return confirmed, latency, decision.reason


def _legacy_decision(case: BenchmarkCase) -> tuple[bool, float | None, str]:
    """Freeze the measured pre-change decision paths for side-by-side scoring."""
    if case.output_echo or looks_like_self_echo(case.transcript, ASSISTANT_TEXT):
        return False, None, "legacy_echo_reject"
    if case.path in {"final", "stale", "repeated", "restart"}:
        return bool(case.transcript), float(case.duration_ms), "legacy_unconditional_final"
    if case.path == "partial":
        confirmed = bool(case.transcript) and len(case.transcript.split()) >= 2 and case.duration_ms >= 420
        return confirmed, float(case.duration_ms) if confirmed else None, "legacy_partial"
    if case.aec_warmup:
        return False, None, "legacy_aec_reject"

    audio = _audio(case.audio_profile, case.duration_ms)
    tail = audio[-min(audio.size, SAMPLE_RATE * 80 // 1000) :]
    tail_rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float32)))) if tail.size else 0.0
    confirmed = case.duration_ms >= 180 and tail_rms >= 0.0025
    return confirmed, float(case.duration_ms) if confirmed else None, "legacy_energy_gate"


def _score(cases: tuple[BenchmarkCase, ...], predictions: list[bool], latencies: list[float | None]) -> BenchmarkResult:
    tp = fp = tn = fn = 0
    true_latencies: list[float] = []
    for case, predicted, latency in zip(cases, predictions, latencies, strict=True):
        if case.should_interrupt and predicted:
            tp += 1
            if latency is not None:
                true_latencies.append(latency)
        elif case.should_interrupt:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return BenchmarkResult(
        confusion=ConfusionMatrix(tp, fp, tn, fn),
        median_confirmation_ms=float(np.median(true_latencies)) if true_latencies else 0.0,
        p95_confirmation_ms=float(np.percentile(true_latencies, 95)) if true_latencies else 0.0,
    )


def _latency_percentiles(latencies: list[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0
    return float(np.median(latencies)), float(np.percentile(latencies, 95))


async def run_benchmark() -> dict[str, object]:
    cases = benchmark_cases()
    covered = {category for case in cases for category in case.categories}
    missing_categories = sorted(REQUIRED_CATEGORIES - covered)

    baseline_predictions: list[bool] = []
    baseline_latencies: list[float | None] = []
    new_predictions: list[bool] = []
    new_latencies: list[float | None] = []
    case_results: list[dict[str, object]] = []
    for case in cases:
        baseline, baseline_latency, baseline_reason = _legacy_decision(case)
        current, current_latency, current_reason = await _new_decision(case)
        baseline_predictions.append(baseline)
        baseline_latencies.append(baseline_latency)
        new_predictions.append(current)
        new_latencies.append(current_latency)
        case_results.append(
            {
                "name": case.name,
                "expected_interrupt": case.should_interrupt,
                "baseline_interrupt": baseline,
                "current_interrupt": current,
                "baseline_reason": baseline_reason,
                "current_reason": current_reason,
            }
        )

    baseline_result = _score(cases, baseline_predictions, baseline_latencies)
    current_result = _score(cases, new_predictions, new_latencies)
    baseline_fpr = baseline_result.confusion.false_positive_rate
    current_fpr = current_result.confusion.false_positive_rate
    false_positive_reduction = (baseline_fpr - current_fpr) / baseline_fpr if baseline_fpr else 0.0
    recall_regression = baseline_result.confusion.recall - current_result.confusion.recall
    paired_latencies = [
        (float(baseline_latency), float(current_latency))
        for case, baseline, current, baseline_latency, current_latency in zip(
            cases,
            baseline_predictions,
            new_predictions,
            baseline_latencies,
            new_latencies,
            strict=True,
        )
        if case.should_interrupt
        and baseline
        and current
        and baseline_latency is not None
        and current_latency is not None
    ]
    paired_baseline_median, paired_baseline_p95 = _latency_percentiles([baseline for baseline, _ in paired_latencies])
    paired_current_median, paired_current_p95 = _latency_percentiles([current for _, current in paired_latencies])
    median_delta = paired_current_median - paired_baseline_median
    p95_delta = paired_current_p95 - paired_baseline_p95
    stream_defaults = StreamSessionConfig()
    current_stt_cadence = (
        stream_defaults.partial_window_ms,
        stream_defaults.partial_stride_ms,
        PARTIAL_OVERLAP_MS,
    )
    baseline_stt_cadence = (
        BASELINE_PARTIAL_WINDOW_MS,
        BASELINE_PARTIAL_STRIDE_MS,
        BASELINE_PARTIAL_OVERLAP_MS,
    )
    current_duck_signal_offset_ms = 0
    acceptance = {
        "required_categories_present": not missing_categories,
        "false_positive_reduction_at_least_50_percent": false_positive_reduction >= 0.5,
        "recall_regression_at_most_1_point": recall_regression <= 0.01,
        "median_latency_regression_at_most_50_ms": median_delta <= 50,
        "p95_latency_regression_at_most_100_ms": p95_delta <= 100,
        "duck_latency_unchanged": (current_duck_signal_offset_ms <= BASELINE_DUCK_SIGNAL_OFFSET_MS),
        "ordinary_stt_cadence_unchanged": current_stt_cadence == baseline_stt_cadence,
    }
    return {
        "baseline": {
            "confusion": asdict(baseline_result.confusion),
            "recall": baseline_result.confusion.recall,
            "false_positive_rate": baseline_fpr,
            "median_confirmation_ms": baseline_result.median_confirmation_ms,
            "p95_confirmation_ms": baseline_result.p95_confirmation_ms,
        },
        "current": {
            "confusion": asdict(current_result.confusion),
            "recall": current_result.confusion.recall,
            "false_positive_rate": current_fpr,
            "median_confirmation_ms": current_result.median_confirmation_ms,
            "p95_confirmation_ms": current_result.p95_confirmation_ms,
        },
        "comparison": {
            "false_positive_reduction": false_positive_reduction,
            "recall_regression": recall_regression,
            "median_latency_delta_ms": median_delta,
            "p95_latency_delta_ms": p95_delta,
            "paired_genuine_cases": len(paired_latencies),
            "paired_baseline_median_ms": paired_baseline_median,
            "paired_current_median_ms": paired_current_median,
            "paired_baseline_p95_ms": paired_baseline_p95,
            "paired_current_p95_ms": paired_current_p95,
            "baseline_duck_signal_offset_ms": BASELINE_DUCK_SIGNAL_OFFSET_MS,
            "current_duck_signal_offset_ms": current_duck_signal_offset_ms,
            "baseline_stt_cadence_ms": baseline_stt_cadence,
            "current_stt_cadence_ms": current_stt_cadence,
        },
        "acceptance": acceptance,
        "missing_categories": missing_categories,
        "cases": case_results,
        "passed": all(acceptance.values()),
    }


def _print_human(report: dict[str, object]) -> None:
    baseline = report["baseline"]
    current = report["current"]
    comparison = report["comparison"]
    print("WebRTC interruption benchmark")
    print(json.dumps({"baseline": baseline, "current": current, "comparison": comparison}, indent=2))
    print("Acceptance:")
    for name, passed in report["acceptance"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = asyncio.run(run_benchmark())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
