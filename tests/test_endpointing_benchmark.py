import hashlib
import io
import json
import wave
from pathlib import Path

import pytest

from scripts.benchmark_endpointing import (
    benchmark,
    collect_runtime_evidence,
    current_policy_delay_ms,
    evaluate_policy,
    load_corpus,
    previous_policy_delay_ms,
    runtime_evidence_matches,
    verify_recordings,
)


def _synthetic_recording(tmp_path: Path) -> tuple[dict, Path]:
    samples = b"\x01\x00" * 320
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples)
    audio_bytes = buffer.getvalue()
    path = tmp_path / "sample.wav"
    path.write_bytes(audio_bytes)
    corpus = {
        "sources": [
            {
                "file": path.name,
                "sha256": hashlib.sha256(audio_bytes).hexdigest(),
                "processing_ms": 17,
                "audio": {
                    "channels": 1,
                    "sample_rate_hz": 16_000,
                    "sample_width_bytes": 2,
                    "frame_count": 320,
                    "duration_ms": 20,
                },
            }
        ],
        "continuations": [],
        "terminals": [
            {
                "source": path.name,
                "transcript": "synthetic transcript",
                "eou_probability": 0.625,
            }
        ],
    }
    return corpus, tmp_path


def test_endpointing_corpus_spans_recorded_thinking_pauses():
    corpus = load_corpus()
    pauses = [case["pause_ms"] for case in corpus["continuations"]]

    assert len(corpus["sources"]) == 5
    assert len(corpus["continuations"]) == 8
    assert min(pauses) <= 500 + 100
    assert max(pauses) >= 2000
    assert all(len(source["sha256"]) == 64 for source in corpus["sources"])


def test_endpointing_recording_verifier_accepts_valid_pcm_wav(tmp_path: Path):
    corpus, recordings_dir = _synthetic_recording(tmp_path)
    recordings = verify_recordings(corpus, recordings_dir)

    assert len(recordings) == 1
    assert all(recording.sample_rate_hz == 16_000 for recording in recordings)
    assert all(recording.channels == 1 for recording in recordings)
    assert all(recording.sample_width_bytes == 2 for recording in recordings)
    assert all(recording.decoded_samples == recording.frame_count for recording in recordings)


def test_endpointing_corpus_rejects_missing_recording(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="speech-context-20260722T070950Z.wav"):
        verify_recordings(load_corpus(), tmp_path)


def test_endpointing_corpus_rejects_changed_recording_hash(tmp_path: Path):
    corpus, recordings_dir = _synthetic_recording(tmp_path)
    source = corpus["sources"][0]
    recording = recordings_dir / source["file"]
    recording.write_bytes(recording.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_recordings(corpus, recordings_dir)


def test_endpointing_corpus_rejects_changed_audio_properties(tmp_path: Path):
    corpus, recordings_dir = _synthetic_recording(tmp_path)
    corpus = json.loads(json.dumps(corpus))
    corpus["sources"][0]["audio"]["frame_count"] += 1

    with pytest.raises(ValueError, match="audio properties"):
        verify_recordings(corpus, recordings_dir)


def test_endpointing_corpus_rejects_malformed_wav_with_matching_hash(tmp_path: Path):
    audio_bytes = b"not a wave file"
    recording = tmp_path / "sample.wav"
    recording.write_bytes(audio_bytes)
    corpus = {
        "sources": [
            {
                "file": recording.name,
                "sha256": hashlib.sha256(audio_bytes).hexdigest(),
                "audio": {},
            }
        ],
        "continuations": [],
        "terminals": [],
    }

    with pytest.raises(ValueError, match="readable WAV"):
        verify_recordings(corpus, tmp_path)


def test_runtime_evidence_reads_audio_and_invokes_stt_and_eou(tmp_path: Path):
    corpus, recordings_dir = _synthetic_recording(tmp_path)
    transcribed: list[str] = []
    scored: list[str] = []

    def transcribe(recording):
        transcribed.append(recording.file)
        return {
            "text": f"transcript:{recording.file}",
            "processing_ms": 17,
            "model": "test-stt",
        }

    def score(text: str) -> float:
        scored.append(text)
        return 0.625

    evidence = collect_runtime_evidence(
        corpus,
        recordings_dir,
        transcribe=transcribe,
        score_eou=score,
    )

    assert transcribed == [source["file"] for source in corpus["sources"]]
    assert scored == ["synthetic transcript"]
    assert all(recording["stt"]["model"] == "test-stt" for recording in evidence["recordings"])
    assert all(case["observed_eou_probability"] == 0.625 for case in evidence["cases"])


def test_runtime_verification_rejects_transcript_and_eou_drift():
    evidence = {
        "recordings": [{"stt": {"matches_recorded_terminal": True}}],
        "cases": [{"absolute_eou_delta": 0.01}],
    }

    assert runtime_evidence_matches(evidence, eou_tolerance=0.01)

    evidence["recordings"][0]["stt"]["matches_recorded_terminal"] = False
    assert not runtime_evidence_matches(evidence, eou_tolerance=0.01)

    evidence["recordings"][0]["stt"]["matches_recorded_terminal"] = True
    evidence["cases"][0]["absolute_eou_delta"] = 0.011
    assert not runtime_evidence_matches(evidence, eou_tolerance=0.01)


def test_endpointing_candidate_matches_legacy_false_endpoint_rate_with_lower_latency():
    results = benchmark()

    assert results["v0.2.123"].false_endpoints == 8
    assert results["candidate"].false_endpoints == results["v0.2.94"].false_endpoints == 3
    assert (
        results["v0.2.94"].mean_complete_terminal_latency_ms - results["candidate"].mean_complete_terminal_latency_ms
        >= 150
    )
    assert results["v0.2.94"].mean_terminal_latency_ms - results["candidate"].mean_terminal_latency_ms >= 100


def test_endpointing_candidate_preserves_high_confidence_latency():
    high_confidence_scores = (0.85, 0.9, 1.0)

    assert all(current_policy_delay_ms(score) == previous_policy_delay_ms(score) for score in high_confidence_scores)


def test_endpointing_candidate_records_threshold_complete_tradeoff():
    results = benchmark()
    added_latency_ms = (
        results["candidate"].mean_complete_terminal_latency_ms - results["v0.2.123"].mean_complete_terminal_latency_ms
    )

    assert 200 <= added_latency_ms <= 210
    assert current_policy_delay_ms(0.70) > previous_policy_delay_ms(0.70)


def test_false_endpoint_does_not_contribute_future_pause_history():
    corpus = {
        "sources": [{"file": "sample.wav", "processing_ms": 0}],
        "continuations": [
            {"source": "sample.wav", "pause_ms": 1000, "eou_probability": 0.1},
            {"source": "sample.wav", "pause_ms": 200, "eou_probability": 0.1},
        ],
        "terminals": [{"source": "sample.wav", "eou_probability": 0.9}],
    }
    histories: list[tuple[int, ...]] = []

    def delay(_probability: float, history: tuple[int, ...]) -> int:
        histories.append(history)
        return 500

    metrics = evaluate_policy(corpus, delay)

    assert metrics.false_endpoints == 1
    assert histories == [(), (), (200,)]
