from __future__ import annotations

import asyncio
import hashlib
import inspect
import io
import json
import os
import wave
from argparse import ArgumentParser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import httpx
import numpy as np

from vox.audio.pipeline import prepare_for_stt
from vox.conversation.transcripts import EndpointCommitDelayPolicy, normalise_transcript_text
from vox.streaming.eou import EOU_MODEL_REVISION, ConversationTurn, EOUModel
from vox.streaming.types import TARGET_SAMPLE_RATE

CORPUS_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "endpointing_recorded.json"
EOU_THRESHOLD = 0.5
MAX_DELAY_MS = 3000
MIN_DELAY_MS = 350


@dataclass(frozen=True)
class EndpointingMetrics:
    false_endpoints: int
    continuation_count: int
    mean_terminal_latency_ms: float
    mean_complete_terminal_latency_ms: float
    terminal_count: int

    @property
    def false_endpoint_rate(self) -> float:
        return self.false_endpoints / self.continuation_count


@dataclass(frozen=True)
class VerifiedRecording:
    file: str
    path: Path
    sha256: str
    byte_count: int
    channels: int
    sample_rate_hz: int
    sample_width_bytes: int
    frame_count: int
    duration_ms: int
    decoded_samples: int

    def payload(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "audio": {
                "channels": self.channels,
                "sample_rate_hz": self.sample_rate_hz,
                "sample_width_bytes": self.sample_width_bytes,
                "frame_count": self.frame_count,
                "duration_ms": self.duration_ms,
                "decoded_samples": self.decoded_samples,
            },
        }


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_recordings(
    corpus: dict[str, Any],
    recordings_dir: Path,
) -> tuple[VerifiedRecording, ...]:
    sources = corpus.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("endpointing corpus has no sources")

    verified: list[VerifiedRecording] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("endpointing corpus source must be an object")
        file = str(source.get("file") or "")
        if not file or Path(file).name != file:
            raise ValueError(f"invalid endpointing recording filename {file!r}")
        if file in seen:
            raise ValueError(f"duplicate endpointing recording {file}")
        seen.add(file)

        path = recordings_dir / file
        if not path.is_file():
            raise FileNotFoundError(f"endpointing recording is missing: {path}")
        audio_bytes = path.read_bytes()
        actual_sha256 = hashlib.sha256(audio_bytes).hexdigest()
        expected_sha256 = str(source.get("sha256") or "")
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"endpointing recording SHA-256 mismatch for {file}: expected {expected_sha256}, got {actual_sha256}"
            )

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
                if wav.getcomptype() != "NONE":
                    raise ValueError(f"endpointing recording {file} is not uncompressed PCM WAV")
                channels = wav.getnchannels()
                sample_rate_hz = wav.getframerate()
                sample_width_bytes = wav.getsampwidth()
                frame_count = wav.getnframes()
                pcm_bytes = wav.readframes(frame_count)
        except wave.Error as exc:
            raise ValueError(f"endpointing recording {file} is not a readable WAV") from exc

        expected_pcm_bytes = frame_count * channels * sample_width_bytes
        if len(pcm_bytes) != expected_pcm_bytes:
            raise ValueError(
                f"endpointing recording {file} has truncated PCM data: "
                f"expected {expected_pcm_bytes} bytes, got {len(pcm_bytes)}"
            )

        duration_ms = int(frame_count / sample_rate_hz * 1000)
        actual_audio = {
            "channels": channels,
            "sample_rate_hz": sample_rate_hz,
            "sample_width_bytes": sample_width_bytes,
            "frame_count": frame_count,
            "duration_ms": duration_ms,
        }
        expected_audio = source.get("audio")
        if expected_audio != actual_audio:
            raise ValueError(
                f"endpointing recording audio properties mismatch for {file}: "
                f"expected {expected_audio}, got {actual_audio}"
            )

        decoded = prepare_for_stt(
            audio_bytes,
            target_rate=TARGET_SAMPLE_RATE,
            format_hint="wav",
        )
        if decoded.ndim != 1 or decoded.dtype != np.float32 or not np.isfinite(decoded).all():
            raise ValueError(f"endpointing recording production decode failed for {file}")
        decoded_samples = int(decoded.shape[0])
        if sample_rate_hz == TARGET_SAMPLE_RATE and decoded_samples != frame_count:
            raise ValueError(
                f"endpointing recording decoded sample count mismatch for {file}: "
                f"expected {frame_count}, got {decoded_samples}"
            )

        verified.append(
            VerifiedRecording(
                file=file,
                path=path,
                sha256=actual_sha256,
                byte_count=len(audio_bytes),
                channels=channels,
                sample_rate_hz=sample_rate_hz,
                sample_width_bytes=sample_width_bytes,
                frame_count=frame_count,
                duration_ms=duration_ms,
                decoded_samples=decoded_samples,
            )
        )

    source_names = {recording.file for recording in verified}
    for section in ("continuations", "terminals"):
        cases = corpus.get(section)
        if not isinstance(cases, list):
            raise ValueError(f"endpointing corpus {section} must be a list")
        unknown = {str(case.get("source") or "") for case in cases if str(case.get("source") or "") not in source_names}
        if unknown:
            raise ValueError(f"endpointing corpus {section} references unknown sources: {sorted(unknown)}")
    return tuple(verified)


def collect_runtime_evidence(
    corpus: dict[str, Any],
    recordings_dir: Path,
    *,
    transcribe: Callable[[VerifiedRecording], dict[str, Any]],
    score_eou: Callable[[str], float],
) -> dict[str, Any]:
    recordings = verify_recordings(corpus, recordings_dir)
    terminal_by_source = {str(case["source"]): case for case in corpus["terminals"]}
    runtime_recordings: list[dict[str, Any]] = []
    for recording in recordings:
        result = transcribe(recording)
        text = str(result.get("text") or "")
        expected_text = str(terminal_by_source[recording.file]["transcript"])
        runtime_recordings.append(
            {
                **recording.payload(),
                "stt": {
                    "model": str(result.get("model") or ""),
                    "text": text,
                    "processing_ms": int(result.get("processing_ms") or 0),
                    "matches_recorded_terminal": normalise_transcript_text(text)
                    == normalise_transcript_text(expected_text),
                },
            }
        )

    runtime_cases: list[dict[str, Any]] = []
    for section in ("continuations", "terminals"):
        for index, case in enumerate(corpus[section]):
            observed = float(score_eou(str(case["transcript"])))
            runtime_cases.append(
                {
                    "kind": section[:-1],
                    "index": index,
                    "source": str(case["source"]),
                    "transcript": str(case["transcript"]),
                    "recorded_eou_probability": float(case["eou_probability"]),
                    "observed_eou_probability": observed,
                    "absolute_eou_delta": abs(observed - float(case["eou_probability"])),
                }
            )

    corpus_sha256 = hashlib.sha256(json.dumps(corpus, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "corpus_sha256": corpus_sha256,
        "runtime": {
            "turn_detector": "livekit",
            "turn_detector_revision": EOU_MODEL_REVISION,
        },
        "recordings": runtime_recordings,
        "cases": runtime_cases,
    }


def runtime_evidence_matches(
    evidence: dict[str, Any],
    *,
    eou_tolerance: float,
) -> bool:
    transcripts_match = all(bool(recording["stt"]["matches_recorded_terminal"]) for recording in evidence["recordings"])
    eou_matches = all(float(case["absolute_eou_delta"]) <= eou_tolerance for case in evidence["cases"])
    return transcripts_match and eou_matches


def current_policy_delay_ms(
    eou_probability: float,
    recent_pause_ms: tuple[int, ...] = (),
) -> int:
    policy = EndpointCommitDelayPolicy(
        max_delay_ms=MAX_DELAY_MS,
        min_delay_ms=MIN_DELAY_MS,
        dynamic_endpointing=True,
    )
    return policy.commit_delay_ms(
        recent_pause_ms=recent_pause_ms,
        eou_probability=eou_probability,
        eou_threshold=EOU_THRESHOLD,
    )


def previous_policy_delay_ms(
    eou_probability: float,
    _recent_pause_ms: tuple[int, ...] = (),
) -> int:
    base_ms = 650
    if eou_probability < EOU_THRESHOLD:
        incompletion = (EOU_THRESHOLD - eou_probability) / EOU_THRESHOLD
        return round(base_ms + incompletion * 150)
    confidence = (eou_probability - EOU_THRESHOLD) / (1.0 - EOU_THRESHOLD)
    return int(base_ms - confidence * (base_ms - MIN_DELAY_MS))


def legacy_policy_delay_ms(
    eou_probability: float,
    _recent_pause_ms: tuple[int, ...] = (),
) -> int:
    base_ms = 1200
    if eou_probability < EOU_THRESHOLD:
        return base_ms
    confidence = (eou_probability - EOU_THRESHOLD) / (1.0 - EOU_THRESHOLD)
    return int(base_ms - confidence * (base_ms - MIN_DELAY_MS))


def evaluate_policy(
    corpus: dict[str, Any],
    delay_ms: Callable[[float, tuple[int, ...]], int],
) -> EndpointingMetrics:
    sources = {source["file"]: source for source in corpus["sources"]}
    pause_history: dict[str, list[int]] = {}
    false_endpoints = 0
    for case in corpus["continuations"]:
        source = str(case["source"])
        history = pause_history.setdefault(source, [])
        commit_ms = delay_ms(float(case["eou_probability"]), tuple(history))
        pause_ms = int(case["pause_ms"])
        false_endpoint = commit_ms < pause_ms
        false_endpoints += false_endpoint
        if not false_endpoint:
            history.append(pause_ms)

    terminal_latencies = [
        int(sources[case["source"]]["processing_ms"])
        + delay_ms(
            float(case["eou_probability"]),
            tuple(pause_history.get(str(case["source"]), ())),
        )
        for case in corpus["terminals"]
    ]
    complete_terminal_latencies = [
        latency
        for latency, case in zip(terminal_latencies, corpus["terminals"], strict=True)
        if float(case["eou_probability"]) >= EOU_THRESHOLD
    ]
    return EndpointingMetrics(
        false_endpoints=false_endpoints,
        continuation_count=len(corpus["continuations"]),
        mean_terminal_latency_ms=fmean(terminal_latencies),
        mean_complete_terminal_latency_ms=fmean(complete_terminal_latencies),
        terminal_count=len(terminal_latencies),
    )


def benchmark(path: Path = CORPUS_PATH) -> dict[str, EndpointingMetrics]:
    corpus = load_corpus(path)
    return {
        "v0.2.94": evaluate_policy(corpus, legacy_policy_delay_ms),
        "v0.2.123": evaluate_policy(corpus, previous_policy_delay_ms),
        "candidate": evaluate_policy(corpus, current_policy_delay_ms),
    }


def _print_policy_results(path: Path) -> None:
    results = benchmark(path)
    print("policy\tfalse_endpoints\trate\tmean_terminal_latency_ms\tmean_complete_terminal_latency_ms")
    for name, metrics in results.items():
        print(
            f"{name}\t{metrics.false_endpoints}/{metrics.continuation_count}"
            f"\t{metrics.false_endpoint_rate:.3f}\t{metrics.mean_terminal_latency_ms:.1f}"
            f"\t{metrics.mean_complete_terminal_latency_ms:.1f}"
        )


def _print_recording_results(corpus_path: Path, recordings_dir: Path) -> None:
    recordings = verify_recordings(load_corpus(corpus_path), recordings_dir)
    print(json.dumps({"recordings": [recording.payload() for recording in recordings]}, indent=2))


def _vox_transcriber(
    client: httpx.Client,
    *,
    vox_url: str,
    api_key: str | None,
    model: str,
    language: str,
) -> Callable[[VerifiedRecording], dict[str, Any]]:
    endpoint = f"{vox_url.rstrip('/')}/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def transcribe(recording: VerifiedRecording) -> dict[str, Any]:
        response = client.post(
            endpoint,
            headers=headers,
            files={"file": (recording.file, recording.path.read_bytes(), "audio/wav")},
            data={
                "model": model,
                "language": language,
                "response_format": "verbose_json",
                "timestamp_granularities": '["segment","word"]',
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Vox returned a non-object transcription for {recording.file}")
        return payload

    return transcribe


def _livekit_eou_scorer() -> Callable[[str], float]:
    detector = EOUModel()
    asyncio.run(detector.preload())

    async def resolve(prediction: Awaitable[Any]) -> float:
        return float(await prediction)

    def score(text: str) -> float:
        prediction = detector.predict(
            [ConversationTurn(role="user", content=text)],
            audio=np.array([], dtype=np.float32),
            sample_rate=TARGET_SAMPLE_RATE,
        )
        if inspect.isawaitable(prediction):
            return asyncio.run(resolve(prediction))
        return float(prediction)

    return score


def _runtime_results(args: Any) -> int:
    corpus = load_corpus(args.corpus)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    with httpx.Client(timeout=args.timeout_seconds) as client:
        evidence = collect_runtime_evidence(
            corpus,
            args.recordings_dir,
            transcribe=_vox_transcriber(
                client,
                vox_url=args.vox_url,
                api_key=api_key,
                model=args.model,
                language=args.language,
            ),
            score_eou=_livekit_eou_scorer(),
        )
    payload = json.dumps(evidence, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(f"{payload}\n")
    if args.verify and not runtime_evidence_matches(
        evidence,
        eou_tolerance=args.eou_tolerance,
    ):
        return 1
    return 0


def main() -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    policy = subparsers.add_parser("policy")
    policy.add_argument("--corpus", type=Path, default=CORPUS_PATH)

    recordings = subparsers.add_parser("verify-recordings")
    recordings.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    recordings.add_argument("--recordings-dir", type=Path, required=True)

    runtime = subparsers.add_parser("runtime")
    runtime.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    runtime.add_argument("--recordings-dir", type=Path, required=True)
    runtime.add_argument("--vox-url", required=True)
    runtime.add_argument("--api-key-env", default="VOX_API_KEY")
    runtime.add_argument("--model", default="parakeet-stt:tdt-0.6b-v3")
    runtime.add_argument("--language", default="en")
    runtime.add_argument("--timeout-seconds", type=float, default=300.0)
    runtime.add_argument("--eou-tolerance", type=float, default=0.01)
    runtime.add_argument("--verify", action="store_true")
    runtime.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "policy":
        _print_policy_results(args.corpus)
        return
    if args.command == "verify-recordings":
        _print_recording_results(args.corpus, args.recordings_dir)
        return
    raise SystemExit(_runtime_results(args))


if __name__ == "__main__":
    main()
