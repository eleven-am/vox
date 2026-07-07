#!/usr/bin/env python3
"""Smoke expressive TTS models through an already-running Vox HTTP server.

This script intentionally does not call Kubernetes, create PVCs, pull models,
or mutate adapter/model storage. It is for validating the server that is
already serving requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_SHORT_TEXT = "This is a short expressive adapter smoke test."
DEFAULT_LONG_TEXT = (
    "This is a longer expressive adapter smoke test. It should produce usable speech, "
    "keep a stable voice, and avoid silence or obvious truncation while the existing "
    "Vox server handles the request."
)
FAILURE_CLASSES = ("none", "Vox", "adapter", "dependency", "upstream", "hardware")
EVIDENCE_SCHEMA_VERSION = 1
NOT_CLEAN_PULL_BLOCKER = (
    "existing-server smoke cannot prove a clean model pull or clean adapter runtime install"
)


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes
    elapsed_s: float


@dataclass(frozen=True)
class AudioStats:
    bytes: int
    sha256: str
    duration_s: float | None
    sample_rate: int | None
    channels: int | None
    sample_width: int | None
    peak: float | None
    rms: float | None
    silent: bool


@dataclass(frozen=True)
class SynthesisEvidence:
    name: str
    text_chars: int
    status: int
    elapsed_s: float
    content_type: str
    output_path: str
    audio: AudioStats
    memory_samples: dict[str, Any] | None = None
    error: str | None = None


def _parse_json_object(value: str, *, field_name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{field_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{field_name} must be a JSON object")
    return parsed


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"x-api-key": api_key}


def _request_json(url: str, *, timeout: float, api_key: str | None) -> HttpResult:
    req = urllib.request.Request(url, headers=_auth_headers(api_key), method="GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.status
            headers = dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
        headers = dict(exc.headers.items())
    return HttpResult(status=status, headers=headers, body=body, elapsed_s=time.perf_counter() - started)


def _post_json(url: str, payload: dict[str, Any], *, timeout: float, api_key: str | None) -> HttpResult:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **_auth_headers(api_key)},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read()
            status = resp.status
            headers = dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        response_body = exc.read()
        status = exc.code
        headers = dict(exc.headers.items())
    return HttpResult(status=status, headers=headers, body=response_body, elapsed_s=time.perf_counter() - started)


def _response_evidence(result: HttpResult, *, max_chars: int = 12_000) -> dict[str, Any]:
    raw_text = result.body.decode("utf-8", errors="replace")
    payload: dict[str, Any] = {
        "status": result.status,
        "elapsed_s": result.elapsed_s,
    }

    if raw_text:
        try:
            payload["json"] = json.loads(raw_text)
        except json.JSONDecodeError:
            payload["text"] = raw_text[:max_chars]
            if len(raw_text) > max_chars:
                payload["truncated"] = True

    return payload


def _walk_numbers(value: Any, keys: set[str]) -> list[float]:
    matches: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, int | float):
                matches.append(float(child))
            matches.extend(_walk_numbers(child, keys))
    elif isinstance(value, list):
        for child in value:
            matches.extend(_walk_numbers(child, keys))
    return matches


def _peak_from_samples(samples: list[dict[str, Any]], keys: set[str]) -> float | None:
    values: list[float] = []
    for sample in samples:
        values.extend(_walk_numbers(sample.get("json", {}), keys))
    return max(values) if values else None


def _memory_sample_summary(samples: list[dict[str, Any]], *, interval_s: float) -> dict[str, Any]:
    return {
        "interval_s": interval_s,
        "count": len(samples),
        "samples": samples,
        "peak_ram_used_bytes": _peak_from_samples(samples, {"used_bytes", "ram_used_bytes"}),
        "peak_gpu_memory_used_mib": _peak_from_samples(
            samples,
            {"memory_used_mib", "gpu_memory_used_mib", "used_mib"},
        ),
    }


def _collect_memory_sample(
    *,
    samples: list[dict[str, Any]],
    base_url: str,
    timeout: float,
    api_key: str | None,
) -> None:
    memory_url = f"{base_url.rstrip('/')}/v1/system/memory"
    result = _request_json(memory_url, timeout=min(timeout, 30.0), api_key=api_key)
    samples.append(_response_evidence(result))


def _sample_memory_until(
    *,
    stop: threading.Event,
    samples: list[dict[str, Any]],
    base_url: str,
    timeout: float,
    api_key: str | None,
    interval_s: float,
) -> None:
    while not stop.is_set():
        _collect_memory_sample(samples=samples, base_url=base_url, timeout=timeout, api_key=api_key)
        if stop.wait(interval_s):
            break


def _pcm_stats(frames: bytes, *, sample_width: int) -> tuple[float | None, float | None]:
    if not frames or sample_width not in {1, 2, 4}:
        return None, None

    if sample_width == 1:
        values = [byte - 128 for byte in frames]
        normalizer = 128.0
    else:
        values = [
            int.from_bytes(frames[i:i + sample_width], byteorder="little", signed=True)
            for i in range(0, len(frames) - sample_width + 1, sample_width)
        ]
        normalizer = float(2 ** (8 * sample_width - 1))

    if not values:
        return None, None
    peak = max(abs(value) for value in values) / normalizer
    rms = math.sqrt(sum(value * value for value in values) / len(values)) / normalizer
    return peak, rms


def _audio_stats(path: Path) -> AudioStats:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    duration_s: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sample_width: int | None = None
    peak: float | None = None
    rms: float | None = None

    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames_count = wav.getnframes()
            duration_s = frames_count / sample_rate if sample_rate else None
            frames = wav.readframes(frames_count)
            peak, rms = _pcm_stats(frames, sample_width=sample_width)
    except (wave.Error, EOFError):
        pass

    silent = bool(rms is not None and rms < 0.0001)
    return AudioStats(
        bytes=len(data),
        sha256=digest,
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        peak=peak,
        rms=rms,
        silent=silent,
    )


def _speech_payload(
    *,
    model: str,
    text: str,
    voice: str | None,
    response_format: str,
    speed: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "response_format": response_format,
        "speed": speed,
    }
    if voice:
        payload["voice"] = voice
    if params:
        payload["params"] = params
    return payload


def _run_case(
    *,
    name: str,
    base_url: str,
    model: str,
    text: str,
    voice: str | None,
    response_format: str,
    speed: float,
    params: dict[str, Any],
    timeout: float,
    api_key: str | None,
    output_dir: Path,
    memory_sample_interval: float,
) -> SynthesisEvidence:
    memory_samples: list[dict[str, Any]] = []
    stop_sampling = threading.Event()
    sampler: threading.Thread | None = None
    if memory_sample_interval > 0:
        _collect_memory_sample(samples=memory_samples, base_url=base_url, timeout=timeout, api_key=api_key)
        sampler = threading.Thread(
            target=_sample_memory_until,
            kwargs={
                "stop": stop_sampling,
                "samples": memory_samples,
                "base_url": base_url,
                "timeout": timeout,
                "api_key": api_key,
                "interval_s": memory_sample_interval,
            },
            daemon=True,
        )
        sampler.start()
    result = _post_json(
        f"{base_url.rstrip('/')}/v1/audio/speech",
        _speech_payload(
            model=model,
            text=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
            params=params,
        ),
        timeout=timeout,
        api_key=api_key,
    )
    if sampler is not None:
        stop_sampling.set()
        sampler.join(timeout=min(memory_sample_interval + 5.0, 30.0))
    suffix = response_format.lower().lstrip(".") or "wav"
    output_path = output_dir / f"{name}.{suffix}"
    output_path.write_bytes(result.body)
    content_type = result.headers.get("Content-Type", "")

    error: str | None = None
    if result.status >= 400:
        try:
            error = result.body.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            error = f"HTTP {result.status}"

    return SynthesisEvidence(
        name=name,
        text_chars=len(text),
        status=result.status,
        elapsed_s=result.elapsed_s,
        content_type=content_type,
        output_path=str(output_path),
        audio=_audio_stats(output_path),
        memory_samples=_memory_sample_summary(memory_samples, interval_s=memory_sample_interval),
        error=error,
    )


def _write_evidence(evidence: dict[str, Any], output_dir: Path) -> Path:
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"Evidence written to {evidence_path}", file=sys.stderr)
    return evidence_path


def _status_failed(evidence: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value = evidence.get(key)
        if isinstance(value, dict) and int(value.get("status", 0)) >= 400:
            return True
    return False


def _read_failure_reasons(evidence: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    reasons: list[str] = []
    for key in keys:
        value = evidence.get(key)
        if isinstance(value, dict) and int(value.get("status", 0)) >= 400:
            reasons.append(f"{key} returned HTTP {value['status']}")
    return reasons


def _synthesis_failure_reasons(cases: list[SynthesisEvidence], audio_usable: str) -> list[str]:
    reasons: list[str] = []
    for case in cases:
        if case.status >= 400:
            reasons.append(f"{case.name} synthesis returned HTTP {case.status}")
        if case.error:
            reasons.append(f"{case.name} synthesis error: {case.error}")
        if case.status >= 400:
            continue
        if case.audio.bytes <= 0:
            reasons.append(f"{case.name} synthesis returned empty audio")
        elif case.audio.duration_s is None:
            reasons.append(f"{case.name} synthesis has no readable WAV duration")
        elif case.audio.duration_s <= 0:
            reasons.append(f"{case.name} synthesis has non-positive duration")
        if case.audio.silent:
            reasons.append(f"{case.name} synthesis returned silent audio")
    durations = {case.name: case.audio.duration_s for case in cases if case.audio.bytes > 0}
    short_duration = durations.get("short")
    long_duration = durations.get("long")
    if short_duration is not None and long_duration is not None and long_duration < short_duration:
        reasons.append(
            f"long synthesis duration {long_duration:.3f}s is shorter than short synthesis duration "
            f"{short_duration:.3f}s"
        )
    if audio_usable != "yes":
        reasons.append(f"manual audio usability verdict is {audio_usable}")
    return reasons


def _failure_classification_reasons(
    *,
    failure_reasons: list[str],
    failure_class: str,
    failure_note: str,
) -> list[str]:
    if failure_reasons and failure_class == "none":
        return [
            "failing smoke run must set --failure-class to one of "
            "Vox, adapter, dependency, upstream, or hardware"
        ]
    if failure_reasons and failure_class != "none" and not failure_note.strip():
        return ["classified failing smoke run must include --failure-note"]
    if not failure_reasons and failure_class != "none":
        return ["passing smoke run must use --failure-class none"]
    if not failure_reasons and failure_note.strip():
        return ["passing smoke run must not set --failure-note"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Existing Vox HTTP base URL")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VOX_API_KEY", ""),
        help="Vox API key. Defaults to VOX_API_KEY. The value is not written to evidence.",
    )
    parser.add_argument("--model", required=True, help="Model reference already available to the server")
    parser.add_argument("--voice", default=None, help="Voice id or server-side voice/reference path")
    parser.add_argument("--params-json", default="{}", help="Synthesis params JSON object")
    parser.add_argument("--short-text", default=DEFAULT_SHORT_TEXT)
    parser.add_argument("--long-text", default=DEFAULT_LONG_TEXT)
    parser.add_argument("--response-format", default="wav")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--memory-sample-interval",
        type=float,
        default=1.0,
        help=(
            "Seconds between /v1/system/memory samples during each synthesis request. "
            "Use 0 to disable per-request memory sampling."
        ),
    )
    parser.add_argument("--output-dir", default="/tmp/vox-served-smoke")
    parser.add_argument("--audio-usable", choices=("yes", "no", "unchecked"), default="unchecked")
    parser.add_argument(
        "--failure-class",
        choices=FAILURE_CLASSES,
        default="none",
        help=(
            "Failure owner for non-passing evidence. Use none only for passing runs; "
            "otherwise choose Vox, adapter, dependency, upstream, or hardware."
        ),
    )
    parser.add_argument(
        "--failure-note",
        default="",
        help="Required with --failure-class on failing runs. Describe the concrete cause or next fix.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only collect read-only endpoint evidence; do not call /v1/audio/speech.",
    )
    args = parser.parse_args()

    params = _parse_json_object(args.params_json, field_name="--params-json")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence: dict[str, Any] = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "mode": "existing-server",
        "clean_pull_proof": False,
        "clean_pull_blockers": [NOT_CLEAN_PULL_BLOCKER],
        "base_url": args.base_url,
        "model": args.model,
        "voice": args.voice or None,
        "response_format": args.response_format,
        "speed": args.speed,
        "params": params,
        "api_key_provided": bool(args.api_key),
        "audio_usable": args.audio_usable,
        "memory_sample_interval_s": args.memory_sample_interval,
        "failure_class": args.failure_class,
        "failure_note": args.failure_note,
        "inspect_only": args.inspect_only,
        "output_dir": str(output_dir),
    }

    health = _request_json(
        f"{args.base_url.rstrip('/')}/v1/health",
        timeout=min(args.timeout, 30.0),
        api_key=args.api_key,
    )
    models = _request_json(
        f"{args.base_url.rstrip('/')}/v1/models",
        timeout=min(args.timeout, 30.0),
        api_key=args.api_key,
    )
    model_detail = _request_json(
        f"{args.base_url.rstrip('/')}/v1/models/{urllib.parse.quote(args.model, safe='')}",
        timeout=min(args.timeout, 30.0),
        api_key=args.api_key,
    )
    loaded = _request_json(
        f"{args.base_url.rstrip('/')}/v1/models/loaded",
        timeout=min(args.timeout, 30.0),
        api_key=args.api_key,
    )
    memory = _request_json(
        f"{args.base_url.rstrip('/')}/v1/system/memory",
        timeout=min(args.timeout, 30.0),
        api_key=args.api_key,
    )
    evidence["health"] = _response_evidence(health)
    evidence["models"] = _response_evidence(models)
    evidence["model_detail"] = _response_evidence(model_detail)
    evidence["loaded_before"] = _response_evidence(loaded)
    evidence["memory_before"] = _response_evidence(memory)

    if args.inspect_only:
        evidence["synthesis"] = []
        evidence["synthesis_skipped"] = "inspect_only"
        failure_reasons = _read_failure_reasons(
            evidence,
            ("health", "models", "model_detail", "loaded_before", "memory_before"),
        )
        evidence["failure_reasons"] = [
            *failure_reasons,
            *_failure_classification_reasons(
                failure_reasons=failure_reasons,
                failure_class=args.failure_class,
                failure_note=args.failure_note,
            ),
        ]
        _write_evidence(evidence, output_dir)
        return 1 if evidence["failure_reasons"] else 0

    cases = [
        _run_case(
            name="short",
            base_url=args.base_url,
            model=args.model,
            text=args.short_text,
            voice=args.voice,
            response_format=args.response_format,
            speed=args.speed,
            params=params,
            timeout=args.timeout,
            api_key=args.api_key,
            output_dir=output_dir,
            memory_sample_interval=args.memory_sample_interval,
        ),
        _run_case(
            name="long",
            base_url=args.base_url,
            model=args.model,
            text=args.long_text,
            voice=args.voice,
            response_format=args.response_format,
            speed=args.speed,
            params=params,
            timeout=args.timeout,
            api_key=args.api_key,
            output_dir=output_dir,
            memory_sample_interval=args.memory_sample_interval,
        ),
    ]
    evidence["synthesis"] = [asdict(case) for case in cases]

    loaded_after = _request_json(
        f"{args.base_url.rstrip('/')}/v1/models/loaded",
        timeout=min(args.timeout, 30.0),
        api_key=args.api_key,
    )
    memory_after = _request_json(
        f"{args.base_url.rstrip('/')}/v1/system/memory",
        timeout=min(args.timeout, 30.0),
        api_key=args.api_key,
    )
    evidence["loaded_after"] = _response_evidence(loaded_after)
    evidence["memory_after"] = _response_evidence(memory_after)
    failure_reasons = [
        *_read_failure_reasons(
            evidence,
            (
                "health",
                "models",
                "model_detail",
                "loaded_before",
                "memory_before",
                "loaded_after",
                "memory_after",
            ),
        ),
        *_synthesis_failure_reasons(cases, args.audio_usable),
    ]
    evidence["failure_reasons"] = [
        *failure_reasons,
        *_failure_classification_reasons(
            failure_reasons=failure_reasons,
            failure_class=args.failure_class,
            failure_note=args.failure_note,
        ),
    ]

    _write_evidence(evidence, output_dir)

    return 1 if evidence["failure_reasons"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
