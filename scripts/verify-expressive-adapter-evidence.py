#!/usr/bin/env python3
"""Verify local expressive-adapter smoke evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA_VERSION = 2
REQUIRED_COMMANDS = (
    "resource snapshot before pull",
    "runtime snapshot",
    "download size estimate",
    "pre-pull clean state",
    "pull",
    "short synthesis",
    "long synthesis",
    "resource snapshot after smoke",
)
REQUIRED_RESOURCE_SAMPLE_LABELS = ("pull", "short synthesis", "long synthesis")
REQUIRED_GOAL_CHECKS = (
    "clean_runtime_and_model_store",
    "vox_pull",
    "adapter_runtime_and_model_artifacts",
    "adapter_package_version",
    "isolated_runtime_directory",
    "model_artifacts",
    "short_synthesis",
    "long_synthesis",
    "latency_recorded",
    "ram_vram_recorded",
    "manual_audio_usability",
    "failure_classification",
)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"could not read evidence: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "evidence root must be a JSON object"
    return payload, None


def _require_empty_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        return []
    if value == []:
        return []
    if isinstance(value, list):
        return [f"{key} must be empty for passing proof: {item}" for item in value]
    return [f"{key} must be an empty list for passing proof"]


def _verify_expected_state(payload: dict[str, Any]) -> list[str]:
    expected = payload.get("expected_state")
    if not isinstance(expected, dict):
        return ["expected_state must be an object"]

    blockers: list[str] = []
    for key in ("adapters", "runtime", "model_links"):
        value = expected.get(key)
        if not isinstance(value, list) or not value:
            blockers.append(f"expected_state.{key} must be a non-empty list")
    packages = expected.get("adapter_packages")
    if not isinstance(packages, dict) or not packages:
        blockers.append("expected_state.adapter_packages must be a non-empty object")
    return blockers


def _expected_manifest_names(expected: dict[str, Any]) -> list[str]:
    model_links = expected.get("model_links")
    if not isinstance(model_links, list):
        return []
    return [value for value in model_links if isinstance(value, str) and value]


def _verify_post_pull_state(payload: dict[str, Any]) -> list[str]:
    expected = payload.get("expected_state")
    if not isinstance(expected, dict):
        return ["expected_state must be an object before checking post-pull state"]
    state = payload.get("state_after_pull")
    if not isinstance(state, dict):
        return ["state_after_pull must be an object"]

    blockers: list[str] = []
    for label, key in (
        ("adapter package directory", "adapters"),
        ("runtime directory", "runtime"),
        ("model link", "model_links"),
    ):
        expected_values = expected.get(key)
        present_values = state.get(key)
        if not isinstance(expected_values, list):
            continue
        if not isinstance(present_values, list):
            blockers.append(f"state_after_pull.{key} must be a list")
            continue
        present = set(present_values)
        for expected_value in expected_values:
            if expected_value not in present:
                blockers.append(
                    f"state_after_pull missing expected {label}: {expected_value}"
                )

    expected_manifests = _expected_manifest_names(expected)
    present_manifests = state.get("manifests")
    if expected_manifests:
        if not isinstance(present_manifests, list):
            blockers.append("state_after_pull.manifests must be a list")
        else:
            present = set(present_manifests)
            for expected_manifest in expected_manifests:
                if expected_manifest not in present:
                    blockers.append(
                        f"state_after_pull missing expected manifest: {expected_manifest}"
                    )

    expected_packages = expected.get("adapter_packages")
    present_packages = state.get("adapter_packages")
    if isinstance(expected_packages, dict):
        if not isinstance(present_packages, dict):
            blockers.append("state_after_pull.adapter_packages must be an object")
        else:
            for package_name, expected_version in expected_packages.items():
                actual_version = present_packages.get(package_name)
                if actual_version is None:
                    blockers.append(
                        "state_after_pull missing expected adapter package metadata: "
                        f"{package_name}=={expected_version}"
                    )
                elif actual_version != expected_version:
                    blockers.append(
                        f"state_after_pull adapter package {package_name} expected "
                        f"{expected_version}, found {actual_version}"
                    )

    bytes_by_area = state.get("bytes")
    if not isinstance(bytes_by_area, dict):
        blockers.append("state_after_pull.bytes must be an object")
    else:
        blob_bytes = bytes_by_area.get("blobs")
        if not isinstance(blob_bytes, int) or blob_bytes <= 0:
            blockers.append("state_after_pull.bytes.blobs must be positive")
    return blockers


def _verify_pre_pull_state(payload: dict[str, Any]) -> list[str]:
    expected = payload.get("expected_state")
    if not isinstance(expected, dict):
        return ["expected_state must be an object before checking pre-pull state"]
    state = payload.get("pre_pull_state")
    if not isinstance(state, dict):
        return ["pre_pull_state must be an object"]

    blockers: list[str] = []
    for label, key in (
        ("adapter package directory", "adapters"),
        ("runtime directory", "runtime"),
        ("model link", "model_links"),
    ):
        expected_values = expected.get(key)
        present_values = state.get(key)
        if not isinstance(expected_values, list):
            continue
        if not isinstance(present_values, list):
            blockers.append(f"pre_pull_state.{key} must be a list")
            continue
        present = set(present_values)
        for expected_value in expected_values:
            if expected_value in present:
                blockers.append(
                    f"pre_pull_state already contains expected {label}: {expected_value}"
                )

    expected_manifests = _expected_manifest_names(expected)
    present_manifests = state.get("manifests")
    if expected_manifests:
        if not isinstance(present_manifests, list):
            blockers.append("pre_pull_state.manifests must be a list")
        else:
            present = set(present_manifests)
            for expected_manifest in expected_manifests:
                if expected_manifest in present:
                    blockers.append(
                        f"pre_pull_state already contains expected manifest: {expected_manifest}"
                    )

    expected_packages = expected.get("adapter_packages")
    present_packages = state.get("adapter_packages")
    if isinstance(expected_packages, dict):
        if not isinstance(present_packages, dict):
            blockers.append("pre_pull_state.adapter_packages must be an object")
        else:
            for package_name in expected_packages:
                actual_version = present_packages.get(package_name)
                if actual_version is not None:
                    blockers.append(
                        "pre_pull_state already contains expected adapter package metadata: "
                        f"{package_name}=={actual_version}"
                    )
    return blockers


def _verify_commands(payload: dict[str, Any]) -> list[str]:
    commands = payload.get("commands")
    if not isinstance(commands, list):
        return ["commands must be a list"]

    by_label: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for item in commands:
        if not isinstance(item, dict):
            blockers.append("commands entries must be objects")
            continue
        label = item.get("label")
        if isinstance(label, str):
            by_label[label] = item

    for label in REQUIRED_COMMANDS:
        command = by_label.get(label)
        if command is None:
            blockers.append(f"missing command result: {label}")
            continue
        status = command.get("status")
        if status != 0:
            blockers.append(f"command {label} exited {status!r}")
        elapsed = command.get("elapsed_s")
        if not isinstance(elapsed, (int, float)) or elapsed < 0:
            blockers.append(f"command {label} missing non-negative elapsed_s")
    return blockers


def _audio_file_candidates(item: dict[str, Any], *, evidence_path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    recorded = item.get("path")
    if isinstance(recorded, str) and recorded:
        candidates.append(Path(recorded))
        if evidence_path is not None:
            candidates.append(evidence_path.parent / Path(recorded).name)
    return candidates


def _verify_audio_file(
    item: dict[str, Any],
    *,
    label: str,
    evidence_path: Path | None,
) -> list[str]:
    candidates = _audio_file_candidates(item, evidence_path=evidence_path)
    if not candidates:
        return [f"{label}.wav missing artifact path"]

    artifact = next((candidate for candidate in candidates if candidate.exists()), None)
    if artifact is None:
        rendered = ", ".join(str(candidate) for candidate in candidates)
        return [f"{label}.wav artifact file is not available for verification: {rendered}"]

    blockers: list[str] = []
    try:
        data = artifact.read_bytes()
    except OSError as exc:
        return [f"{label}.wav artifact file could not be read: {exc}"]

    expected_bytes = item.get("bytes")
    if isinstance(expected_bytes, int) and len(data) != expected_bytes:
        blockers.append(
            f"{label}.wav artifact byte count {len(data)} does not match evidence {expected_bytes}"
        )
    expected_sha = item.get("sha256")
    if isinstance(expected_sha, str) and hashlib.sha256(data).hexdigest() != expected_sha:
        blockers.append(f"{label}.wav artifact sha256 does not match evidence")
    return blockers


def _verify_audio(payload: dict[str, Any], *, evidence_path: Path | None) -> list[str]:
    audio = payload.get("audio")
    if not isinstance(audio, list):
        return ["audio must be a list"]

    by_name: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for item in audio:
        if not isinstance(item, dict):
            blockers.append("audio entries must be objects")
            continue
        stem = Path(str(item.get("path") or "")).stem
        if stem:
            by_name[stem] = item

    durations: dict[str, float] = {}
    for name in ("short", "long"):
        item = by_name.get(name)
        if item is None:
            blockers.append(f"missing audio artifact stats: {name}.wav")
            continue
        if item.get("exists") is not True:
            blockers.append(f"{name}.wav does not exist")
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            blockers.append(f"{name}.wav has no bytes")
        duration = item.get("duration_s")
        if not isinstance(duration, (int, float)) or duration <= 0:
            blockers.append(f"{name}.wav has no positive duration")
        else:
            durations[name] = float(duration)
        if item.get("silent") is not False:
            blockers.append(f"{name}.wav must be non-silent")
        for metric in ("sample_rate", "channels", "sample_width", "peak", "rms", "sha256"):
            if item.get(metric) is None:
                blockers.append(f"{name}.wav missing {metric}")
        blockers.extend(_verify_audio_file(item, label=name, evidence_path=evidence_path))

    if "short" in durations and "long" in durations and durations["long"] < durations["short"]:
        blockers.append("long.wav duration is shorter than short.wav duration")
    return blockers


def _verify_resource_samples(payload: dict[str, Any]) -> list[str]:
    samples = payload.get("resource_samples")
    if not isinstance(samples, dict):
        return ["resource_samples must be an object"]

    blockers: list[str] = []
    for label in REQUIRED_RESOURCE_SAMPLE_LABELS:
        summary = samples.get(label)
        if not isinstance(summary, dict):
            blockers.append(f"missing resource sample summary: {label}")
            continue
        if summary.get("exists") is not True:
            blockers.append(f"resource sample file missing for {label}")
        count = summary.get("samples")
        if not isinstance(count, int) or count <= 0:
            blockers.append(f"resource sample summary has no samples for {label}")
    return blockers


def _verify_goal_checklist(payload: dict[str, Any]) -> list[str]:
    checklist = payload.get("goal_checklist")
    if not isinstance(checklist, dict):
        return ["goal_checklist must be an object"]

    blockers: list[str] = []
    for name in REQUIRED_GOAL_CHECKS:
        item = checklist.get(name)
        if not isinstance(item, dict):
            blockers.append(f"goal_checklist missing check: {name}")
            continue
        if item.get("status") != "passed":
            blockers.append(
                f"goal_checklist.{name} status is {item.get('status')!r}, expected 'passed'"
            )
        if item.get("blockers") not in ([], None):
            blockers.append(f"goal_checklist.{name} has blockers: {item.get('blockers')!r}")
        evidence_keys = item.get("evidence")
        if not isinstance(evidence_keys, list) or not evidence_keys:
            blockers.append(f"goal_checklist.{name} must list evidence keys")
    return blockers


def _verify_evidence(
    payload: dict[str, Any],
    *,
    evidence_path: Path | None = None,
    expect_model: str | None = None,
    expect_proof_target: str | None = None,
) -> list[str]:
    blockers: list[str] = []

    version = payload.get("evidence_schema_version")
    if version != EXPECTED_SCHEMA_VERSION:
        blockers.append(
            f"unsupported evidence_schema_version {version!r}; "
            f"expected {EXPECTED_SCHEMA_VERSION}"
        )

    if payload.get("mode") != "local-docker-clean-pull":
        blockers.append("evidence mode must be local-docker-clean-pull")

    if expect_model is not None and payload.get("model") != expect_model:
        blockers.append(
            f"model mismatch: expected {expect_model!r}, found {payload.get('model')!r}"
        )
    if (
        expect_proof_target is not None
        and payload.get("proof_target") != expect_proof_target
    ):
        blockers.append(
            "proof_target mismatch: "
            f"expected {expect_proof_target!r}, found {payload.get('proof_target')!r}"
        )

    proof_ready = payload.get("proof_ready")
    proof_blockers = payload.get("proof_blockers")
    if proof_ready is not True:
        blockers.append("proof_ready is not true")
    if proof_blockers:
        if isinstance(proof_blockers, list):
            blockers.extend(f"proof blocker: {item}" for item in proof_blockers)
        else:
            blockers.append("proof_blockers must be an empty list for passing proof")
    elif proof_blockers != []:
        blockers.append("proof_blockers must be present as a list")

    for key in (
        "model",
        "image",
        "expected_state",
        "pre_pull_state",
        "state_after_pull",
        "commands",
        "audio",
        "resource_samples",
        "goal_checklist",
    ):
        if key not in payload:
            blockers.append(f"missing evidence field: {key}")

    if payload.get("estimate_only") is True:
        blockers.append("estimate-only evidence cannot be proof")
    if payload.get("allow_download") is not True:
        blockers.append("allow_download must be true for proof")
    if payload.get("audio_usable") != "yes":
        blockers.append("audio_usable must be yes for proof")
    if payload.get("failure_class") != "none":
        blockers.append("failure_class must be none for proof")
    if payload.get("failure_note"):
        blockers.append("failure_note must be empty for proof")

    image_inspect = payload.get("image_inspect")
    if not isinstance(image_inspect, dict) or image_inspect.get("status") != 0:
        blockers.append("image_inspect must show status 0")
    if payload.get("resource_before") is None:
        blockers.append("resource_before must be present")
    if payload.get("resource_after") is None:
        blockers.append("resource_after must be present")
    if payload.get("download_estimate") is None:
        blockers.append("download_estimate must be present")

    for key in (
        "failure_reasons",
        "download_guard_failures",
        "clean_state_failures",
        "state_failures",
        "voice_reference_failures",
        "skipped_commands",
    ):
        blockers.extend(_require_empty_list(payload, key))

    blockers.extend(_verify_expected_state(payload))
    blockers.extend(_verify_pre_pull_state(payload))
    blockers.extend(_verify_post_pull_state(payload))
    blockers.extend(_verify_commands(payload))
    blockers.extend(_verify_audio(payload, evidence_path=evidence_path))
    blockers.extend(_verify_resource_samples(payload))
    blockers.extend(_verify_goal_checklist(payload))

    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evidence",
        nargs="+",
        type=Path,
        help="Path to one or more local-smoke-evidence.json files.",
    )
    parser.add_argument(
        "--expect-model",
        help="Require every evidence file to prove this exact model id.",
    )
    parser.add_argument(
        "--expect-proof-target",
        help="Require every evidence file to prove this named proof target.",
    )
    args = parser.parse_args()

    failed = False
    for path in args.evidence:
        payload, error = _load_json(path)
        if error is not None:
            failed = True
            print(f"{path}: FAIL", file=sys.stderr)
            print(f"  - {error}", file=sys.stderr)
            continue
        assert payload is not None
        blockers = _verify_evidence(
            payload,
            evidence_path=path,
            expect_model=args.expect_model,
            expect_proof_target=args.expect_proof_target,
        )
        if blockers:
            failed = True
            print(f"{path}: FAIL", file=sys.stderr)
            for blocker in blockers:
                print(f"  - {blocker}", file=sys.stderr)
            continue

        model = payload.get("model", "?")
        target = payload.get("proof_target") or "custom"
        print(f"{path}: PASS model={model} proof_target={target}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
