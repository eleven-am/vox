from __future__ import annotations

from pathlib import Path
from typing import Any

SAMPLED_COMMAND_LABELS = {"pull", "short synthesis", "long synthesis"}


def _command_statuses(evidence: dict[str, Any]) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for command in evidence.get("commands", []):
        if not isinstance(command, dict):
            continue
        label = command.get("label")
        status = command.get("status")
        if isinstance(label, str) and isinstance(status, int):
            statuses[label] = status
    return statuses


def _audio_proof_blockers(evidence: dict[str, Any]) -> list[str]:
    audio = evidence.get("audio")
    if not isinstance(audio, list) or len(audio) < 2:
        return ["proof requires short and long synthesis audio artifacts"]

    blockers: list[str] = []
    durations: dict[str, float] = {}
    for item in audio:
        if not isinstance(item, dict):
            blockers.append("proof audio entry is not an object")
            continue
        path = Path(str(item.get("path") or ""))
        label = path.stem or "audio"
        if not item.get("exists"):
            blockers.append(f"proof audio missing: {label}")
            continue
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            blockers.append(f"proof audio is empty: {label}")
        duration = item.get("duration_s")
        if not isinstance(duration, int | float) or duration <= 0:
            blockers.append(f"proof audio has no positive duration: {label}")
        else:
            durations[label] = float(duration)
        if item.get("silent"):
            blockers.append(f"proof audio is silent: {label}")
        for metric in ("sample_rate", "channels", "sample_width", "peak", "rms"):
            if item.get(metric) is None:
                blockers.append(f"proof audio missing {metric}: {label}")

    short_duration = durations.get("short")
    long_duration = durations.get("long")
    if short_duration is not None and long_duration is not None and long_duration < short_duration:
        blockers.append("proof long synthesis audio is shorter than short synthesis audio")
    return blockers


def _expected_state_proof_blockers(evidence: dict[str, Any]) -> list[str]:
    expected = evidence.get("expected_state")
    if not isinstance(expected, dict):
        return ["proof requires expected adapter/runtime/model state"]

    required = {
        "adapters": "expected adapter name",
        "adapter_packages": "expected adapter package version",
        "runtime": "expected runtime directory",
        "model_links": "expected model link",
    }
    blockers: list[str] = []
    for key, label in required.items():
        value = expected.get(key)
        missing = not value if isinstance(value, dict | list) else True
        if missing:
            blockers.append(f"proof requires {label}")
    return blockers


def _resource_sample_proof_blockers(evidence: dict[str, Any]) -> list[str]:
    interval = evidence.get("resource_sample_interval_s")
    if not isinstance(interval, int | float) or interval <= 0:
        return ["proof requires continuous resource sampling during pull and synthesis"]

    samples = evidence.get("resource_samples")
    if not isinstance(samples, dict):
        return ["proof requires resource sample summaries"]

    blockers: list[str] = []
    for label in sorted(SAMPLED_COMMAND_LABELS):
        summary = samples.get(label)
        if not isinstance(summary, dict):
            blockers.append(f"proof missing resource samples for {label}")
            continue
        if not summary.get("exists"):
            blockers.append(f"proof missing resource sample file for {label}")
        sample_count = summary.get("samples")
        if not isinstance(sample_count, int) or sample_count <= 0:
            blockers.append(f"proof has no resource samples for {label}")
    return blockers


def proof_readiness(evidence: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether evidence satisfies the clean-pull proof contract."""

    blockers: list[str] = []
    if evidence.get("estimate_only"):
        blockers.append("estimate-only evidence is not a synthesis proof")
    if not evidence.get("allow_download"):
        blockers.append("proof requires --allow-download")
    if evidence.get("failure_reasons"):
        blockers.extend(f"smoke failure: {reason}" for reason in evidence["failure_reasons"])
    if evidence.get("download_guard_failures"):
        blockers.extend(f"download guard blocked proof: {reason}" for reason in evidence["download_guard_failures"])
    if evidence.get("clean_state_failures"):
        blockers.extend(f"pre-pull state was not clean: {reason}" for reason in evidence["clean_state_failures"])
    if evidence.get("state_failures"):
        blockers.extend(f"post-pull state mismatch: {reason}" for reason in evidence["state_failures"])
    if evidence.get("voice_reference_failures"):
        blockers.extend(f"voice reference invalid: {reason}" for reason in evidence["voice_reference_failures"])

    for command in evidence.get("skipped_commands") or []:
        if isinstance(command, dict):
            blockers.append(
                f"proof command skipped: {command.get('label', '?')} ({command.get('reason', '?')})"
            )

    statuses = _command_statuses(evidence)
    for label in (
        "resource snapshot before pull",
        "runtime snapshot",
        "download size estimate",
        "pre-pull clean state",
        "pull",
        "short synthesis",
        "long synthesis",
        "resource snapshot after smoke",
    ):
        status = statuses.get(label)
        if status is None:
            blockers.append(f"proof missing command: {label}")
        elif status != 0:
            blockers.append(f"proof command failed: {label} exited {status}")

    image_inspect = evidence.get("image_inspect")
    if not isinstance(image_inspect, dict) or image_inspect.get("status") != 0:
        blockers.append("proof requires successful Docker image inspect metadata")
    if evidence.get("resource_before") is None:
        blockers.append("proof requires resource snapshot before pull")
    if evidence.get("resource_after") is None:
        blockers.append("proof requires resource snapshot after smoke")
    if evidence.get("download_estimate") is None:
        blockers.append("proof requires a parseable download estimate")
    if evidence.get("audio_usable") != "yes":
        blockers.append("proof requires manual --audio-usable yes")

    blockers.extend(_expected_state_proof_blockers(evidence))
    blockers.extend(_audio_proof_blockers(evidence))
    blockers.extend(_resource_sample_proof_blockers(evidence))

    deduped: list[str] = []
    for blocker in blockers:
        if blocker not in deduped:
            deduped.append(blocker)
    return not deduped, deduped


def _goal_check(status: str, evidence_keys: list[str], blockers: list[str]) -> dict[str, Any]:
    return {
        "status": status,
        "evidence": evidence_keys,
        "blockers": blockers,
    }


def _audio_by_stem(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_stem: dict[str, dict[str, Any]] = {}
    audio = evidence.get("audio")
    if not isinstance(audio, list):
        return by_stem
    for item in audio:
        if not isinstance(item, dict):
            continue
        stem = Path(str(item.get("path") or "")).stem
        if stem:
            by_stem[stem] = item
    return by_stem


def _audio_item_is_usable(item: dict[str, Any] | None) -> list[str]:
    if not isinstance(item, dict):
        return ["audio artifact is missing"]
    blockers: list[str] = []
    if item.get("exists") is not True:
        blockers.append("audio artifact does not exist")
    if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
        blockers.append("audio artifact is empty")
    duration = item.get("duration_s")
    if not isinstance(duration, int | float) or duration <= 0:
        blockers.append("audio artifact has no positive duration")
    if item.get("silent") is not False:
        blockers.append("audio artifact is silent or silence was not checked")
    return blockers


def _goal_check_status(blockers: list[str]) -> str:
    return "passed" if not blockers else "blocked"


def goal_checklist(evidence: dict[str, Any]) -> dict[str, Any]:
    """Map local clean-pull evidence to the model-readiness checklist."""

    statuses = _command_statuses(evidence)
    audio = _audio_by_stem(evidence)
    expected = evidence.get("expected_state") if isinstance(evidence.get("expected_state"), dict) else {}
    state_after_pull = evidence.get("state_after_pull") if isinstance(evidence.get("state_after_pull"), dict) else {}
    bytes_after_pull = state_after_pull.get("bytes", {}) if isinstance(state_after_pull, dict) else {}

    checks: dict[str, dict[str, Any]] = {}

    clean_blockers: list[str] = []
    if statuses.get("pre-pull clean state") != 0:
        clean_blockers.append("pre-pull clean-state command did not pass")
    clean_blockers.extend(str(reason) for reason in evidence.get("clean_state_failures") or [])
    checks["clean_runtime_and_model_store"] = _goal_check(
        _goal_check_status(clean_blockers),
        ["pre_pull_state", "clean_state_failures", "commands"],
        clean_blockers,
    )

    pull_blockers: list[str] = []
    if statuses.get("pull") != 0:
        pull_blockers.append("vox pull did not complete successfully")
    pull_blockers.extend(str(reason) for reason in evidence.get("download_guard_failures") or [])
    checks["vox_pull"] = _goal_check(
        _goal_check_status(pull_blockers),
        ["commands", "download_estimate", "download_guard_failures"],
        pull_blockers,
    )

    state_blockers = [
        *_expected_state_proof_blockers(evidence),
        *(str(reason) for reason in evidence.get("state_failures") or []),
    ]
    checks["adapter_runtime_and_model_artifacts"] = _goal_check(
        _goal_check_status(state_blockers),
        ["expected_state", "state_after_pull", "state_failures"],
        state_blockers,
    )

    package_blockers: list[str] = []
    adapter_packages = expected.get("adapter_packages") if isinstance(expected, dict) else None
    if not isinstance(adapter_packages, dict) or not adapter_packages:
        package_blockers.append("expected adapter package version was not declared")
    checks["adapter_package_version"] = _goal_check(
        _goal_check_status(package_blockers),
        ["expected_state.adapter_packages", "state_after_pull.adapter_packages"],
        package_blockers,
    )

    runtime_blockers: list[str] = []
    runtimes = expected.get("runtime") if isinstance(expected, dict) else None
    if not isinstance(runtimes, list) or not runtimes:
        runtime_blockers.append("expected isolated runtime directory was not declared")
    checks["isolated_runtime_directory"] = _goal_check(
        _goal_check_status(runtime_blockers),
        ["expected_state.runtime", "state_after_pull.runtime"],
        runtime_blockers,
    )

    model_artifact_blockers: list[str] = []
    model_links = expected.get("model_links") if isinstance(expected, dict) else None
    if not isinstance(model_links, list) or not model_links:
        model_artifact_blockers.append("expected model link was not declared")
    if not isinstance(bytes_after_pull.get("blobs"), int) or bytes_after_pull["blobs"] <= 0:
        model_artifact_blockers.append("model blob storage is empty after pull")
    checks["model_artifacts"] = _goal_check(
        _goal_check_status(model_artifact_blockers),
        ["expected_state.model_links", "state_after_pull.model_links", "state_after_pull.bytes.blobs"],
        model_artifact_blockers,
    )

    short_blockers = _audio_item_is_usable(audio.get("short"))
    if statuses.get("short synthesis") != 0:
        short_blockers.append("short synthesis command did not pass")
    checks["short_synthesis"] = _goal_check(
        _goal_check_status(short_blockers),
        ["commands", "audio.short"],
        short_blockers,
    )

    long_blockers = _audio_item_is_usable(audio.get("long"))
    if statuses.get("long synthesis") != 0:
        long_blockers.append("long synthesis command did not pass")
    checks["long_synthesis"] = _goal_check(
        _goal_check_status(long_blockers),
        ["commands", "audio.long"],
        long_blockers,
    )

    latency_blockers: list[str] = []
    for label in ("pull", "short synthesis", "long synthesis"):
        command = next(
            (
                item
                for item in evidence.get("commands", [])
                if isinstance(item, dict) and item.get("label") == label
            ),
            None,
        )
        elapsed = command.get("elapsed_s") if isinstance(command, dict) else None
        if not isinstance(elapsed, int | float) or elapsed < 0:
            latency_blockers.append(f"{label} elapsed_s is missing")
    checks["latency_recorded"] = _goal_check(
        _goal_check_status(latency_blockers),
        ["commands[].elapsed_s"],
        latency_blockers,
    )

    resource_blockers = _resource_sample_proof_blockers(evidence)
    if evidence.get("resource_before") is None:
        resource_blockers.append("resource snapshot before pull is missing")
    if evidence.get("resource_after") is None:
        resource_blockers.append("resource snapshot after smoke is missing")
    checks["ram_vram_recorded"] = _goal_check(
        _goal_check_status(resource_blockers),
        ["resource_before", "resource_after", "resource_samples"],
        resource_blockers,
    )

    audio_usability_blockers: list[str] = []
    if evidence.get("audio_usable") != "yes":
        audio_usability_blockers.append("manual audio usability was not confirmed with --audio-usable yes")
    checks["manual_audio_usability"] = _goal_check(
        _goal_check_status(audio_usability_blockers),
        ["audio_usable"],
        audio_usability_blockers,
    )

    failure_blockers: list[str] = []
    if evidence.get("failure_reasons") and evidence.get("failure_class") == "none":
        failure_blockers.append("failing evidence is not classified")
    if evidence.get("failure_reasons") and not str(evidence.get("failure_note") or "").strip():
        failure_blockers.append("failing evidence has no concrete failure note")
    checks["failure_classification"] = _goal_check(
        _goal_check_status(failure_blockers),
        ["failure_class", "failure_note", "failure_reasons"],
        failure_blockers,
    )

    return checks
