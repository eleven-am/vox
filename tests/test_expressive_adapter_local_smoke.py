from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_local_smoke_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "expressive-adapter-local-smoke.py"
    spec = importlib.util.spec_from_file_location("expressive_adapter_local_smoke", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verifier_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "verify-expressive-adapter-evidence.py"


def _wav_artifact_stats(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ready_evidence(artifact_dir: Path | None = None) -> dict:
    short_path = Path("/tmp/vox-adapter-lab/dia-tts-1.6b/artifacts/short.wav")
    long_path = Path("/tmp/vox-adapter-lab/dia-tts-1.6b/artifacts/long.wav")
    short_bytes = 4800
    long_bytes = 9600
    short_sha = "a" * 64
    long_sha = "b" * 64
    if artifact_dir is not None:
        short_path = artifact_dir / "short.wav"
        long_path = artifact_dir / "long.wav"
        _write_wav(short_path, frames=2400)
        _write_wav(long_path, frames=4800)
        short_stats = _wav_artifact_stats(short_path)
        long_stats = _wav_artifact_stats(long_path)
        short_bytes = short_stats["bytes"]
        long_bytes = long_stats["bytes"]
        short_sha = short_stats["sha256"]
        long_sha = long_stats["sha256"]

    commands = [
        {"label": label, "status": 0, "elapsed_s": 0.1, "command": ["true"], "stdout": "", "stderr": ""}
        for label in (
            "resource snapshot before pull",
            "runtime snapshot",
            "download size estimate",
            "pre-pull clean state",
            "pull",
            "short synthesis",
            "long synthesis",
            "resource snapshot after smoke",
        )
    ]
    audio = [
        {
            "path": f"/tmp/vox-adapter-lab/dia-tts-1.6b/artifacts/{name}.wav",
            "exists": True,
            "bytes": bytes_count,
            "sha256": digest,
            "duration_s": duration,
            "sample_rate": 24_000,
            "channels": 1,
            "sample_width": 2,
            "peak": 0.25,
            "rms": 0.1,
            "silent": False,
        }
        for name, bytes_count, digest, duration in (
            ("short", short_bytes, short_sha, 0.1),
            ("long", long_bytes, long_sha, 0.2),
        )
    ]
    if artifact_dir is not None:
        audio[0]["path"] = str(short_path)
        audio[1]["path"] = str(long_path)
    resource_samples = {
        label: {
            "path": f"/tmp/vox-adapter-lab/dia-tts-1.6b/tmp/resource-{label}.jsonl",
            "exists": True,
            "samples": 2,
            "peak_ram_used_bytes": 1024,
            "peak_gpu_memory_used_mib": 512,
        }
        for label in ("pull", "short synthesis", "long synthesis")
    }
    goal_checklist = {
        name: {"status": "passed", "evidence": ["test"], "blockers": []}
        for name in (
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
    }
    return {
        "evidence_schema_version": 2,
        "mode": "local-docker-clean-pull",
        "model": "dia-tts:1.6b",
        "image": "ghcr.io/eleven-am/vox:latest",
        "expected_state": {
            "adapters": ["vox-dia"],
            "adapter_packages": {"vox-dia": "0.2.15"},
            "runtime": ["dia"],
            "model_links": ["dia-tts"],
        },
        "commands": commands,
        "audio": audio,
        "resource_samples": resource_samples,
        "goal_checklist": goal_checklist,
        "proof_target": "dia",
        "proof_ready": True,
        "proof_blockers": [],
        "estimate_only": False,
        "allow_download": True,
        "audio_usable": "yes",
        "failure_class": "none",
        "failure_note": "",
        "image_inspect": {"status": 0},
        "resource_before": {"ram": {"total_bytes": 2048, "available_bytes": 1024}},
        "resource_after": {"ram": {"total_bytes": 2048, "available_bytes": 1024}},
        "download_estimate": {"source": "nari-labs/Dia-1.6B-0626"},
        "pre_pull_state": {
            "adapters": [],
            "adapter_packages": {},
            "runtime": [],
            "manifests": [],
            "model_links": [],
            "voices": [],
            "file_counts": {
                "adapters": 0,
                "runtime": 0,
                "manifests": 0,
                "model_links": 0,
                "blobs": 0,
                "voices": 0,
            },
            "bytes": {
                "adapters": 0,
                "runtime": 0,
                "manifests": 0,
                "model_links": 0,
                "blobs": 0,
                "voices": 0,
            },
        },
        "state_after_pull": {
            "adapters": ["vox-dia"],
            "adapter_packages": {"vox-dia": "0.2.15"},
            "runtime": ["dia"],
            "manifests": ["dia-tts"],
            "model_links": ["dia-tts"],
            "voices": [],
            "file_counts": {
                "adapters": 2,
                "runtime": 10,
                "manifests": 1,
                "model_links": 1,
                "blobs": 12,
                "voices": 0,
            },
            "bytes": {
                "adapters": 1024,
                "runtime": 2048,
                "manifests": 512,
                "model_links": 128,
                "blobs": 4096,
                "voices": 0,
            },
        },
        "failure_reasons": [],
        "download_guard_failures": [],
        "clean_state_failures": [],
        "state_failures": [],
        "voice_reference_failures": [],
        "skipped_commands": [],
    }


def _resource_stdout() -> str:
    return json.dumps({
        "ram": {"total_bytes": 1024, "available_bytes": 512},
        "nvidia_smi_available": False,
        "gpus": [],
    })


def _image_inspect_stdout() -> str:
    return json.dumps({
        "Id": "sha256:abc123",
        "RepoDigests": ["ghcr.io/eleven-am/vox@sha256:def456"],
        "RepoTags": ["vox:test"],
        "Architecture": "amd64",
        "Os": "linux",
        "Size": 123456,
    })


def _download_estimate_stdout(
    *,
    known_bytes: int = 1234,
    known_gib: float = 1234 / (1024 ** 3),
    unknown_size_files: list[str] | None = None,
    missing: list[str] | None = None,
) -> str:
    unknown = unknown_size_files or []
    missing_items = missing or []
    return json.dumps({
        "model": "chatterbox-tts-turbo:0.1.7",
        "variant": "onnx",
        "resolved_name": "chatterbox-tts-turbo",
        "resolved_tag": "0.1.7",
        "resolved_variant": "onnx",
        "source": "ResembleAI/chatterbox-turbo-ONNX",
        "file_count": 2,
        "known_bytes": known_bytes,
        "known_gib": known_gib,
        "unknown_size_files": unknown,
        "files": [
            {"filename": "onnx/model.onnx", "size": 1000},
            {"filename": "onnx/voices.bin", "size": 234},
        ],
        "missing": missing_items,
        "warnings": [],
    })


def _fake_registry_entry(_url: str) -> dict:
    return {
        "name": "chatterbox-tts-turbo",
        "tag": "0.1.7",
        "type": "tts",
        "variants": [
            {
                "id": "onnx",
                "priority": 0,
                "fallback": True,
                "adapter": "chatterbox-tts-onnx",
                "adapter_package": "vox-chatterbox",
                "format": "onnx",
                "source": "ResembleAI/chatterbox-turbo-ONNX",
                "architecture": "chatterbox",
                "files": ["onnx/model.onnx", "onnx/voices.bin"],
                "requires": {},
            }
        ],
    }


def _fake_repo_info(_source: str, *, unknown: bool = False):
    return SimpleNamespace(siblings=[
        SimpleNamespace(rfilename="onnx/model.onnx", size=1000),
        SimpleNamespace(rfilename="onnx/voices.bin", size=None if unknown else 234),
        SimpleNamespace(rfilename=".gitattributes", size=10),
    ])


def _common_success_response(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    inner = command[-1]
    if "meminfo_bytes()" in inner:
        return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
    if "repo_info = api.repo_info(source)" in inner:
        return subprocess.CompletedProcess(command, 0, stdout=_download_estimate_stdout(), stderr="")
    return None


def _write_resource_samples(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "at": 1.0,
            "ram": {"total_bytes": 2048, "available_bytes": 1536},
            "gpus": [{"index": "0", "memory_total_mib": 8192, "memory_used_mib": 128, "memory_free_mib": 8064}],
        },
        {
            "at": 2.0,
            "ram": {"total_bytes": 2048, "available_bytes": 1024},
            "gpus": [{"index": "0", "memory_total_mib": 8192, "memory_used_mib": 512, "memory_free_mib": 7680}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_local_smoke_refuses_download_without_explicit_flag(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fail_run(*args, **kwargs):
        raise AssertionError("docker must not run without --allow-download")

    monkeypatch.setattr(subprocess, "run", fail_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["allow_download"] is False
    assert evidence["failure_reasons"] == [
        "--allow-download is required before running clean-pull smoke",
        "failing smoke run must set --failure-class to one of "
        "Vox, adapter, dependency, upstream, or hardware",
    ]
    assert evidence["failure_class"] == "none"
    assert evidence["scratch"] == str(tmp_path / "dia-tts-1.6b")
    assert "storage_before_cleanup" in evidence
    assert "storage_after_cleanup" in evidence
    assert "disk_after" in evidence
    assert "disk_after_cleanup" in evidence
    assert evidence["cleanup_removed"] == []


def test_local_smoke_cleanup_is_recorded_even_before_download_guard_passes(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fail_run(*args, **kwargs):
        raise AssertionError("docker must not run without --allow-download")

    monkeypatch.setattr(subprocess, "run", fail_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--cleanup",
        ],
    )

    assert module.main() == 1

    scratch = tmp_path / "dia-tts-1.6b"
    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["cleanup_requested"] is True
    assert evidence["storage_before_cleanup"]["artifacts"] == 0
    assert evidence["storage_after_cleanup"]["artifacts"] == 0
    assert evidence["disk_after_cleanup"]["free"] <= evidence["disk_after_cleanup"]["total"]
    assert not (scratch / "vox-home").exists()
    assert not (scratch / "hf-cache").exists()
    assert not (scratch / "tmp").exists()


def test_download_estimate_resolves_variant_and_selected_files():
    module = _load_local_smoke_module()

    estimate = module._download_estimate(
        "chatterbox-tts-turbo:0.1.7",
        "onnx",
        registry_get=_fake_registry_entry,
        repo_info_fetcher=_fake_repo_info,
    )

    assert estimate["resolved_name"] == "chatterbox-tts-turbo"
    assert estimate["resolved_tag"] == "0.1.7"
    assert estimate["resolved_variant"] == "onnx"
    assert estimate["source"] == "ResembleAI/chatterbox-turbo-ONNX"
    assert estimate["file_count"] == 2
    assert estimate["known_bytes"] == 1234
    assert estimate["unknown_size_files"] == []
    assert [item["filename"] for item in estimate["files"]] == [
        "onnx/model.onnx",
        "onnx/voices.bin",
    ]


def test_local_smoke_estimate_only_does_not_require_download_or_docker(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fail_run(*args, **kwargs):
        raise AssertionError("docker must not run for --estimate-only")

    monkeypatch.setattr(subprocess, "run", fail_run)
    monkeypatch.setattr(
        module,
        "_download_estimate",
        lambda model, variant: {
            "model": model,
            "variant": variant or "auto",
            "known_bytes": 3 * 1024 ** 3,
            "known_gib": 3.0,
            "unknown_size_files": [],
            "files": [{"filename": "model.bin", "size": 3 * 1024 ** 3}],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--estimate-only",
            "--max-download-gb",
            "1",
        ],
    )

    assert module.main() == 0

    scratch = tmp_path / "dia-tts-1.6b"
    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["estimate_only"] is True
    assert evidence["allow_download"] is False
    assert evidence["download_estimate"]["known_gib"] == 3.0
    assert evidence["would_skip_pull"] is True
    assert evidence["download_guard_failures"] == [
        "estimated download size 3.00GiB exceeds --max-download-gb 1; "
        "pass --allow-large-download to pull anyway"
    ]
    assert evidence["failure_reasons"] == []
    assert evidence["proof_ready"] is False
    assert "estimate-only evidence is not a synthesis proof" in evidence["proof_blockers"]
    assert "proof requires --allow-download" in evidence["proof_blockers"]
    assert evidence["cleanup_removed"] == []


def test_local_smoke_estimate_only_reports_incompatible_runtime_as_pull_blocker(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fail_run(*args, **kwargs):
        raise AssertionError("docker must not run for --estimate-only")

    monkeypatch.setattr(subprocess, "run", fail_run)
    monkeypatch.setattr(
        module,
        "_download_estimate",
        lambda model, variant: {
            "model": model,
            "variant": variant or "auto",
            "known_bytes": 1024,
            "known_gib": 1024 / (1024 ** 3),
            "unknown_size_files": [],
            "missing": [
                "requires one of accelerators cuda, which is not available in this environment",
                "requires system linux; detected darwin",
            ],
            "files": [{"filename": "model.bin", "size": 1024}],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--estimate-only",
        ],
    )

    assert module.main() == 0

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["would_skip_pull"] is True
    assert evidence["download_guard_failures"] == [
        "model is incompatible with the detected runtime: "
        "requires one of accelerators cuda, which is not available in this environment; "
        "requires system linux; detected darwin",
    ]
    assert evidence["failure_reasons"] == []
    assert evidence["proof_ready"] is False
    assert any("download guard blocked proof" in blocker for blocker in evidence["proof_blockers"])


def test_local_smoke_proof_target_applies_safe_defaults_without_download(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fail_run(*args, **kwargs):
        raise AssertionError("docker must not run for --estimate-only")

    monkeypatch.setattr(subprocess, "run", fail_run)
    monkeypatch.setattr(
        module,
        "_download_estimate",
        lambda model, variant: {
            "model": model,
            "variant": variant or "auto",
            "known_bytes": 1024,
            "known_gib": 1024 / (1024 ** 3),
            "unknown_size_files": [],
            "files": [{"filename": "model.bin", "size": 1024}],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--proof-target",
            "cosyvoice",
            "--scratch-root",
            str(tmp_path),
            "--estimate-only",
        ],
    )

    assert module.main() == 0

    evidence = json.loads((tmp_path / "cosyvoice2-tts-0.5b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["proof_target"] == "cosyvoice"
    assert evidence["model"] == "cosyvoice2-tts:0.5b"
    assert evidence["image"] == "ghcr.io/eleven-am/vox:latest"
    assert evidence["expected_state"] == {
        "adapters": ["vox-cosyvoice"],
        "adapter_packages": {"vox-cosyvoice": "0.1.10"},
        "runtime": ["cosyvoice"],
        "model_links": ["cosyvoice2-tts"],
    }
    assert evidence["allow_download"] is False
    assert evidence["estimate_only"] is True
    assert evidence["failure_reasons"] == []
    assert evidence["proof_ready"] is False
    assert "estimate-only evidence is not a synthesis proof" in evidence["proof_blockers"]


def test_local_smoke_proof_target_rejects_conflicting_model(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fail_run(*args, **kwargs):
        raise AssertionError("docker must not run when argument validation fails")

    monkeypatch.setattr(subprocess, "run", fail_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--proof-target",
            "dia",
            "--model",
            "orpheus-tts:medium-3b",
            "--scratch-root",
            str(tmp_path),
            "--estimate-only",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_local_smoke_builds_isolated_docker_commands_and_records_audio(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    commands: list[list[str]] = []
    scratch = tmp_path / "chatterbox-tts-turbo-0.1.7-onnx"

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=_image_inspect_stdout(), stderr="")
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        if "repo_info = api.repo_info(source)" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_download_estimate_stdout(), stderr="")
        if "vox pull" in inner:
            _write_resource_samples(scratch / "tmp" / "resource-pull.jsonl")
            (scratch / "vox-home" / "adapters" / "vox-chatterbox").mkdir(parents=True)
            (scratch / "vox-home" / "adapters" / "vox-chatterbox" / "__init__.py").write_text("")
            dist_info = scratch / "vox-home" / "adapters" / "vox-chatterbox" / "vox_chatterbox-0.1.2.dist-info"
            dist_info.mkdir(parents=True)
            (dist_info / "METADATA").write_text("Name: vox-chatterbox\nVersion: 0.1.2\n")
            (scratch / "vox-home" / "runtime" / "chatterbox").mkdir(parents=True)
            (scratch / "vox-home" / "runtime" / "chatterbox" / "runtime.py").write_text("")
            (scratch / "vox-home" / "manifests" / "chatterbox-tts-turbo").mkdir(parents=True)
            (scratch / "vox-home" / "manifests" / "chatterbox-tts-turbo" / "0.1.7.json").write_text("{}")
            (scratch / "vox-home" / "models" / "links" / "chatterbox-tts-turbo" / "0.1.7").mkdir(parents=True)
            (scratch / "vox-home" / "models" / "links" / "chatterbox-tts-turbo" / "0.1.7" / "model.onnx").write_text("")
            (scratch / "vox-home" / "models" / "blobs").mkdir(parents=True)
            (scratch / "vox-home" / "models" / "blobs" / "sha256-test").write_bytes(b"blob")
        elif "/tmp/vox-tmp/short.wav" in inner:
            _write_resource_samples(scratch / "tmp" / "resource-short-synthesis.jsonl")
            _write_wav(scratch / "tmp" / "short.wav")
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_resource_samples(scratch / "tmp" / "resource-long-synthesis.jsonl")
            _write_wav(scratch / "tmp" / "long.wav")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "chatterbox-tts-turbo:0.1.7",
            "--variant",
            "onnx",
            "--image",
            "vox:test",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--expect-adapter",
            "vox-chatterbox",
            "--expect-adapter-package",
            "vox-chatterbox==0.1.2",
            "--expect-runtime",
            "chatterbox",
            "--expect-model-link",
            "chatterbox-tts-turbo",
        ],
    )

    assert module.main() == 0

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["evidence_schema_version"] == 2
    assert evidence["mode"] == "local-docker-clean-pull"
    assert evidence["variant"] == "onnx"
    assert evidence["voice"] is None
    assert evidence["resource_sample_interval_s"] == 1.0
    assert evidence["voice_reference"] == {
        "path": None,
        "path_like": False,
        "exists": None,
    }
    assert evidence["image_inspect"] == {
        "status": 0,
        "elapsed_s": evidence["image_inspect"]["elapsed_s"],
        "id": "sha256:abc123",
        "repo_digests": ["ghcr.io/eleven-am/vox@sha256:def456"],
        "repo_tags": ["vox:test"],
        "architecture": "amd64",
        "os": "linux",
        "size_bytes": 123456,
    }
    assert evidence["expected_state"] == {
        "adapters": ["vox-chatterbox"],
        "adapter_packages": {"vox-chatterbox": "0.1.2"},
        "runtime": ["chatterbox"],
        "model_links": ["chatterbox-tts-turbo"],
    }
    assert evidence["pre_pull_state"]["adapters"] == []
    assert evidence["pre_pull_state"]["runtime"] == []
    assert evidence["pre_pull_state"]["model_links"] == []
    assert evidence["clean_state_failures"] == []
    assert evidence["state_failures"] == []
    assert evidence["voice_reference_failures"] == []
    assert evidence["failure_class"] == "none"
    assert evidence["resource_before"] == {
        "ram": {"total_bytes": 1024, "available_bytes": 512},
        "nvidia_smi_available": False,
        "gpus": [],
    }
    assert evidence["resource_after"] == {
        "ram": {"total_bytes": 1024, "available_bytes": 512},
        "nvidia_smi_available": False,
        "gpus": [],
    }
    assert evidence["download_estimate"]["source"] == "ResembleAI/chatterbox-turbo-ONNX"
    assert evidence["download_estimate"]["known_bytes"] == 1234
    assert evidence["download_estimate"]["unknown_size_files"] == []
    assert evidence["resource_samples"]["pull"]["samples"] == 2
    assert evidence["resource_samples"]["pull"]["peak_ram_used_bytes"] == 1024
    assert evidence["resource_samples"]["pull"]["peak_gpu_memory_used_mib"] == 512
    assert evidence["resource_samples"]["short synthesis"]["samples"] == 2
    assert evidence["resource_samples"]["long synthesis"]["samples"] == 2
    assert evidence["failure_reasons"] == []
    assert evidence["skipped_commands"] == []
    assert evidence["proof_ready"] is True
    assert evidence["proof_blockers"] == []
    assert set(evidence["goal_checklist"]) == {
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
    }
    assert all(item["status"] == "passed" for item in evidence["goal_checklist"].values())
    assert all(item["blockers"] == [] for item in evidence["goal_checklist"].values())
    assert [item["exists"] for item in evidence["audio"]] == [True, True]
    assert [item["sample_width"] for item in evidence["audio"]] == [2, 2]
    assert [item["silent"] for item in evidence["audio"]] == [False, False]
    assert all(item["peak"] > 0 for item in evidence["audio"])
    assert all(item["rms"] > 0 for item in evidence["audio"])
    assert [Path(item["path"]).parent.name for item in evidence["audio"]] == ["artifacts", "artifacts"]
    assert evidence["state_after_pull"]["adapters"] == ["vox-chatterbox"]
    assert evidence["state_after_pull"]["adapter_packages"] == {"vox-chatterbox": "0.1.2"}
    assert evidence["state_after_pull"]["runtime"] == ["chatterbox"]
    assert evidence["state_after_pull"]["manifests"] == ["chatterbox-tts-turbo"]
    assert evidence["state_after_pull"]["model_links"] == ["chatterbox-tts-turbo"]
    assert evidence["state_after_pull"]["file_counts"]["adapters"] == 2
    assert evidence["state_after_pull"]["file_counts"]["runtime"] == 1
    assert evidence["state_after_pull"]["file_counts"]["manifests"] == 1
    assert evidence["state_after_pull"]["file_counts"]["model_links"] == 1
    assert evidence["state_after_pull"]["file_counts"]["blobs"] == 1
    assert evidence["state_after_pull"]["bytes"]["blobs"] > 0
    assert evidence["storage_before_cleanup"]["tmp"] > 0
    assert evidence["storage_before_cleanup"]["artifacts"] > 0
    assert evidence["cleanup_removed"] == []
    assert len(commands) == 9
    assert [command["label"] for command in evidence["commands"]] == [
        "resource snapshot before pull",
        "runtime snapshot",
        "download size estimate",
        "pre-pull clean state",
        "pull",
        "short synthesis",
        "long synthesis",
        "resource snapshot after smoke",
    ]
    assert commands[0][:3] == ["docker", "image", "inspect"]
    assert all(command[:2] == ["docker", "run"] for command in commands[1:])
    assert all("vox:test" in command for command in commands)
    assert any("VOX_HOME=/home/vox/.vox" in item for command in commands for item in command)
    assert any("HF_HOME=/tmp/vox-hf" in item for command in commands for item in command)
    assert any("HUGGINGFACE_HUB_CACHE=/tmp/vox-hf/hub" in item for command in commands for item in command)
    assert any("TMPDIR=/tmp/vox-tmp" in item for command in commands for item in command)
    assert any("vox pull 'chatterbox-tts-turbo:0.1.7' --variant 'onnx'" in command[-1] for command in commands)


def test_local_smoke_passes_voice_id_without_file_check(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=_image_inspect_stdout(), stderr="")
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        if "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav")
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--voice",
            "samantha",
        ],
    )

    assert module.main() == 0

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["voice"] == "samantha"
    assert evidence["voice_reference"] == {
        "path": "samantha",
        "path_like": False,
        "exists": None,
    }
    assert evidence["voice_reference_failures"] == []
    assert evidence["failure_reasons"] == []
    assert all(command["label"] != "voice reference check" for command in evidence["commands"])
    assert any("--voice 'samantha'" in command[-1] for command in commands)


def test_local_smoke_fails_when_adapter_package_version_mismatches(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"

    def fake_run(command, *, capture_output, text, timeout):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=_image_inspect_stdout(), stderr="")
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        if "vox pull" in inner:
            dist_info = scratch / "vox-home" / "adapters" / "vox-dia" / "vox_dia-0.1.0.dist-info"
            dist_info.mkdir(parents=True)
            (dist_info / "METADATA").write_text("Name: vox-dia\nVersion: 0.1.0\n")
        elif "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav")
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--expect-adapter",
            "vox-dia",
            "--expect-adapter-package",
            "vox-dia==0.1.3",
            "--failure-class",
            "adapter",
            "--failure-note",
            "published adapter version did not match the runbook baseline",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["expected_state"]["adapter_packages"] == {"vox-dia": "0.1.3"}
    assert evidence["state_after_pull"]["adapter_packages"] == {"vox-dia": "0.1.0"}
    assert evidence["state_failures"] == [
        "expected adapter package vox-dia==0.1.3, found 0.1.0",
    ]
    assert "expected adapter package vox-dia==0.1.3, found 0.1.0" in evidence["failure_reasons"]
    assert evidence["failure_class"] == "adapter"
    assert evidence["failure_note"] == "published adapter version did not match the runbook baseline"
    assert evidence["proof_ready"] is False
    assert any("post-pull state mismatch" in blocker for blocker in evidence["proof_blockers"])


def test_local_smoke_can_disable_continuous_resource_sampling(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        if "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav")
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--resource-sample-interval",
            "0",
        ],
    )

    assert module.main() == 0

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["resource_sample_interval_s"] == 0.0
    assert evidence["resource_samples"]["pull"]["exists"] is False
    assert evidence["resource_samples"]["short synthesis"]["exists"] is False
    assert evidence["resource_samples"]["long synthesis"]["exists"] is False
    assert evidence["proof_ready"] is False
    assert "proof requires continuous resource sampling during pull and synthesis" in evidence["proof_blockers"]
    assert all("sample_path =" not in command[-1] for command in commands)


def test_local_smoke_skips_pull_when_voice_reference_path_is_missing(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=_image_inspect_stdout(), stderr="")
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        if inner.startswith("test -f "):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "indextts-tts:2",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--voice",
            "/home/vox/.vox/smoke-voices/reference.wav",
            "--failure-class",
            "adapter",
            "--failure-note",
            "reference voice was not mounted into the disposable runtime",
        ],
    )

    assert module.main() == 1

    scratch = tmp_path / "indextts-tts-2"
    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["voice_reference"] == {
        "path": "/home/vox/.vox/smoke-voices/reference.wav",
        "path_like": True,
        "exists": False,
        "status": 1,
        "stdout": "",
        "stderr": "",
    }
    assert evidence["voice_reference_failures"] == [
        "voice reference path missing inside container: /home/vox/.vox/smoke-voices/reference.wav",
    ]
    assert evidence["skipped_commands"] == [
        {"label": "download size estimate", "reason": "voice reference check failed"},
        {"label": "pre-pull clean state", "reason": "voice reference check failed"},
        {"label": "pull", "reason": "voice reference check failed"},
        {"label": "short synthesis", "reason": "voice reference check failed"},
        {"label": "long synthesis", "reason": "voice reference check failed"},
    ]
    assert evidence["failure_reasons"][:6] == [
        "voice reference check exited 1",
        "download size estimate skipped: voice reference check failed",
        "pre-pull clean state skipped: voice reference check failed",
        "pull skipped: voice reference check failed",
        "short synthesis skipped: voice reference check failed",
        "long synthesis skipped: voice reference check failed",
    ]
    assert "voice reference path missing inside container: /home/vox/.vox/smoke-voices/reference.wav" in (
        evidence["failure_reasons"]
    )
    assert all("vox pull" not in command[-1] for command in commands)


def test_local_smoke_records_command_and_audio_failures(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fake_run(command, *, capture_output, text, timeout):
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        status = 9 if "vox pull" in inner else 0
        return subprocess.CompletedProcess(command, status, stdout="", stderr="failed" if status else "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--failure-class",
            "dependency",
            "--failure-note",
            "pull command failed before synthesis",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["failure_reasons"] == [
        "pull exited 9",
        "short synthesis skipped: pull failed",
        "long synthesis skipped: pull failed",
        "short.wav was not copied out",
        "long.wav was not copied out",
        "manual audio usability verdict is unchecked",
    ]
    assert evidence["failure_class"] == "dependency"
    assert evidence["failure_note"] == "pull command failed before synthesis"
    assert evidence["skipped_commands"] == [
        {"label": "short synthesis", "reason": "pull failed"},
        {"label": "long synthesis", "reason": "pull failed"},
    ]
    assert [command["label"] for command in evidence["commands"]] == [
        "resource snapshot before pull",
        "runtime snapshot",
        "download size estimate",
        "pre-pull clean state",
        "pull",
        "resource snapshot after smoke",
    ]


def test_local_smoke_skips_pull_when_estimated_download_is_too_large(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        if "repo_info = api.repo_info(source)" in inner:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_download_estimate_stdout(known_gib=25.0),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--max-download-gb",
            "20",
            "--failure-class",
            "dependency",
            "--failure-note",
            "download estimate exceeded local scratch limit",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["download_estimate"]["known_gib"] == 25.0
    assert evidence["download_guard_failures"] == [
        "estimated download size 25.00GiB exceeds --max-download-gb 20; "
        "pass --allow-large-download to pull anyway",
    ]
    assert evidence["skipped_commands"] == [
        {"label": "pull", "reason": "pre-pull guard failed"},
        {"label": "short synthesis", "reason": "pre-pull guard failed"},
        {"label": "long synthesis", "reason": "pre-pull guard failed"},
    ]
    assert any("estimated download size 25.00GiB" in reason for reason in evidence["failure_reasons"])
    assert all("vox pull" not in command[-1] for command in commands)


def test_local_smoke_skips_pull_when_estimate_reports_incompatible_runtime(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        if "repo_info = api.repo_info(source)" in inner:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_download_estimate_stdout(
                    missing=[
                        "requires one of accelerators cuda, which is not available in this environment",
                    ],
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--failure-class",
            "hardware",
            "--failure-note",
            "selected runtime lacks CUDA",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["download_guard_failures"] == [
        "model is incompatible with the detected runtime: "
        "requires one of accelerators cuda, which is not available in this environment",
    ]
    assert evidence["skipped_commands"] == [
        {"label": "pull", "reason": "pre-pull guard failed"},
        {"label": "short synthesis", "reason": "pre-pull guard failed"},
        {"label": "long synthesis", "reason": "pre-pull guard failed"},
    ]
    assert any("model is incompatible with the detected runtime" in reason for reason in evidence["failure_reasons"])
    assert all("vox pull" not in command[-1] for command in commands)


def test_download_guard_does_not_allow_large_download_to_bypass_incompatible_runtime():
    module = _load_local_smoke_module()

    failures = module._download_guard_failures(
        {
            "known_bytes": 50 * 1024 ** 3,
            "known_gib": 50.0,
            "unknown_size_files": ["model.bin"],
            "missing": ["requires one of accelerators cuda, which is not available in this environment"],
        },
        estimate_status=0,
        max_download_gb=1,
        scratch_free_bytes=100 * 1024 ** 3,
        min_free_bytes=50 * 1024 ** 3,
        allow_large_download=True,
    )

    assert failures == [
        "model is incompatible with the detected runtime: "
        "requires one of accelerators cuda, which is not available in this environment",
    ]


def test_local_smoke_skips_pull_when_estimate_would_breach_disk_reserve(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    commands: list[list[str]] = []
    gib = 1024 ** 3

    def fake_disk_snapshot(path):
        return {"total": 100 * gib, "used": 75 * gib, "free": 25 * gib}

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        if "repo_info = api.repo_info(source)" in inner:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_download_estimate_stdout(known_bytes=10 * gib, known_gib=10.0),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(module, "_disk_snapshot", fake_disk_snapshot)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--max-download-gb",
            "20",
            "--min-free-gb",
            "20",
            "--failure-class",
            "dependency",
            "--failure-note",
            "download estimate would breach local disk reserve",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["download_estimate"]["known_gib"] == 10.0
    assert evidence["download_guard_failures"] == [
        "estimated download size 10.00GiB would leave 15.00GiB free, "
        "below --min-free-gb 20; pass --allow-large-download to pull anyway",
    ]
    assert evidence["skipped_commands"] == [
        {"label": "pull", "reason": "pre-pull guard failed"},
        {"label": "short synthesis", "reason": "pre-pull guard failed"},
        {"label": "long synthesis", "reason": "pre-pull guard failed"},
    ]
    assert all("vox pull" not in command[-1] for command in commands)


def test_local_smoke_skips_pull_when_estimate_has_unknown_file_sizes(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fake_run(command, *, capture_output, text, timeout):
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        if "repo_info = api.repo_info(source)" in inner:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_download_estimate_stdout(unknown_size_files=["weights/model.safetensors"]),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--failure-class",
            "dependency",
            "--failure-note",
            "download estimate had unknown file sizes",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["download_estimate"]["unknown_size_files"] == ["weights/model.safetensors"]
    assert evidence["download_guard_failures"] == [
        "download estimate has 1 file(s) with unknown size; "
        "pass --allow-large-download to pull anyway",
    ]
    assert evidence["skipped_commands"][0] == {"label": "pull", "reason": "pre-pull guard failed"}


def test_local_smoke_skips_pull_when_download_estimate_is_not_parseable(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        if "repo_info = api.repo_info(source)" in inner:
            return subprocess.CompletedProcess(command, 0, stdout="not-json", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--failure-class",
            "dependency",
            "--failure-note",
            "download estimate output was not JSON",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["download_estimate"] is None
    assert evidence["download_guard_failures"] == [
        "download size estimate was not parseable; pass --allow-large-download to pull anyway",
    ]
    assert evidence["skipped_commands"][0] == {"label": "pull", "reason": "pre-pull guard failed"}
    assert all("vox pull" not in command[-1] for command in commands)


def test_local_smoke_allows_large_download_after_explicit_override(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        if "repo_info = api.repo_info(source)" in inner:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_download_estimate_stdout(known_gib=25.0),
                stderr="",
            )
        if "vox pull" in inner:
            (scratch / "vox-home" / "runtime" / "dia").mkdir(parents=True)
        elif "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav")
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--allow-large-download",
            "--audio-usable",
            "yes",
            "--expect-runtime",
            "dia",
        ],
    )

    assert module.main() == 0

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["allow_large_download"] is True
    assert evidence["download_estimate"]["known_gib"] == 25.0
    assert evidence["download_guard_failures"] == []
    assert evidence["failure_reasons"] == []
    assert any("vox pull" in command[-1] for command in commands)


def test_local_smoke_skips_pull_when_runtime_snapshot_fails(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="no docker")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--failure-class",
            "hardware",
            "--failure-note",
            "docker runtime was unavailable",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert [command["label"] for command in evidence["commands"]] == [
        "resource snapshot before pull",
        "runtime snapshot",
    ]
    assert evidence["skipped_commands"] == [
        {"label": "download size estimate", "reason": "runtime snapshot failed"},
        {"label": "pre-pull clean state", "reason": "runtime snapshot failed"},
        {"label": "pull", "reason": "runtime snapshot failed"},
        {"label": "short synthesis", "reason": "runtime snapshot failed"},
        {"label": "long synthesis", "reason": "runtime snapshot failed"},
    ]
    assert evidence["failure_reasons"][:6] == [
        "resource snapshot before pull exited 7",
        "runtime snapshot exited 7",
        "download size estimate skipped: runtime snapshot failed",
        "pre-pull clean state skipped: runtime snapshot failed",
        "pull skipped: runtime snapshot failed",
        "short synthesis skipped: runtime snapshot failed",
    ]
    assert "long synthesis skipped: runtime snapshot failed" in evidence["failure_reasons"]
    assert evidence["failure_class"] == "hardware"
    assert evidence["failure_note"] == "docker runtime was unavailable"
    assert len(commands) == 3
    assert commands[0][:3] == ["docker", "image", "inspect"]
    assert all("vox pull" not in command[-1] for command in commands)


def test_local_smoke_skips_pull_when_clean_state_check_fails(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    calls = 0
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        nonlocal calls
        if command[:3] == ["docker", "image", "inspect"]:
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=_image_inspect_stdout(), stderr="")
        calls += 1
        commands.append(command)
        inner = command[-1]
        if "meminfo_bytes()" in inner:
            return subprocess.CompletedProcess(command, 0, stdout=_resource_stdout(), stderr="")
        status = 8 if calls == 4 else 0
        return subprocess.CompletedProcess(command, status, stdout="", stderr="dirty" if status else "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--failure-class",
            "Vox",
            "--failure-note",
            "pre-pull clean-state command failed",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert [command["label"] for command in evidence["commands"]] == [
        "resource snapshot before pull",
        "runtime snapshot",
        "download size estimate",
        "pre-pull clean state",
        "resource snapshot after smoke",
    ]
    assert evidence["skipped_commands"] == [
        {"label": "pull", "reason": "pre-pull clean state failed"},
        {"label": "short synthesis", "reason": "pre-pull clean state failed"},
        {"label": "long synthesis", "reason": "pre-pull clean state failed"},
    ]
    assert evidence["failure_reasons"][:4] == [
        "pre-pull clean state exited 8",
        "pull skipped: pre-pull clean state failed",
        "short synthesis skipped: pre-pull clean state failed",
        "long synthesis skipped: pre-pull clean state failed",
    ]
    assert evidence["failure_class"] == "Vox"
    assert evidence["failure_note"] == "pre-pull clean-state command failed"
    assert len(commands) == 6
    assert commands[0][:3] == ["docker", "image", "inspect"]
    assert all("vox pull" not in command[-1] for command in commands)


def test_local_smoke_fails_when_expected_state_is_missing(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"

    def fake_run(command, *, capture_output, text, timeout):
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        if "vox pull" in inner:
            (scratch / "vox-home" / "adapters" / "vox-dia").mkdir(parents=True)
            (scratch / "vox-home" / "runtime" / "dia").mkdir(parents=True)
        elif "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav")
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--expect-adapter",
            "vox-dia",
            "--expect-runtime",
            "dia",
            "--expect-model-link",
            "dia-tts",
            "--failure-class",
            "adapter",
            "--failure-note",
            "pull did not create the expected model link",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["expected_state"]["model_links"] == ["dia-tts"]
    assert evidence["pre_pull_state"]["adapters"] == []
    assert evidence["pre_pull_state"]["runtime"] == []
    assert evidence["clean_state_failures"] == []
    assert evidence["state_after_pull"]["adapters"] == ["vox-dia"]
    assert evidence["state_after_pull"]["runtime"] == ["dia"]
    assert evidence["state_after_pull"]["model_links"] == []
    assert evidence["state_failures"] == [
        "expected model link missing from post-pull state: dia-tts",
    ]
    assert "expected model link missing from post-pull state: dia-tts" in evidence["failure_reasons"]
    assert evidence["failure_class"] == "adapter"
    assert evidence["failure_note"] == "pull did not create the expected model link"


def test_local_smoke_skips_pull_when_expected_state_already_present(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"
    (scratch / "vox-home" / "runtime" / "dia").mkdir(parents=True)
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        common = _common_success_response(command)
        if common is not None:
            return common
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--expect-runtime",
            "dia",
            "--failure-class",
            "dependency",
            "--failure-note",
            "scratch already had runtime contents",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["pre_pull_state"]["runtime"] == ["dia"]
    assert evidence["clean_state_failures"] == [
        "expected runtime already present before pull: dia",
    ]
    assert evidence["skipped_commands"] == [
        {"label": "pull", "reason": "expected state already present before pull"},
        {"label": "short synthesis", "reason": "expected state already present before pull"},
        {"label": "long synthesis", "reason": "expected state already present before pull"},
    ]
    assert "expected runtime already present before pull: dia" in evidence["failure_reasons"]
    assert evidence["failure_class"] == "dependency"
    assert evidence["failure_note"] == "scratch already had runtime contents"
    assert [command["label"] for command in evidence["commands"]] == [
        "resource snapshot before pull",
        "runtime snapshot",
        "download size estimate",
        "pre-pull clean state",
        "resource snapshot after smoke",
    ]
    assert len(commands) == 6
    assert commands[0][:3] == ["docker", "image", "inspect"]
    assert all("vox pull" not in command[-1] for command in commands)


def test_local_smoke_skips_pull_when_expected_package_metadata_already_present(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"
    dist_info = scratch / "vox-home" / "adapters" / "vox-dia" / "vox_dia-0.2.15.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text("Name: vox-dia\nVersion: 0.2.15\n")
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, text, timeout):
        commands.append(command)
        common = _common_success_response(command)
        if common is not None:
            return common
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--expect-adapter-package",
            "vox-dia==0.2.15",
            "--failure-class",
            "dependency",
            "--failure-note",
            "scratch already had adapter package metadata",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["pre_pull_state"]["adapter_packages"] == {"vox-dia": "0.2.15"}
    assert evidence["clean_state_failures"] == [
        "expected adapter package metadata already present before pull: vox-dia==0.2.15",
    ]
    assert evidence["skipped_commands"] == [
        {"label": "pull", "reason": "expected state already present before pull"},
        {"label": "short synthesis", "reason": "expected state already present before pull"},
        {"label": "long synthesis", "reason": "expected state already present before pull"},
    ]
    assert "expected adapter package metadata already present before pull: vox-dia==0.2.15" in (
        evidence["failure_reasons"]
    )
    assert evidence["failure_class"] == "dependency"
    assert evidence["failure_note"] == "scratch already had adapter package metadata"
    assert [command["label"] for command in evidence["commands"]] == [
        "resource snapshot before pull",
        "runtime snapshot",
        "download size estimate",
        "pre-pull clean state",
        "resource snapshot after smoke",
    ]
    assert len(commands) == 6
    assert all("vox pull" not in command[-1] for command in commands)


def test_local_smoke_cleanup_removes_cache_dirs_but_keeps_evidence_and_audio(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"
    (scratch / "tmp").mkdir(parents=True)
    (scratch / "vox-home" / "runtime" / "dia").mkdir(parents=True)
    (scratch / "vox-home" / "runtime" / "dia" / "runtime.py").write_text("")
    (scratch / "hf-cache" / "hub").mkdir(parents=True)
    (scratch / "hf-cache" / "hub" / "model.bin").write_bytes(b"model")
    _write_wav(scratch / "tmp" / "short.wav")
    _write_wav(scratch / "tmp" / "long.wav")

    def fake_run(command, *, capture_output, text, timeout):
        common = _common_success_response(command)
        if common is not None:
            return common
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--cleanup",
        ],
    )

    assert module.main() == 0

    evidence_path = scratch / "artifacts" / "local-smoke-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence_path.exists()
    assert (scratch / "artifacts" / "short.wav").exists()
    assert (scratch / "artifacts" / "long.wav").exists()
    assert not (scratch / "hf-cache").exists()
    assert not (scratch / "tmp").exists()
    assert str(scratch / "hf-cache") in evidence["cleanup_removed"]
    assert str(scratch / "tmp") in evidence["cleanup_removed"]
    assert evidence["state_after_pull"]["runtime"] == ["dia"]
    assert evidence["failure_class"] == "none"
    assert evidence["failure_note"] == ""
    assert evidence["resource_before"] == {
        "ram": {"total_bytes": 1024, "available_bytes": 512},
        "nvidia_smi_available": False,
        "gpus": [],
    }
    assert evidence["resource_after"] == {
        "ram": {"total_bytes": 1024, "available_bytes": 512},
        "nvidia_smi_available": False,
        "gpus": [],
    }
    assert evidence["storage_after_cleanup"]["hf_cache"] == 0
    assert evidence["storage_after_cleanup"]["tmp"] == 0
    assert evidence["storage_after_cleanup"]["vox_home"] == 0
    assert evidence["storage_after_cleanup"]["artifacts"] > 0


def test_local_smoke_rejects_passing_run_with_failure_class(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"

    def fake_run(command, *, capture_output, text, timeout):
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        if "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav")
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--failure-class",
            "adapter",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["failure_class"] == "adapter"
    assert evidence["failure_reasons"] == ["passing smoke run must use --failure-class none"]


def test_local_smoke_rejects_silent_audio(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"

    def fake_run(command, *, capture_output, text, timeout):
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        if "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav", silent=True)
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav", silent=True)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--failure-class",
            "upstream",
            "--failure-note",
            "model generated silent audio",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert [item["silent"] for item in evidence["audio"]] == [True, True]
    assert evidence["failure_reasons"] == [
        "short.wav is silent",
        "long.wav is silent",
    ]
    assert evidence["failure_class"] == "upstream"
    assert evidence["failure_note"] == "model generated silent audio"


def test_local_smoke_rejects_long_audio_shorter_than_short_audio(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"

    def fake_run(command, *, capture_output, text, timeout):
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        if "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav", frames=4800)
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav", frames=2400)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--failure-class",
            "upstream",
            "--failure-note",
            "long synthesis was shorter than short synthesis",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["failure_reasons"] == [
        "long.wav duration 0.100s is shorter than short.wav duration 0.200s",
    ]
    assert evidence["failure_class"] == "upstream"
    assert evidence["failure_note"] == "long synthesis was shorter than short synthesis"


def test_local_smoke_requires_failure_note_for_classified_failure(tmp_path, monkeypatch):
    module = _load_local_smoke_module()

    def fake_run(command, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="no docker")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "unchecked",
            "--failure-class",
            "hardware",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "dia-tts-1.6b" / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["failure_class"] == "hardware"
    assert "classified failing smoke run must include --failure-note" in evidence["failure_reasons"]


def test_local_smoke_rejects_passing_run_with_failure_note(tmp_path, monkeypatch):
    module = _load_local_smoke_module()
    scratch = tmp_path / "dia-tts-1.6b"

    def fake_run(command, *, capture_output, text, timeout):
        inner = command[-1]
        common = _common_success_response(command)
        if common is not None:
            return common
        if "/tmp/vox-tmp/short.wav" in inner:
            _write_wav(scratch / "tmp" / "short.wav")
        elif "/tmp/vox-tmp/long.wav" in inner:
            _write_wav(scratch / "tmp" / "long.wav")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-local-smoke.py",
            "--model",
            "dia-tts:1.6b",
            "--scratch-root",
            str(tmp_path),
            "--allow-download",
            "--audio-usable",
            "yes",
            "--failure-note",
            "this should not be set on a pass",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((scratch / "artifacts" / "local-smoke-evidence.json").read_text())
    assert evidence["failure_class"] == "none"
    assert evidence["failure_note"] == "this should not be set on a pass"
    assert evidence["failure_reasons"] == ["passing smoke run must not set --failure-note"]


def test_verify_expressive_adapter_evidence_accepts_ready_proof(tmp_path):
    evidence = _ready_evidence(tmp_path)
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PASS model=dia-tts:1.6b proof_target=dia" in result.stdout
    assert result.stderr == ""


def test_verify_expressive_adapter_evidence_accepts_expected_model_and_target(tmp_path):
    evidence = _ready_evidence(tmp_path)
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_verifier_script()),
            "--expect-model",
            "dia-tts:1.6b",
            "--expect-proof-target",
            "dia",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PASS model=dia-tts:1.6b proof_target=dia" in result.stdout
    assert result.stderr == ""


def test_verify_expressive_adapter_evidence_rejects_wrong_model_and_target(tmp_path):
    evidence = _ready_evidence(tmp_path)
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_verifier_script()),
            "--expect-model",
            "indextts-tts:2",
            "--expect-proof-target",
            "indextts",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "model mismatch: expected 'indextts-tts:2', found 'dia-tts:1.6b'" in result.stderr
    assert "proof_target mismatch: expected 'indextts', found 'dia'" in result.stderr


def test_verify_expressive_adapter_evidence_rejects_non_proof(tmp_path):
    evidence = {
        "evidence_schema_version": 2,
        "mode": "local-docker-clean-pull",
        "model": "dia-tts:1.6b",
        "image": "ghcr.io/eleven-am/vox:latest",
        "expected_state": {},
        "commands": [],
        "audio": [],
        "resource_samples": {},
        "proof_ready": False,
        "proof_blockers": [
            "estimate-only evidence is not a synthesis proof",
            "proof requires --allow-download",
        ],
    }
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert f"{path}: FAIL" in result.stderr
    assert "proof_ready is not true" in result.stderr
    assert "proof blocker: estimate-only evidence is not a synthesis proof" in result.stderr


def test_verify_expressive_adapter_evidence_rejects_unknown_schema(tmp_path):
    evidence = _ready_evidence(tmp_path)
    evidence["evidence_schema_version"] = 999
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "unsupported evidence_schema_version 999" in result.stderr


def test_verify_expressive_adapter_evidence_recomputes_core_invariants(tmp_path):
    evidence = _ready_evidence(tmp_path)
    evidence["commands"] = [
        command for command in evidence["commands"]
        if command["label"] != "long synthesis"
    ]
    evidence["audio"][1]["silent"] = True
    evidence["proof_ready"] = True
    evidence["proof_blockers"] = []
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "missing command result: long synthesis" in result.stderr
    assert "long.wav must be non-silent" in result.stderr


def test_verify_expressive_adapter_evidence_requires_goal_checklist(tmp_path):
    evidence = _ready_evidence(tmp_path)
    evidence.pop("goal_checklist")
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "missing evidence field: goal_checklist" in result.stderr
    assert "goal_checklist must be an object" in result.stderr


def test_verify_expressive_adapter_evidence_requires_post_pull_state(tmp_path):
    evidence = _ready_evidence(tmp_path)
    evidence.pop("state_after_pull")
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "state_after_pull must be an object" in result.stderr


def test_verify_expressive_adapter_evidence_requires_pre_pull_state(tmp_path):
    evidence = _ready_evidence(tmp_path)
    evidence.pop("pre_pull_state")
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "missing evidence field: pre_pull_state" in result.stderr
    assert "pre_pull_state must be an object" in result.stderr


def test_verify_expressive_adapter_evidence_recomputes_pre_pull_state(tmp_path):
    evidence = _ready_evidence(tmp_path)
    evidence["pre_pull_state"] = {
        "adapters": ["vox-dia"],
        "adapter_packages": {"vox-dia": "0.2.15"},
        "runtime": ["dia"],
        "manifests": ["dia-tts"],
        "model_links": ["dia-tts"],
    }
    evidence["clean_state_failures"] = []
    evidence["goal_checklist"] = {
        name: {"status": "passed", "evidence": ["test"], "blockers": []}
        for name in evidence["goal_checklist"]
    }
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "pre_pull_state already contains expected adapter package directory: vox-dia" in result.stderr
    assert "pre_pull_state already contains expected runtime directory: dia" in result.stderr
    assert "pre_pull_state already contains expected manifest: dia-tts" in result.stderr
    assert "pre_pull_state already contains expected model link: dia-tts" in result.stderr
    assert "pre_pull_state already contains expected adapter package metadata: vox-dia==0.2.15" in result.stderr


def test_verify_expressive_adapter_evidence_recomputes_post_pull_state(tmp_path):
    evidence = _ready_evidence(tmp_path)
    evidence["state_after_pull"] = {
        "adapters": [],
        "adapter_packages": {},
        "runtime": [],
        "manifests": [],
        "model_links": [],
        "bytes": {"blobs": 0},
    }
    evidence["state_failures"] = []
    evidence["goal_checklist"] = {
        name: {"status": "passed", "evidence": ["test"], "blockers": []}
        for name in evidence["goal_checklist"]
    }
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "state_after_pull missing expected adapter package directory: vox-dia" in result.stderr
    assert "state_after_pull missing expected runtime directory: dia" in result.stderr
    assert "state_after_pull missing expected manifest: dia-tts" in result.stderr
    assert "state_after_pull missing expected model link: dia-tts" in result.stderr
    assert "state_after_pull missing expected adapter package metadata: vox-dia==0.2.15" in result.stderr
    assert "state_after_pull.bytes.blobs must be positive" in result.stderr


def test_verify_expressive_adapter_evidence_rejects_blocked_goal_checklist(tmp_path):
    evidence = _ready_evidence(tmp_path)
    evidence["goal_checklist"]["vox_pull"] = {
        "status": "blocked",
        "evidence": ["commands"],
        "blockers": ["vox pull did not complete successfully"],
    }
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "goal_checklist.vox_pull status is 'blocked', expected 'passed'" in result.stderr
    assert "goal_checklist.vox_pull has blockers" in result.stderr


def test_verify_expressive_adapter_evidence_rejects_missing_audio_artifacts(tmp_path):
    evidence = _ready_evidence(tmp_path)
    Path(evidence["audio"][0]["path"]).unlink()
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "short.wav artifact file is not available for verification" in result.stderr


def test_verify_expressive_adapter_evidence_rejects_audio_hash_mismatch(tmp_path):
    evidence = _ready_evidence(tmp_path)
    Path(evidence["audio"][1]["path"]).write_bytes(b"not the recorded wav")
    path = tmp_path / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_verifier_script()), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "long.wav artifact byte count" in result.stderr
    assert "long.wav artifact sha256 does not match evidence" in result.stderr


def _write_wav(path: Path, *, silent: bool = False, frames: int = 2400) -> None:
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        frame = b"\x00\x00" if silent else b"\x00\x10"
        wav.writeframes(frame * frames)
