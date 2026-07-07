#!/usr/bin/env python3
"""Run clean-pull expressive adapter smoke in a local Docker scratch area.

This helper is intentionally guarded because it can download large model
artifacts. It never calls Kubernetes. It mounts all Vox, Hugging Face, cache,
and temp state under one disposable scratch directory so the caller can inspect
or delete the artifacts after the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_SHORT_TEXT = "This is a short local clean-pull expressive adapter smoke test."
DEFAULT_LONG_TEXT = (
    "This is a longer local clean-pull expressive adapter smoke test. It should "
    "produce usable speech, keep the requested voice behavior, and leave all "
    "runtime and model artifacts under the disposable scratch directory."
)
FAILURE_CLASSES = ("none", "Vox", "adapter", "dependency", "upstream", "hardware")
SAMPLED_COMMAND_LABELS = {"pull", "short synthesis", "long synthesis"}
EVIDENCE_SCHEMA_VERSION = 2
PROOF_TARGETS: dict[str, dict[str, Any]] = {
    "cosyvoice": {
        "model": "cosyvoice2-tts:0.5b",
        "image": "ghcr.io/eleven-am/vox:latest",
        "expect_adapter": ["vox-cosyvoice"],
        "expect_adapter_package": ["vox-cosyvoice==0.1.10"],
        "expect_runtime": ["cosyvoice"],
        "expect_model_link": ["cosyvoice2-tts"],
    },
    "dia": {
        "model": "dia-tts:1.6b",
        "image": "ghcr.io/eleven-am/vox:latest",
        "expect_adapter": ["vox-dia"],
        "expect_adapter_package": ["vox-dia==0.2.15"],
        "expect_runtime": ["dia"],
        "expect_model_link": ["dia-tts"],
    },
    "orpheus": {
        "model": "orpheus-tts:medium-3b",
        "image": "ghcr.io/eleven-am/vox:latest",
        "expect_adapter": ["vox-orpheus"],
        "expect_adapter_package": ["vox-orpheus==0.1.7"],
        "expect_runtime": ["orpheus"],
        "expect_model_link": ["orpheus-tts"],
    },
    "indextts": {
        "model": "indextts-tts:2",
        "image": "ghcr.io/eleven-am/vox:latest",
        "voice": "samantha",
        "expect_adapter": ["vox-indextts"],
        "expect_adapter_package": ["vox-indextts==0.1.21"],
        "expect_runtime": ["indextts"],
        "expect_model_link": ["indextts-tts"],
    },
}


@dataclass(frozen=True)
class CommandResult:
    label: str
    command: list[str]
    status: int
    elapsed_s: float
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SkippedCommand:
    label: str
    reason: str


@dataclass(frozen=True)
class AudioStats:
    path: str
    exists: bool
    bytes: int
    sha256: str | None
    duration_s: float | None
    sample_rate: int | None
    channels: int | None
    sample_width: int | None
    peak: float | None
    rms: float | None
    silent: bool


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _safe_name(model: str, variant: str | None) -> str:
    value = model
    if variant:
        value = f"{value}-{variant}"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)


def _disk_snapshot(path: Path) -> dict[str, int]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def _run_command(label: str, command: list[str], *, timeout: float) -> CommandResult:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return CommandResult(
            label=label,
            command=command,
            status=result.returncode,
            elapsed_s=time.perf_counter() - started,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            label=label,
            command=command,
            status=124,
            elapsed_s=time.perf_counter() - started,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"command timed out after {timeout}s",
        )


def _docker_command(
    *,
    image: str,
    scratch: Path,
    inner_command: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{scratch / 'vox-home'}:/home/vox/.vox",
        "--volume",
        f"{scratch / 'hf-cache'}:/tmp/vox-hf",
        "--volume",
        f"{scratch / 'xdg-cache'}:/tmp/vox-cache",
        "--volume",
        f"{scratch / 'tmp'}:/tmp/vox-tmp",
        "--env",
        "VOX_HOME=/home/vox/.vox",
        "--env",
        "HF_HOME=/tmp/vox-hf",
        "--env",
        "HUGGINGFACE_HUB_CACHE=/tmp/vox-hf/hub",
        "--env",
        "TRANSFORMERS_CACHE=/tmp/vox-hf/transformers",
        "--env",
        "XDG_CACHE_HOME=/tmp/vox-cache",
        "--env",
        "TMPDIR=/tmp/vox-tmp",
        image,
        "sh",
        "-lc",
        inner_command,
    ]


def _sample_path_for_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")
    return f"/tmp/vox-tmp/resource-{safe}.jsonl"


def _sampled_inner_command(inner_command: str, *, label: str, interval_s: float) -> str:
    if interval_s <= 0 or label not in SAMPLED_COMMAND_LABELS:
        return inner_command

    sample_path = _sample_path_for_label(label)
    return (
        "set +e\n"
        "python - <<'PY' &\n"
        "import json\n"
        "import shutil\n"
        "import subprocess\n"
        "import time\n"
        f"sample_path = {sample_path!r}\n"
        f"interval_s = {interval_s!r}\n"
        "\n"
        "def ram_snapshot():\n"
        "    keys = {'MemTotal': 'total_bytes', 'MemAvailable': 'available_bytes'}\n"
        "    result = {}\n"
        "    try:\n"
        "        with open('/proc/meminfo', encoding='utf-8') as handle:\n"
        "            for line in handle:\n"
        "                name, value = line.split(':', 1)\n"
        "                if name in keys:\n"
        "                    fields = value.strip().split()\n"
        "                    if fields:\n"
        "                        result[keys[name]] = int(fields[0]) * 1024\n"
        "    except OSError as exc:\n"
        "        result['error'] = str(exc)\n"
        "    return result\n"
        "\n"
        "def gpu_snapshot():\n"
        "    if shutil.which('nvidia-smi') is None:\n"
        "        return []\n"
        "    query = 'index,memory.total,memory.used,memory.free'\n"
        "    cmd = ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits']\n"
        "    try:\n"
        "        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)\n"
        "    except Exception:\n"
        "        return []\n"
        "    if proc.returncode != 0:\n"
        "        return []\n"
        "    gpus = []\n"
        "    for row in proc.stdout.splitlines():\n"
        "        parts = [part.strip() for part in row.split(',')]\n"
        "        if len(parts) == 4:\n"
        "            try:\n"
        "                gpus.append({\n"
        "                    'index': parts[0],\n"
        "                    'memory_total_mib': int(parts[1]),\n"
        "                    'memory_used_mib': int(parts[2]),\n"
        "                    'memory_free_mib': int(parts[3]),\n"
        "                })\n"
        "            except ValueError:\n"
        "                continue\n"
        "    return gpus\n"
        "\n"
        "while True:\n"
        "    payload = {'at': time.time(), 'ram': ram_snapshot(), 'gpus': gpu_snapshot()}\n"
        "    with open(sample_path, 'a', encoding='utf-8') as handle:\n"
        "        handle.write(json.dumps(payload, sort_keys=True) + '\\n')\n"
        "    time.sleep(interval_s)\n"
        "PY\n"
        "_vox_sampler_pid=$!\n"
        f"{inner_command}\n"
        "_vox_command_status=$?\n"
        "kill $_vox_sampler_pid >/dev/null 2>&1\n"
        "wait $_vox_sampler_pid >/dev/null 2>&1\n"
        "exit $_vox_command_status"
    )


def _sample_file_for_label(scratch: Path, label: str) -> Path:
    return scratch / "tmp" / Path(_sample_path_for_label(label)).name


def _pull_command(model: str, variant: str | None) -> str:
    command = f"vox pull {model!r}"
    if variant:
        command = f"{command} --variant {variant!r}"
    return command


def _run_command_text(model: str, text: str, output_path: str, voice: str | None) -> str:
    if voice:
        return f"vox run {model!r} {text!r} --voice {voice!r} --output {output_path!r}"
    return f"vox run {model!r} {text!r} --output {output_path!r}"


def _voice_looks_like_path(voice: str | None) -> bool:
    return bool(voice and (voice.startswith("/") or voice.startswith("./") or voice.startswith("../")))


def _voice_reference_command(voice: str) -> str:
    quoted = shlex.quote(voice)
    return f"test -f {quoted} && printf '%s\\n' {quoted}"


def _resource_snapshot_command() -> str:
    return (
        "python - <<'PY'\n"
        "import json\n"
        "import shutil\n"
        "import subprocess\n"
        "\n"
        "def meminfo_bytes():\n"
        "    keys = {'MemTotal': 'total_bytes', 'MemAvailable': 'available_bytes'}\n"
        "    result = {}\n"
        "    try:\n"
        "        with open('/proc/meminfo', encoding='utf-8') as handle:\n"
        "            for line in handle:\n"
        "                name, value = line.split(':', 1)\n"
        "                if name in keys:\n"
        "                    fields = value.strip().split()\n"
        "                    if fields:\n"
        "                        result[keys[name]] = int(fields[0]) * 1024\n"
        "    except OSError as exc:\n"
        "        result['error'] = str(exc)\n"
        "    return result\n"
        "\n"
        "payload = {'ram': meminfo_bytes(), 'nvidia_smi_available': shutil.which('nvidia-smi') is not None}\n"
        "if payload['nvidia_smi_available']:\n"
        "    query = 'index,name,memory.total,memory.used,memory.free'\n"
        "    cmd = ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits']\n"
        "    try:\n"
        "        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)\n"
        "        payload['nvidia_smi_status'] = proc.returncode\n"
        "        payload['nvidia_smi_stderr'] = proc.stderr.strip()\n"
        "        gpus = []\n"
        "        for row in proc.stdout.splitlines():\n"
        "            parts = [part.strip() for part in row.split(',')]\n"
        "            if len(parts) == 5:\n"
        "                gpus.append({\n"
        "                    'index': parts[0],\n"
        "                    'name': parts[1],\n"
        "                    'memory_total_mib': int(parts[2]),\n"
        "                    'memory_used_mib': int(parts[3]),\n"
        "                    'memory_free_mib': int(parts[4]),\n"
        "                })\n"
        "        payload['gpus'] = gpus\n"
        "    except Exception as exc:\n"
        "        payload['nvidia_smi_status'] = 1\n"
        "        payload['nvidia_smi_error'] = str(exc)\n"
        "else:\n"
        "    payload['gpus'] = []\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "PY"
    )


def _download_estimate_command(model: str, variant: str | None) -> str:
    return (
        "python - <<'PY'\n"
        "import json\n"
        "import httpx\n"
        "from huggingface_hub import HfApi\n"
        "from vox.core.alias_resolution import resolve_family_alias\n"
        "from vox.core.model_resolution import parse_model_variant_ref, resolve_catalog_entry\n"
        "from vox.core.registry import REGISTRY_BASE_URL\n"
        f"model_ref = {model!r}\n"
        f"variant_ref = {variant!r}\n"
        "payload = {'model': model_ref, 'variant': variant_ref or 'auto'}\n"
        "parsed = parse_model_variant_ref(model_ref)\n"
        "name, tag = resolve_family_alias(parsed.name, parsed.tag, explicit_tag=parsed.explicit_tag)\n"
        "payload['resolved_name'] = name\n"
        "payload['resolved_tag'] = tag\n"
        "url = f'{REGISTRY_BASE_URL}/library/{name}/{tag}.json'\n"
        "payload['registry_url'] = url\n"
        "entry = httpx.get(url, timeout=20, follow_redirects=True).json()\n"
        "resolution = resolve_catalog_entry(entry, forced_variant=variant_ref)\n"
        "payload['resolved_variant'] = resolution.variant_id or ''\n"
        "payload['missing'] = list(resolution.missing)\n"
        "payload['warnings'] = list(resolution.warnings)\n"
        "concrete = resolution.entry\n"
        "source = concrete.get('source')\n"
        "payload['source'] = source\n"
        "requested_files = set(concrete.get('files') or [])\n"
        "api = HfApi()\n"
        "repo_info = api.repo_info(source)\n"
        "files = []\n"
        "known_bytes = 0\n"
        "unknown_size_files = []\n"
        "for sibling in repo_info.siblings:\n"
        "    filename = sibling.rfilename\n"
        "    if filename.startswith('.'):\n"
        "        continue\n"
        "    if requested_files and filename not in requested_files:\n"
        "        continue\n"
        "    size = getattr(sibling, 'size', None)\n"
        "    files.append({'filename': filename, 'size': size})\n"
        "    if isinstance(size, int):\n"
        "        known_bytes += size\n"
        "    else:\n"
        "        unknown_size_files.append(filename)\n"
        "payload['file_count'] = len(files)\n"
        "payload['known_bytes'] = known_bytes\n"
        "payload['known_gib'] = known_bytes / (1024 ** 3)\n"
        "payload['unknown_size_files'] = unknown_size_files\n"
        "payload['files'] = files[:200]\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "PY"
    )


def _download_estimate(
    model: str,
    variant: str | None,
    *,
    registry_get: Any | None = None,
    repo_info_fetcher: Any | None = None,
) -> dict[str, Any]:
    """Estimate selected Hugging Face files without running Docker or downloading them."""

    import httpx
    from huggingface_hub import HfApi

    from vox.core.alias_resolution import resolve_family_alias
    from vox.core.model_resolution import parse_model_variant_ref, resolve_catalog_entry
    from vox.core.registry import REGISTRY_BASE_URL

    payload: dict[str, Any] = {"model": model, "variant": variant or "auto"}
    parsed = parse_model_variant_ref(model)
    name, tag = resolve_family_alias(parsed.name, parsed.tag, explicit_tag=parsed.explicit_tag)
    payload["resolved_name"] = name
    payload["resolved_tag"] = tag
    url = f"{REGISTRY_BASE_URL}/library/{name}/{tag}.json"
    payload["registry_url"] = url

    if registry_get is None:
        def registry_get(target_url: str) -> dict[str, Any]:
            return httpx.get(target_url, timeout=20, follow_redirects=True).json()
    entry = registry_get(url)
    resolution = resolve_catalog_entry(entry, forced_variant=variant)
    payload["resolved_variant"] = resolution.variant_id or ""
    payload["missing"] = list(resolution.missing)
    payload["warnings"] = list(resolution.warnings)
    concrete = resolution.entry
    if not concrete:
        payload["source"] = None
        payload["file_count"] = 0
        payload["known_bytes"] = 0
        payload["known_gib"] = 0.0
        payload["unknown_size_files"] = []
        payload["files"] = []
        return payload

    source = concrete.get("source")
    payload["source"] = source
    requested_files = set(concrete.get("files") or [])
    if repo_info_fetcher is None:
        api = HfApi()
        repo_info_fetcher = api.repo_info
    repo_info = repo_info_fetcher(source)
    files: list[dict[str, Any]] = []
    known_bytes = 0
    unknown_size_files: list[str] = []
    for sibling in repo_info.siblings:
        filename = sibling.rfilename
        if filename.startswith("."):
            continue
        if requested_files and filename not in requested_files:
            continue
        size = getattr(sibling, "size", None)
        files.append({"filename": filename, "size": size})
        if isinstance(size, int):
            known_bytes += size
        else:
            unknown_size_files.append(filename)
    payload["file_count"] = len(files)
    payload["known_bytes"] = known_bytes
    payload["known_gib"] = known_bytes / (1024 ** 3)
    payload["unknown_size_files"] = unknown_size_files
    payload["files"] = files[:200]
    return payload


def _resource_sample_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "samples": 0,
        "peak_ram_used_bytes": None,
        "peak_gpu_memory_used_mib": None,
    }
    if not path.exists():
        return summary

    peak_ram_used: int | None = None
    peak_gpu_used: int | None = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        summary["error"] = str(exc)
        return summary

    for line in lines:
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary["samples"] += 1
        ram = sample.get("ram", {})
        total = ram.get("total_bytes")
        available = ram.get("available_bytes")
        if isinstance(total, int) and isinstance(available, int):
            used = total - available
            peak_ram_used = used if peak_ram_used is None else max(peak_ram_used, used)
        for gpu in sample.get("gpus", []):
            used = gpu.get("memory_used_mib")
            if isinstance(used, int):
                peak_gpu_used = used if peak_gpu_used is None else max(peak_gpu_used, used)

    summary["peak_ram_used_bytes"] = peak_ram_used
    summary["peak_gpu_memory_used_mib"] = peak_gpu_used
    return summary


def _resource_sample_summaries(scratch: Path) -> dict[str, Any]:
    return {
        label: _resource_sample_summary(_sample_file_for_label(scratch, label))
        for label in sorted(SAMPLED_COMMAND_LABELS)
    }


def _parse_json_stdout(result: CommandResult) -> dict[str, Any] | None:
    if result.status != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text.splitlines()[-1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _image_inspect_result(image: str, *, timeout: float) -> CommandResult:
    return _run_command(
        "image inspect",
        ["docker", "image", "inspect", image, "--format", "{{json .}}"],
        timeout=timeout,
    )


def _image_inspect_evidence(result: CommandResult) -> dict[str, Any]:
    parsed = _parse_json_stdout(result)
    evidence: dict[str, Any] = {
        "status": result.status,
        "elapsed_s": result.elapsed_s,
    }
    if result.status != 0:
        evidence["stderr"] = result.stderr
        return evidence
    if parsed is None:
        evidence["stdout"] = result.stdout
        return evidence

    for source, target in (
        ("Id", "id"),
        ("RepoDigests", "repo_digests"),
        ("RepoTags", "repo_tags"),
        ("Architecture", "architecture"),
        ("Os", "os"),
        ("Size", "size_bytes"),
    ):
        if source in parsed:
            evidence[target] = parsed[source]
    return evidence


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
    if not path.exists():
        return AudioStats(
            path=str(path),
            exists=False,
            bytes=0,
            sha256=None,
            duration_s=None,
            sample_rate=None,
            channels=None,
            sample_width=None,
            peak=None,
            rms=None,
            silent=False,
        )

    data = path.read_bytes()
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
            frames = wav.getnframes()
            duration_s = frames / sample_rate if sample_rate else None
            audio_frames = wav.readframes(frames)
            peak, rms = _pcm_stats(audio_frames, sample_width=sample_width)
    except (EOFError, wave.Error):
        pass
    silent = bool(rms is not None and rms < 0.0001)

    return AudioStats(
        path=str(path),
        exists=True,
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        peak=peak,
        rms=rms,
        silent=silent,
    )


def _copy_audio_artifacts(paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in paths:
        target = output_dir / path.name
        if path.exists():
            shutil.copy2(path, target)
        copied.append(target)
    return copied


def _scratch_storage(scratch: Path) -> dict[str, int]:
    return {
        "vox_home": _dir_size(scratch / "vox-home"),
        "hf_cache": _dir_size(scratch / "hf-cache"),
        "xdg_cache": _dir_size(scratch / "xdg-cache"),
        "tmp": _dir_size(scratch / "tmp"),
        "artifacts": _dir_size(scratch / "artifacts"),
    }


def _child_names(path: Path, *, limit: int = 100) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        names.append(child.name)
        if len(names) >= limit:
            break
    return names


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += 1
    return total


def _metadata_value(metadata_path: Path, key: str) -> str | None:
    prefix = f"{key}:"
    try:
        for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    except OSError:
        return None
    return None


def _adapter_package_versions(adapters: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    if not adapters.exists():
        return versions

    for adapter_dir in sorted((child for child in adapters.iterdir() if child.is_dir()), key=lambda item: item.name):
        for dist_info in sorted(adapter_dir.glob("*.dist-info"), key=lambda item: item.name):
            metadata = dist_info / "METADATA"
            name = _metadata_value(metadata, "Name") or adapter_dir.name
            version = _metadata_value(metadata, "Version")
            if version:
                versions[name] = version
                break
    return versions


def _state_snapshot(scratch: Path) -> dict[str, Any]:
    vox_home = scratch / "vox-home"
    adapters = vox_home / "adapters"
    runtime = vox_home / "runtime"
    manifests = vox_home / "manifests"
    model_links = vox_home / "models" / "links"
    blobs = vox_home / "models" / "blobs"
    voices = vox_home / "voices"
    return {
        "adapters": _child_names(adapters),
        "adapter_packages": _adapter_package_versions(adapters),
        "runtime": _child_names(runtime),
        "manifests": _child_names(manifests),
        "model_links": _child_names(model_links),
        "voices": _child_names(voices),
        "file_counts": {
            "adapters": _count_files(adapters),
            "runtime": _count_files(runtime),
            "manifests": _count_files(manifests),
            "model_links": _count_files(model_links),
            "blobs": _count_files(blobs),
            "voices": _count_files(voices),
        },
        "bytes": {
            "adapters": _dir_size(adapters),
            "runtime": _dir_size(runtime),
            "manifests": _dir_size(manifests),
            "model_links": _dir_size(model_links),
            "blobs": _dir_size(blobs),
            "voices": _dir_size(voices),
        },
    }


def _split_expected(values: list[str]) -> list[str]:
    expected: list[str] = []
    for value in values:
        expected.extend(item.strip() for item in value.split(",") if item.strip())
    return expected


def _split_expected_package_specs(values: list[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for spec in _split_expected(values):
        if "==" not in spec:
            raise SystemExit(f"adapter package specs must use NAME==VERSION: {spec}")
        name, version = (part.strip() for part in spec.split("==", 1))
        if not name or not version:
            raise SystemExit(f"adapter package specs must use NAME==VERSION: {spec}")
        specs[name] = version
    return specs


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for item in additions:
        if item not in result:
            result.append(item)
    return result


def _apply_proof_target_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.proof_target:
        if not args.model:
            parser.error("--model is required unless --proof-target is provided")
        if args.image is None:
            args.image = "ghcr.io/eleven-am/vox:lean"
        return

    preset = PROOF_TARGETS[args.proof_target]
    preset_model = str(preset["model"])
    if args.model and args.model != preset_model:
        parser.error(f"--proof-target {args.proof_target} requires --model {preset_model}")
    args.model = preset_model

    if args.image is None:
        args.image = str(preset["image"])
    if args.voice is None and preset.get("voice"):
        args.voice = str(preset["voice"])
    args.expect_adapter = _append_unique(args.expect_adapter, list(preset["expect_adapter"]))
    args.expect_adapter_package = _append_unique(
        args.expect_adapter_package,
        list(preset["expect_adapter_package"]),
    )
    args.expect_runtime = _append_unique(args.expect_runtime, list(preset["expect_runtime"]))
    args.expect_model_link = _append_unique(args.expect_model_link, list(preset["expect_model_link"]))


def _missing_expected_state(
    *,
    state: dict[str, Any],
    expected_adapters: list[str],
    expected_adapter_packages: dict[str, str],
    expected_runtimes: list[str],
    expected_model_links: list[str],
) -> list[str]:
    checks = (
        ("adapter package", "adapters", expected_adapters),
        ("runtime", "runtime", expected_runtimes),
        ("model link", "model_links", expected_model_links),
    )
    reasons: list[str] = []
    for label, key, expected_names in checks:
        present = set(state.get(key, ()))
        for expected in expected_names:
            if expected not in present:
                reasons.append(f"expected {label} missing from post-pull state: {expected}")
    package_versions = state.get("adapter_packages", {})
    for package_name, expected_version in expected_adapter_packages.items():
        actual_version = package_versions.get(package_name)
        if actual_version is None:
            reasons.append(
                "expected adapter package missing from post-pull metadata: "
                f"{package_name}=={expected_version}"
            )
        elif actual_version != expected_version:
            reasons.append(
                f"expected adapter package {package_name}=={expected_version}, found {actual_version}"
            )
    return reasons


def _present_expected_state(
    *,
    state: dict[str, Any],
    expected_adapters: list[str],
    expected_adapter_packages: dict[str, str],
    expected_runtimes: list[str],
    expected_model_links: list[str],
) -> list[str]:
    checks = (
        ("adapter package", "adapters", expected_adapters),
        ("runtime", "runtime", expected_runtimes),
        ("model link", "model_links", expected_model_links),
    )
    reasons: list[str] = []
    for label, key, expected_names in checks:
        present = set(state.get(key, ()))
        for expected in expected_names:
            if expected in present:
                reasons.append(f"expected {label} already present before pull: {expected}")
    package_versions = state.get("adapter_packages", {})
    for package_name, actual_version in package_versions.items():
        if package_name in expected_adapter_packages:
            reasons.append(
                f"expected adapter package metadata already present before pull: "
                f"{package_name}=={actual_version}"
            )
    return reasons


def _cleanup_scratch(scratch: Path) -> list[str]:
    removed: list[str] = []
    for child in ("vox-home", "hf-cache", "xdg-cache", "tmp"):
        path = scratch / child
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def _failure_reasons(
    *,
    commands: list[CommandResult],
    skipped_commands: list[SkippedCommand],
    audio: list[AudioStats],
    audio_usable: str,
    clean_state_failures: list[str],
    state_failures: list[str],
    voice_reference_failures: list[str],
) -> list[str]:
    reasons: list[str] = []
    for command in commands:
        if command.status != 0:
            reasons.append(f"{command.label} exited {command.status}")
    for command in skipped_commands:
        reasons.append(f"{command.label} skipped: {command.reason}")
    reasons.extend(clean_state_failures)
    reasons.extend(state_failures)
    reasons.extend(voice_reference_failures)
    for stats in audio:
        if not stats.exists:
            reasons.append(f"{Path(stats.path).name} was not copied out")
        elif stats.bytes <= 0:
            reasons.append(f"{Path(stats.path).name} is empty")
        elif stats.duration_s is None:
            reasons.append(f"{Path(stats.path).name} has no readable WAV duration")
        elif stats.duration_s <= 0:
            reasons.append(f"{Path(stats.path).name} has non-positive duration")
        elif stats.silent:
            reasons.append(f"{Path(stats.path).name} is silent")
    durations = {Path(stats.path).stem: stats.duration_s for stats in audio if stats.exists}
    short_duration = durations.get("short")
    long_duration = durations.get("long")
    if short_duration is not None and long_duration is not None and long_duration < short_duration:
        reasons.append(
            f"long.wav duration {long_duration:.3f}s is shorter than short.wav duration {short_duration:.3f}s"
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
        if not isinstance(duration, (int, float)) or duration <= 0:
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
        if isinstance(value, dict):
            missing = not value
        elif isinstance(value, list):
            missing = not value
        else:
            missing = True
        if missing:
            blockers.append(f"proof requires {label}")
    return blockers


def _resource_sample_proof_blockers(evidence: dict[str, Any]) -> list[str]:
    interval = evidence.get("resource_sample_interval_s")
    if not isinstance(interval, (int, float)) or interval <= 0:
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


def _proof_readiness(evidence: dict[str, Any]) -> tuple[bool, list[str]]:
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

    skipped = evidence.get("skipped_commands") or []
    for command in skipped:
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
    if not isinstance(duration, (int, float)) or duration <= 0:
        blockers.append("audio artifact has no positive duration")
    if item.get("silent") is not False:
        blockers.append("audio artifact is silent or silence was not checked")
    return blockers


def _goal_check_status(blockers: list[str]) -> str:
    return "passed" if not blockers else "blocked"


def _goal_checklist(evidence: dict[str, Any]) -> dict[str, Any]:
    """Map local clean-pull evidence to the goal's model-readiness checklist."""

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

    state_blockers: list[str] = []
    state_blockers.extend(_expected_state_proof_blockers(evidence))
    state_blockers.extend(str(reason) for reason in evidence.get("state_failures") or [])
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
                item for item in evidence.get("commands", [])
                if isinstance(item, dict) and item.get("label") == label
            ),
            None,
        )
        elapsed = command.get("elapsed_s") if isinstance(command, dict) else None
        if not isinstance(elapsed, (int, float)) or elapsed < 0:
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


def _download_guard_failures(
    estimate: dict[str, Any] | None,
    *,
    estimate_status: int,
    max_download_gb: float,
    scratch_free_bytes: int,
    min_free_bytes: int,
    allow_large_download: bool,
) -> list[str]:
    if estimate_status != 0:
        if allow_large_download:
            return []
        return ["download size estimate failed; pass --allow-large-download to pull anyway"]
    if estimate is None:
        if allow_large_download:
            return []
        return ["download size estimate was not parseable; pass --allow-large-download to pull anyway"]

    failures: list[str] = []
    missing = estimate.get("missing") or []
    if missing:
        failures.append(
            "model is incompatible with the detected runtime: "
            + "; ".join(str(item) for item in missing)
        )
    if allow_large_download:
        return failures
    unknown = estimate.get("unknown_size_files") or []
    if unknown:
        failures.append(
            f"download estimate has {len(unknown)} file(s) with unknown size; "
            "pass --allow-large-download to pull anyway"
        )
    known_gib = estimate.get("known_gib")
    if isinstance(known_gib, (int, float)) and known_gib > max_download_gb:
        failures.append(
            f"estimated download size {known_gib:.2f}GiB exceeds --max-download-gb {max_download_gb:g}; "
            "pass --allow-large-download to pull anyway"
        )
    known_bytes = estimate.get("known_bytes")
    if isinstance(known_bytes, int) and known_bytes > 0:
        free_after_known_download = scratch_free_bytes - known_bytes
        if free_after_known_download < min_free_bytes:
            known_download_gib = known_bytes / (1024 ** 3)
            min_free_gib = min_free_bytes / (1024 ** 3)
            free_after_gib = free_after_known_download / (1024 ** 3)
            failures.append(
                f"estimated download size {known_download_gib:.2f}GiB would leave "
                f"{free_after_gib:.2f}GiB free, below --min-free-gb {min_free_gib:g}; "
                "pass --allow-large-download to pull anyway"
            )
    return failures


def _write_evidence(evidence: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "local-smoke-evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"Evidence written to {path}", file=sys.stderr)
    return path


def _finalize_evidence(
    evidence: dict[str, Any],
    *,
    scratch: Path,
    output_dir: Path,
    cleanup: bool,
) -> Path:
    evidence.setdefault("storage_before_cleanup", _scratch_storage(scratch))
    evidence.setdefault("disk_after", _disk_snapshot(scratch))
    evidence["cleanup_removed"] = _cleanup_scratch(scratch) if cleanup else []
    evidence["storage_after_cleanup"] = _scratch_storage(scratch)
    evidence["disk_after_cleanup"] = _disk_snapshot(scratch)
    proof_ready, proof_blockers = _proof_readiness(evidence)
    evidence["proof_ready"] = proof_ready
    evidence["proof_blockers"] = proof_blockers
    evidence["goal_checklist"] = _goal_checklist(evidence)
    return _write_evidence(evidence, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument(
        "--proof-target",
        choices=tuple(PROOF_TARGETS),
        help=(
            "Apply the approved clean-pull proof defaults for one remaining "
            "expressive adapter target. Explicit flags still override optional defaults."
        ),
    )
    parser.add_argument("--variant", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--scratch-root", default="/tmp/vox-adapter-lab")
    parser.add_argument("--short-text", default=DEFAULT_SHORT_TEXT)
    parser.add_argument("--long-text", default=DEFAULT_LONG_TEXT)
    parser.add_argument("--voice", default=None)
    parser.add_argument(
        "--expect-adapter",
        action="append",
        default=[],
        help="Expected child name under VOX_HOME/adapters after pull. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--expect-adapter-package",
        action="append",
        default=[],
        help="Expected adapter package metadata as NAME==VERSION after pull. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--expect-runtime",
        action="append",
        default=[],
        help="Expected child name under VOX_HOME/runtime after pull. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--expect-model-link",
        action="append",
        default=[],
        help="Expected child name under VOX_HOME/models/links after pull. May be repeated or comma-separated.",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument(
        "--max-download-gb",
        type=float,
        default=20.0,
        help=(
            "Maximum known Hugging Face download size before the helper refuses to pull. "
            "Use --allow-large-download after reviewing the estimate."
        ),
    )
    parser.add_argument(
        "--resource-sample-interval",
        type=float,
        default=1.0,
        help=(
            "Seconds between in-container RAM/VRAM samples for pull and synthesis commands. "
            "Use 0 to disable continuous resource sampling."
        ),
    )
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
        "--cleanup",
        action="store_true",
        help="Remove model/cache/temp scratch directories after writing evidence and copied WAV artifacts.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help=(
            "Resolve the model and estimate selected Hugging Face file sizes on the host, "
            "then exit before Docker or vox pull. This does not prove the model works."
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Required before running docker/vox pull because this can download large artifacts.",
    )
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help=(
            "Allow pull even when the pre-pull download estimate is large, unknown, "
            "unavailable, or would breach the configured free-space reserve."
        ),
    )
    args = parser.parse_args()
    _apply_proof_target_defaults(args, parser)

    safe = _safe_name(args.model, args.variant)
    scratch = Path(args.scratch_root).expanduser() / safe
    output_dir = scratch / "artifacts"
    before_disk = _disk_snapshot(scratch)
    min_free_bytes = int(args.min_free_gb * 1024 * 1024 * 1024)
    expected_adapters = _split_expected(args.expect_adapter)
    expected_adapter_packages = _split_expected_package_specs(args.expect_adapter_package)
    expected_runtimes = _split_expected(args.expect_runtime)
    expected_model_links = _split_expected(args.expect_model_link)

    evidence: dict[str, Any] = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "mode": "local-docker-clean-pull",
        "proof_target": args.proof_target,
        "model": args.model,
        "variant": args.variant or "auto",
        "image": args.image,
        "voice": args.voice or None,
        "scratch": str(scratch),
        "output_dir": str(output_dir),
        "disk_before": before_disk,
        "min_free_gb": args.min_free_gb,
        "max_download_gb": args.max_download_gb,
        "resource_sample_interval_s": args.resource_sample_interval,
        "audio_usable": args.audio_usable,
        "failure_class": args.failure_class,
        "failure_note": args.failure_note,
        "expected_state": {
            "adapters": expected_adapters,
            "adapter_packages": expected_adapter_packages,
            "runtime": expected_runtimes,
            "model_links": expected_model_links,
        },
        "allow_download": args.allow_download,
        "allow_large_download": args.allow_large_download,
        "estimate_only": args.estimate_only,
        "cleanup_requested": args.cleanup,
    }

    if before_disk["free"] < min_free_bytes:
        failure_reasons = [
            f"scratch filesystem has less than {args.min_free_gb:g}GiB free"
        ]
        evidence["failure_reasons"] = [
            *failure_reasons,
            *_failure_classification_reasons(
                failure_reasons=failure_reasons,
                failure_class=args.failure_class,
                failure_note=args.failure_note,
            ),
        ]
        _finalize_evidence(evidence, scratch=scratch, output_dir=output_dir, cleanup=args.cleanup)
        return 1

    if args.estimate_only:
        try:
            download_estimate = _download_estimate(args.model, args.variant)
            estimate_status = 0
        except Exception as exc:
            download_estimate = None
            estimate_status = 1
            evidence["download_estimate_error"] = str(exc)
        evidence["download_estimate"] = download_estimate
        evidence["download_guard_failures"] = _download_guard_failures(
            download_estimate,
            estimate_status=estimate_status,
            max_download_gb=args.max_download_gb,
            scratch_free_bytes=before_disk["free"],
            min_free_bytes=min_free_bytes,
            allow_large_download=args.allow_large_download,
        )
        evidence["would_skip_pull"] = bool(evidence["download_guard_failures"])
        evidence["failure_reasons"] = (
            ["download estimate failed"] if estimate_status != 0 else []
        )
        _finalize_evidence(evidence, scratch=scratch, output_dir=output_dir, cleanup=args.cleanup)
        return 1 if estimate_status != 0 else 0

    if not args.allow_download:
        failure_reasons = ["--allow-download is required before running clean-pull smoke"]
        evidence["failure_reasons"] = [
            *failure_reasons,
            *_failure_classification_reasons(
                failure_reasons=failure_reasons,
                failure_class=args.failure_class,
                failure_note=args.failure_note,
            ),
        ]
        _finalize_evidence(evidence, scratch=scratch, output_dir=output_dir, cleanup=args.cleanup)
        return 1

    for child in ("vox-home", "hf-cache", "xdg-cache", "tmp", "artifacts"):
        (scratch / child).mkdir(parents=True, exist_ok=True)

    image_inspect = _image_inspect_result(args.image, timeout=min(args.timeout, 300.0))
    evidence["image_inspect"] = _image_inspect_evidence(image_inspect)

    commands: list[CommandResult] = []
    skipped_commands: list[SkippedCommand] = []
    voice_reference_failures: list[str] = []
    if _voice_looks_like_path(args.voice):
        voice_result = _run_command(
            "voice reference check",
            _docker_command(
                image=args.image,
                scratch=scratch,
                inner_command=_voice_reference_command(args.voice or ""),
            ),
            timeout=min(args.timeout, 300.0),
        )
        commands.append(voice_result)
        evidence["voice_reference"] = {
            "path": args.voice,
            "path_like": True,
            "exists": voice_result.status == 0,
            "status": voice_result.status,
            "stdout": voice_result.stdout,
            "stderr": voice_result.stderr,
        }
        if voice_result.status != 0:
            voice_reference_failures.append(
                f"voice reference path missing inside container: {args.voice}"
            )
    else:
        evidence["voice_reference"] = {
            "path": args.voice,
            "path_like": False,
            "exists": None,
        }

    resource_before = _run_command(
        "resource snapshot before pull",
        _docker_command(
            image=args.image,
            scratch=scratch,
            inner_command=_resource_snapshot_command(),
        ),
        timeout=min(args.timeout, 300.0),
    )
    commands.append(resource_before)
    evidence["resource_before"] = _parse_json_stdout(resource_before)

    runtime_result = _run_command(
        "runtime snapshot",
        _docker_command(
            image=args.image,
            scratch=scratch,
            inner_command=(
                "python - <<'PY'\n"
                "from vox.core.runtime import detect_runtime_capabilities\n"
                "print(detect_runtime_capabilities())\n"
                "PY"
            ),
        ),
        timeout=min(args.timeout, 300.0),
    )
    commands.append(runtime_result)

    if runtime_result.status != 0:
        skipped_commands.extend((
            SkippedCommand(label="download size estimate", reason="runtime snapshot failed"),
            SkippedCommand(label="pre-pull clean state", reason="runtime snapshot failed"),
            SkippedCommand(label="pull", reason="runtime snapshot failed"),
            SkippedCommand(label="short synthesis", reason="runtime snapshot failed"),
            SkippedCommand(label="long synthesis", reason="runtime snapshot failed"),
        ))
    elif voice_reference_failures:
        skipped_commands.extend((
            SkippedCommand(label="download size estimate", reason="voice reference check failed"),
            SkippedCommand(label="pre-pull clean state", reason="voice reference check failed"),
            SkippedCommand(label="pull", reason="voice reference check failed"),
            SkippedCommand(label="short synthesis", reason="voice reference check failed"),
            SkippedCommand(label="long synthesis", reason="voice reference check failed"),
        ))
    else:
        estimate_result = _run_command(
            "download size estimate",
            _docker_command(
                image=args.image,
                scratch=scratch,
                inner_command=_download_estimate_command(args.model, args.variant),
            ),
            timeout=min(args.timeout, 300.0),
        )
        commands.append(estimate_result)
        download_estimate = _parse_json_stdout(estimate_result)
        evidence["download_estimate"] = download_estimate
        download_guard_failures = _download_guard_failures(
            download_estimate,
            estimate_status=estimate_result.status,
            max_download_gb=args.max_download_gb,
            scratch_free_bytes=before_disk["free"],
            min_free_bytes=min_free_bytes,
            allow_large_download=args.allow_large_download,
        )

        pre_pull_result = _run_command(
            "pre-pull clean state",
            _docker_command(
                image=args.image,
                scratch=scratch,
                inner_command="find /home/vox/.vox -maxdepth 3 -mindepth 1 -print | sort",
            ),
            timeout=min(args.timeout, 300.0),
        )
        commands.append(pre_pull_result)

        if pre_pull_result.status != 0:
            skipped_commands.extend((
                SkippedCommand(label="pull", reason="pre-pull clean state failed"),
                SkippedCommand(label="short synthesis", reason="pre-pull clean state failed"),
                SkippedCommand(label="long synthesis", reason="pre-pull clean state failed"),
            ))
        else:
            pre_pull_state = _state_snapshot(scratch)
            evidence["pre_pull_state"] = pre_pull_state
            clean_state_failures = _present_expected_state(
                state=pre_pull_state,
                expected_adapters=expected_adapters,
                expected_adapter_packages=expected_adapter_packages,
                expected_runtimes=expected_runtimes,
                expected_model_links=expected_model_links,
            )
            evidence["clean_state_failures"] = clean_state_failures
            if download_guard_failures:
                evidence["download_guard_failures"] = download_guard_failures
                skipped_commands.extend((
                    SkippedCommand(label="pull", reason="pre-pull guard failed"),
                    SkippedCommand(label="short synthesis", reason="pre-pull guard failed"),
                    SkippedCommand(label="long synthesis", reason="pre-pull guard failed"),
                ))
            elif clean_state_failures:
                skipped_commands.extend((
                    SkippedCommand(label="pull", reason="expected state already present before pull"),
                    SkippedCommand(label="short synthesis", reason="expected state already present before pull"),
                    SkippedCommand(label="long synthesis", reason="expected state already present before pull"),
                ))
            else:
                pull_result = _run_command(
                    "pull",
                    _docker_command(
                        image=args.image,
                        scratch=scratch,
                        inner_command=_sampled_inner_command(
                            _pull_command(args.model, args.variant),
                            label="pull",
                            interval_s=args.resource_sample_interval,
                        ),
                    ),
                    timeout=args.timeout,
                )
                commands.append(pull_result)
                if pull_result.status == 0:
                    commands.append(_run_command(
                        "short synthesis",
                        _docker_command(
                            image=args.image,
                            scratch=scratch,
                            inner_command=_sampled_inner_command(
                                _run_command_text(
                                    args.model,
                                    args.short_text,
                                    "/tmp/vox-tmp/short.wav",
                                    args.voice,
                                ),
                                label="short synthesis",
                                interval_s=args.resource_sample_interval,
                            ),
                        ),
                        timeout=args.timeout,
                    ))
                    commands.append(_run_command(
                        "long synthesis",
                        _docker_command(
                            image=args.image,
                            scratch=scratch,
                            inner_command=_sampled_inner_command(
                                _run_command_text(
                                    args.model,
                                    args.long_text,
                                    "/tmp/vox-tmp/long.wav",
                                    args.voice,
                                ),
                                label="long synthesis",
                                interval_s=args.resource_sample_interval,
                            ),
                        ),
                        timeout=args.timeout,
                    ))
                else:
                    skipped_commands.extend((
                        SkippedCommand(label="short synthesis", reason="pull failed"),
                        SkippedCommand(label="long synthesis", reason="pull failed"),
                    ))

    if resource_before.status == 0:
        resource_after = _run_command(
            "resource snapshot after smoke",
            _docker_command(
                image=args.image,
                scratch=scratch,
                inner_command=_resource_snapshot_command(),
            ),
            timeout=min(args.timeout, 300.0),
        )
        commands.append(resource_after)
        evidence["resource_after"] = _parse_json_stdout(resource_after)
    else:
        evidence["resource_after"] = None

    short_audio = scratch / "tmp" / "short.wav"
    long_audio = scratch / "tmp" / "long.wav"
    copied_audio = _copy_audio_artifacts([short_audio, long_audio], output_dir)
    audio = [_audio_stats(path) for path in copied_audio]
    evidence.setdefault("pre_pull_state", _state_snapshot(scratch))
    clean_state_failures = evidence.setdefault("clean_state_failures", [])
    download_guard_failures = evidence.setdefault("download_guard_failures", [])
    evidence["commands"] = [asdict(command) for command in commands]
    evidence["skipped_commands"] = [asdict(command) for command in skipped_commands]
    evidence["audio"] = [asdict(stats) for stats in audio]
    state_after_pull = _state_snapshot(scratch)
    evidence["state_after_pull"] = state_after_pull
    state_failures = _missing_expected_state(
        state=state_after_pull,
        expected_adapters=expected_adapters,
        expected_adapter_packages=expected_adapter_packages,
        expected_runtimes=expected_runtimes,
        expected_model_links=expected_model_links,
    )
    evidence["state_failures"] = state_failures
    evidence["voice_reference_failures"] = voice_reference_failures
    evidence["resource_samples"] = _resource_sample_summaries(scratch)
    evidence["storage_before_cleanup"] = _scratch_storage(scratch)
    evidence["disk_after"] = _disk_snapshot(scratch)
    failure_reasons = _failure_reasons(
        commands=commands,
        skipped_commands=skipped_commands,
        audio=audio,
        audio_usable=args.audio_usable,
        clean_state_failures=clean_state_failures,
        state_failures=[*download_guard_failures, *state_failures],
        voice_reference_failures=voice_reference_failures,
    )
    evidence["failure_reasons"] = [
        *failure_reasons,
        *_failure_classification_reasons(
            failure_reasons=failure_reasons,
            failure_class=args.failure_class,
            failure_note=args.failure_note,
        ),
    ]
    _finalize_evidence(evidence, scratch=scratch, output_dir=output_dir, cleanup=args.cleanup)

    return 1 if evidence["failure_reasons"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
