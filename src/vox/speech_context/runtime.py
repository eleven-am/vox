from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vox.core.adapter_runtime import runtime_root
from vox.core.worker_host import WorkerHost

RUNTIME_SCHEMA_VERSION = 2
SENSEVOICE_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
)
SENSEVOICE_ARCHIVE_SHA256 = "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e"
SENSEVOICE_ARCHIVE_ROOT = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
YAMNET_MODEL_URL = "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite"
YAMNET_MODEL_SHA256 = "141fba1cdaae842c816f28edc4937e8b4f0af4c8df21862ccc6b52dc567993c3"
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "950c21457b3b3de045cb2b907e973c744e743af9/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)
YAMNET_CLASS_MAP_SHA256 = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"
AUDIOSET_ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/v1/ontology.json"
AUDIOSET_ONTOLOGY_SHA256 = "9c685f4403eecc3ca9be37fd7285cf212feaaea6ff7229d3e7ca89e0d1f2d15d"


class SpeechContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeSpec:
    key: str
    directory: str
    requirements_file: str
    module: str
    license: str
    asset_revision: str
    model_file: str | None = None
    no_deps: bool = False
    required_files: tuple[str, ...] = ()
    required_globs: tuple[str, ...] = ()


RUNTIME_SPECS = {
    "speaker": RuntimeSpec(
        key="speaker",
        directory="speech-context-speaker",
        requirements_file="requirements-sensevoice.txt",
        module="vox.speech_context.sensevoice_worker",
        license="sherpa-onnx Apache-2.0; SenseVoice FunASR Model License",
        asset_revision=SENSEVOICE_ARCHIVE_SHA256,
        model_file="assets/model.int8.onnx",
        no_deps=True,
        required_files=(
            "lib/python3.12/site-packages/numpy/__init__.py",
            "lib/python3.12/site-packages/sherpa_onnx/__init__.py",
            "assets/model.int8.onnx",
            "assets/tokens.txt",
        ),
        required_globs=("lib/python3.12/site-packages/sherpa_onnx/lib/_sherpa_onnx*.so",),
    ),
    "sounds": RuntimeSpec(
        key="sounds",
        directory="speech-context-audio-events",
        requirements_file="requirements-yamnet.txt",
        module="vox.speech_context.yamnet_worker",
        license="YAMNet Apache-2.0; AudioSet ontology CC BY-SA 4.0",
        asset_revision=":".join(
            (
                YAMNET_MODEL_SHA256,
                YAMNET_CLASS_MAP_SHA256,
                AUDIOSET_ONTOLOGY_SHA256,
            )
        ),
        model_file="assets/yamnet.tflite",
        no_deps=True,
        required_files=(
            "lib/python3.12/site-packages/numpy/__init__.py",
            "lib/python3.12/site-packages/ai_edge_litert/interpreter.py",
            "assets/yamnet.tflite",
            "assets/yamnet_class_map.csv",
            "assets/audioset_ontology.json",
        ),
    ),
}


def package_import_root() -> Path:
    return Path(__file__).resolve().parents[2]


def requirements_path(spec: RuntimeSpec) -> Path:
    return Path(__file__).with_name("assets") / spec.requirements_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            with contextlib.suppress(OSError):
                total += item.stat().st_size
    return total


def marker_payload(spec: RuntimeSpec) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "requirements_sha256": sha256_file(requirements_path(spec)),
        "python": "3.12",
        "license": spec.license,
        "asset_revision": spec.asset_revision,
    }


def runtime_path(spec: RuntimeSpec, *, home: Path | None = None) -> Path:
    return runtime_root(home=home) / spec.directory


def runtime_is_ready(spec: RuntimeSpec, path: Path) -> bool:
    marker_path = path / ".vox-speech-context-runtime.json"
    python = path / "bin" / "python"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        python.is_file()
        and marker == marker_payload(spec)
        and all((path / relative).is_file() for relative in spec.required_files)
        and all(any(path.glob(pattern)) for pattern in spec.required_globs)
    )


def run_install(command: list[str], *, env: dict[str, str], timeout: int = 900) -> None:
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise SpeechContextError(f"runtime installation failed: {' '.join(command)}: {detail}")


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise SpeechContextError(
            f"download checksum mismatch for {destination.name}: expected {expected_sha256}, got {actual}"
        )


def extract_archive_files(
    archive_path: Path,
    destination: Path,
    members: dict[str, str],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        by_name = {member.name: member for member in archive.getmembers()}
        for source_name, destination_name in members.items():
            member = by_name.get(source_name)
            if member is None or not member.isfile():
                raise SpeechContextError(f"required model asset is missing from archive: {source_name}")
            source = archive.extractfile(member)
            if source is None:
                raise SpeechContextError(f"could not read model asset: {source_name}")
            with source, (destination / destination_name).open("wb") as output:
                shutil.copyfileobj(source, output)


def install_runtime_assets(spec: RuntimeSpec, stage: Path) -> None:
    if spec.key == "speaker":
        archive = stage / "sensevoice.tar.bz2"
        download_verified(SENSEVOICE_ARCHIVE_URL, archive, SENSEVOICE_ARCHIVE_SHA256)
        try:
            extract_archive_files(
                archive,
                stage / "assets",
                {
                    f"{SENSEVOICE_ARCHIVE_ROOT}/model.int8.onnx": "model.int8.onnx",
                    f"{SENSEVOICE_ARCHIVE_ROOT}/tokens.txt": "tokens.txt",
                },
            )
        finally:
            archive.unlink(missing_ok=True)
        return

    if spec.key == "sounds":
        assets = stage / "assets"
        assets.mkdir()
        download_verified(YAMNET_MODEL_URL, assets / "yamnet.tflite", YAMNET_MODEL_SHA256)
        download_verified(
            YAMNET_CLASS_MAP_URL,
            assets / "yamnet_class_map.csv",
            YAMNET_CLASS_MAP_SHA256,
        )
        download_verified(
            AUDIOSET_ONTOLOGY_URL,
            assets / "audioset_ontology.json",
            AUDIOSET_ONTOLOGY_SHA256,
        )
        return

    raise SpeechContextError(f"unsupported speech-context runtime: {spec.key}")


def install_runtime(spec: RuntimeSpec, *, home: Path | None = None) -> dict[str, Any]:
    target = runtime_path(spec, home=home)
    if runtime_is_ready(spec, target):
        return runtime_details(spec, target)
    if target.exists():
        raise SpeechContextError(
            f"{target} exists but does not match the speech-context runtime lock; remove it explicitly"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{target.name}.installing-{os.getpid()}")
    if stage.exists():
        raise SpeechContextError(f"stale installation directory exists: {stage}")
    cache = target.with_name(f".{target.name}.uv-cache-{os.getpid()}")
    if cache.exists():
        raise SpeechContextError(f"stale installation cache exists: {cache}")
    env = {**os.environ, "UV_CACHE_DIR": str(cache)}
    try:
        run_install(["uv", "venv", str(stage), "--python", "3.12"], env=env, timeout=300)
        run_install(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(stage / "bin" / "python"),
                *(["--no-deps"] if spec.no_deps else []),
                "--requirement",
                str(requirements_path(spec)),
            ],
            env=env,
        )
        install_runtime_assets(spec, stage)
        (stage / ".vox-speech-context-runtime.json").write_text(
            json.dumps(marker_payload(spec), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return runtime_details(spec, target)


def install_speech_context_runtimes(
    *,
    home: Path | None = None,
) -> dict[str, Any]:
    return {key: install_runtime(spec, home=home) for key, spec in RUNTIME_SPECS.items()}


def runtime_details(spec: RuntimeSpec, path: Path) -> dict[str, Any]:
    model_path = path / spec.model_file if spec.model_file else None
    return {
        "status": "ready" if runtime_is_ready(spec, path) else "missing",
        "path": str(path),
        "runtime_bytes": tree_bytes(path),
        "model_bytes": model_path.stat().st_size if model_path and model_path.is_file() else 0,
        "license": spec.license,
    }


def runtime_inventory(*, home: Path | None = None) -> dict[str, Any]:
    return {key: runtime_details(spec, runtime_path(spec, home=home)) for key, spec in RUNTIME_SPECS.items()}


def worker_environment(spec: RuntimeSpec, runtime: Path) -> dict[str, str]:
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "PATH",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
        }
    }
    allowed["PYTHONPATH"] = str(package_import_root())
    allowed["PYTHONDONTWRITEBYTECODE"] = "1"
    allowed["VOX_SPEECH_CONTEXT_ASSETS"] = str(runtime / "assets")
    return allowed


def create_worker_host(
    spec: RuntimeSpec,
    *,
    home: Path | None = None,
    startup_timeout: float,
) -> WorkerHost:
    runtime = runtime_path(spec, home=home)
    if not runtime_is_ready(spec, runtime):
        raise SpeechContextError(f"{spec.key} runtime is not installed; run vox speech-context install first")
    return WorkerHost(
        [str(runtime / "bin" / "python"), "-m", spec.module],
        env=worker_environment(spec, runtime),
        name=f"speech-context-{spec.key}",
        startup_timeout=startup_timeout,
    )
