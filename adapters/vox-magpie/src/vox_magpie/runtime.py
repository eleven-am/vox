from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import httpx

from vox.core.adapter_runtime import runtime_root, staged_target_runtime

RUNTIME_NAME = "magpie-native"
NEMO_SPEECH_VERSION = "0.2.0"
NEMO_SPEECH_ARCHIVE = f"nemo-speech-{NEMO_SPEECH_VERSION}-linux-x86_64-cpu.tar.gz"
NEMO_SPEECH_URL = (
    f"https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{NEMO_SPEECH_VERSION}/"
    f"{NEMO_SPEECH_ARCHIVE}"
)
NEMO_SPEECH_SHA256 = "f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507"

MAGPIE_REPOSITORY = "nvidia/magpie_tts_multilingual_357m"
MAGPIE_REVISION = "19806879b16d3f2ccf28fb112b1bcd16a3c7923e"
TOKENIZER_ARCHIVE = "magpie_tts_multilingual_357m.nemo"
TOKENIZER_URL = (
    f"https://huggingface.co/{MAGPIE_REPOSITORY}/resolve/{MAGPIE_REVISION}/{TOKENIZER_ARCHIVE}"
)
TOKENIZER_SELECTED_SIZE = 31_076_864
TOKENIZER_SELECTED_SHA256 = "169144a64f6d775178696914e5aad8282db44820e73341d5f9a4e7865fac1f26"
TOKENIZER_RANGES = (
    (0, 20_619_263, "model_weights.ckpt"),
    (1_459_750_400, 1_470_207_999, None),
)
TOKENIZER_MEMBERS = {
    "05adc40366e149b69319acd4b28a4919_pt_br_prondict-v1.0.dict": (
        3_580_284,
        "6492abc404db16bbad83dd9d7f9a60eb617699f0a3ff54b3be6184e14507b745",
    ),
    "339da71c54b046f98cbcf38ef6d4ff67_hindi_phoneme_merged_phoneme_dict.dict": (
        7_319_864,
        "3b50cc2c21cd8d34983d8492c91308e9db4a58c1194d6506fdbb2b97aead75fc",
    ),
    "41913ebaa70342058574da74293b7630_magpie_tts_multilingual_357m.nemo.speakers.json": (
        68,
        "36cdcf01ebc0afb506660ac71f2d0a211236374ad5c872d7d8d985a3f9f6ccaf",
    ),
    "61d57a4ccb064d1e8e3d9f871c571932_heteronyms-052722": (
        1_606,
        "b701909aedf753172eff223950f8859cd4b9b4c80199cf0a6e9ac4a307c8f8ec",
    ),
    "74b3e832fe914a569fcd42b51d06b2f1_ipa_dict_nv23.05.txt": (
        5_419,
        "252e8eaf60dfd891520759913dc8534393eebfa8d57192365f76f9229f294e9e",
    ),
    "7dbc31751f224f2486090d59dc95b9f7_es_ES_nv230301.dict": (
        2_230_910,
        "94c9a25bb359f733cd863c887d0112e7ae19a744b0333ef64b6a88e576428669",
    ),
    "c5e4ec2af5a14ce294f4b9edcc936535_de_nv230119.heteronym": (
        44_566,
        "771cc585a574fd35bd14f4ce6108edf1e0e512a8dbc810db7de4c1165ba9d0ef",
    ),
    "cf01ab5c48c84f3282ef7888263361e5_de_nv230119.dict": (
        4_313_907,
        "5c7bbf3346ebd6dc57769b5cc805124215cb1bcc3b8174c4254e9e928c5d6094",
    ),
    "dc7d60d6b15a4651b21c9ca2932b62c6_ipa_cmudict-0.7b_nv23.01.txt": (
        3_093_097,
        "7ea5c4c2c59cc780748b24d4ab60bd689779c4ac4ea7e27c0a8c3e20f300c0fa",
    ),
    "model_config.yaml": (
        8_358,
        "9577bfa4769981849feccd6a54d38425d81fff41ac16c282f60be97faf46ebe4",
    ),
}
_SENTINEL = "runtime.json"
_REQUIRED_NATIVE_LIBRARIES = (
    "libnemo_speech_tts.so",
    "libnemo_speech_text_normalization.so",
    "libggml.so.0",
    "libggml-base.so.0",
    "libggml-cpu.so.0",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libgomp.so.1",
)


def runtime_dir() -> Path:
    return runtime_root() / RUNTIME_NAME


def native_root() -> Path:
    return runtime_dir() / "native"


def tokenizer_dir() -> Path:
    return runtime_dir() / "tokenizer"


def native_library() -> Path:
    return native_root() / "lib" / "libnemo_speech_tts.so"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, byte_range: tuple[int, int] | None = None) -> None:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        if byte_range is not None and response.status_code != 206:
            raise RuntimeError(f"server ignored tokenizer byte range {headers['Range']}")
        with destination.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)


def _safe_extract_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("NeMo-Speech.cpp archive does not contain one runtime root")
    return roots[0]


def _tar_number(raw: bytes) -> int:
    value = raw.strip(b" \0")
    if not value:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in value):
        raise RuntimeError("invalid TAR numeric field in Magpie tokenizer archive")
    return int(value, 8)


def _tar_text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


def _validate_tar_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise RuntimeError("unsafe path in Magpie tokenizer archive")
    return path


def _read_exact(stream: BinaryIO, size: int, remaining: list[int]) -> bytes:
    if size > remaining[0]:
        raise RuntimeError("truncated Magpie tokenizer archive")
    data = stream.read(size)
    if len(data) != size:
        raise RuntimeError("truncated Magpie tokenizer archive")
    remaining[0] -= size
    return data


def _extract_tar_segment(
    stream: BinaryIO,
    destination: Path,
    segment_size: int,
    stop_before: str | None,
) -> None:
    remaining = [segment_size]
    reached_stop = False
    reached_end = False
    while remaining[0] >= 512:
        header = _read_exact(stream, 512, remaining)
        if not any(header):
            reached_end = True
            break
        expected_checksum = _tar_number(header[148:156])
        actual_checksum = sum(byte if not 148 <= index < 156 else ord(" ") for index, byte in enumerate(header))
        if expected_checksum != actual_checksum:
            raise RuntimeError("invalid TAR checksum in Magpie tokenizer archive")
        name = _tar_text(header[:100])
        prefix = _tar_text(header[345:500])
        if prefix:
            name = f"{prefix}/{name}"
        relative = _validate_tar_path(name)
        if stop_before and relative.name == stop_before:
            reached_stop = True
            break
        size = _tar_number(header[124:136])
        kind = header[156:157]
        output = destination.joinpath(*relative.parts)
        if kind == b"5":
            output.mkdir(parents=True, exist_ok=True)
            if size:
                _read_exact(stream, size, remaining)
        elif kind in (b"\0", b"0"):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(_read_exact(stream, size, remaining))
        elif kind in (b"x", b"g") and size <= 64 * 1024:
            metadata = _read_exact(stream, size, remaining)
            if b"path=" in metadata or b"linkpath=" in metadata:
                raise RuntimeError("unsupported PAX path override in Magpie tokenizer archive")
        else:
            raise RuntimeError("unsupported TAR entry in Magpie tokenizer archive")
        padding = (-size) % 512
        if padding:
            _read_exact(stream, padding, remaining)
    if stop_before and not reached_stop:
        raise RuntimeError("tokenizer archive range did not reach the expected stop member")
    if not stop_before and not reached_end and remaining[0] != 0:
        raise RuntimeError("tokenizer archive range ends inside a TAR header")
    if remaining[0]:
        _read_exact(stream, remaining[0], remaining)


def _install_tokenizer(stage: Path) -> None:
    selected = stage / ".tokenizer-ranges.tar"
    digest = hashlib.sha256()
    total = 0
    with selected.open("wb") as combined:
        for index, (start, end, _stop_before) in enumerate(TOKENIZER_RANGES):
            segment = stage / f".tokenizer-range-{index}"
            _download(TOKENIZER_URL, segment, byte_range=(start, end))
            expected_size = end - start + 1
            if segment.stat().st_size != expected_size:
                raise RuntimeError("downloaded Magpie tokenizer range has the wrong size")
            with segment.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    combined.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
            segment.unlink()
    if total != TOKENIZER_SELECTED_SIZE or digest.hexdigest() != TOKENIZER_SELECTED_SHA256:
        raise RuntimeError("Magpie tokenizer ranges failed checksum verification")

    destination = stage / "tokenizer"
    destination.mkdir()
    with selected.open("rb") as source:
        for start, end, stop_before in TOKENIZER_RANGES:
            _extract_tar_segment(source, destination, end - start + 1, stop_before)
    selected.unlink()

    actual_files = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
    if actual_files != set(TOKENIZER_MEMBERS):
        raise RuntimeError("Magpie tokenizer archive contained an unexpected member set")
    for name, (size, sha256) in TOKENIZER_MEMBERS.items():
        path = destination / name
        if path.stat().st_size != size or _sha256_file(path) != sha256:
            raise RuntimeError(f"Magpie tokenizer member failed verification: {name}")


def _sentinel_payload() -> dict[str, Any]:
    return {
        "nemo_speech_version": NEMO_SPEECH_VERSION,
        "nemo_speech_sha256": NEMO_SPEECH_SHA256,
        "magpie_revision": MAGPIE_REVISION,
        "tokenizer_sha256": TOKENIZER_SELECTED_SHA256,
    }


def _is_ready(target: Path) -> bool:
    try:
        payload = json.loads((target / _SENTINEL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload != _sentinel_payload():
        return False
    try:
        native_lib = target / "native" / "lib"
        if not all((native_lib / name).is_file() for name in _REQUIRED_NATIVE_LIBRARIES):
            return False
        tokenizer = target / "tokenizer"
        for name, (size, sha256) in TOKENIZER_MEMBERS.items():
            path = tokenizer / name
            if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != sha256:
                return False
    except OSError:
        return False
    return True


def ensure_runtime() -> Path:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Magpie currently supports Linux x86_64 CPU hosts only")
    target = runtime_dir()
    if _is_ready(target):
        return target

    with staged_target_runtime(target) as stage:
        archive = stage / NEMO_SPEECH_ARCHIVE
        _download(NEMO_SPEECH_URL, archive)
        if _sha256_file(archive) != NEMO_SPEECH_SHA256:
            raise RuntimeError("NeMo-Speech.cpp runtime failed checksum verification")
        extracted = _safe_extract_archive(archive, stage / ".native-extract")
        shutil.move(str(extracted), str(stage / "native"))
        shutil.rmtree(stage / ".native-extract")
        archive.unlink()
        _install_tokenizer(stage)
        (stage / _SENTINEL).write_text(
            json.dumps(_sentinel_payload(), sort_keys=True) + os.linesep,
            encoding="utf-8",
        )
    if not _is_ready(target):
        raise RuntimeError("Magpie runtime publication did not produce a complete runtime")
    return target
