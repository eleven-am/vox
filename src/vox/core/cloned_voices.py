from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from vox.audio.codecs import decode_audio, encode_wav, to_mono
from vox.audio.resampler import resample
from vox.core.store import BlobStore
from vox.core.types import VoiceInfo


@dataclass(frozen=True)
class StoredVoice:
    id: str
    name: str
    language: str | None = None
    gender: str | None = None
    description: str | None = None
    reference_text: str | None = None
    created_at: int = 0

    def to_voice_info(self) -> VoiceInfo:
        return VoiceInfo(
            id=self.id,
            name=self.name,
            language=self.language,
            gender=self.gender,
            description=self.description,
            is_cloned=True,
        )


def _voice_dir(store: BlobStore, voice_id: str) -> Path:
    root = _voices_root(store)
    if root is None:
        raise TypeError("Store does not expose a concrete voices directory")
    return root / voice_id


def _metadata_path(store: BlobStore, voice_id: str) -> Path:
    return _voice_dir(store, voice_id) / "metadata.json"


def _audio_path(store: BlobStore, voice_id: str) -> Path:
    return _voice_dir(store, voice_id) / "reference.wav"


def _format_hint(content_type: str | None) -> str | None:
    if not content_type:
        return None
    fmt = content_type.split("/")[-1].lower()
    replacements = {
        "mpeg": "mp3",
        "x-wav": "wav",
        "x-flac": "flac",
        "ogg": "ogg",
        "webm": "webm",
    }
    return replacements.get(fmt, fmt)


def _voices_root(store: BlobStore) -> Path | None:
    root = getattr(store, "voices_dir", None)
    return root if isinstance(root, Path) else None


def generate_voice_id(store: BlobStore) -> str:
    voices_root = _voices_root(store)
    if voices_root is None:
        raise TypeError("Store does not expose a concrete voices directory")
    voices_root.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = uuid4().hex[:8]
        if not _voice_dir(store, candidate).exists():
            return candidate


def list_stored_voices(store: BlobStore) -> list[StoredVoice]:
    voices: list[StoredVoice] = []
    voices_root = _voices_root(store)
    if voices_root is None or not voices_root.is_dir():
        return voices

    for voice_dir in sorted(voices_root.iterdir()):
        if not voice_dir.is_dir():
            continue
        try:
            voices.append(get_stored_voice(store, voice_dir.name, required=True))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    return voices


def get_stored_voice(store: BlobStore, voice_id: str, *, required: bool = False) -> StoredVoice | None:
    if _voices_root(store) is None:
        if required:
            raise FileNotFoundError(voice_id)
        return None

    meta_path = _metadata_path(store, voice_id)
    if not meta_path.is_file():
        if required:
            raise FileNotFoundError(voice_id)
        return None

    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return StoredVoice(
        id=data["id"],
        name=data["name"],
        language=data.get("language"),
        gender=data.get("gender"),
        description=data.get("description"),
        reference_text=data.get("reference_text"),
        created_at=int(data.get("created_at", 0)),
    )


def create_stored_voice(
    store: BlobStore,
    *,
    voice_id: str,
    name: str,
    audio_bytes: bytes,
    content_type: str | None = None,
    language: str | None = None,
    gender: str | None = None,
    reference_text: str | None = None,
) -> StoredVoice:
    if not name.strip():
        raise ValueError("Voice name is required")

    audio, sample_rate = decode_audio(audio_bytes, format_hint=_format_hint(content_type))
    voice = StoredVoice(
        id=voice_id,
        name=name.strip(),
        language=language or None,
        gender=gender or None,
        description="Custom cloned voice",
        reference_text=reference_text or None,
        created_at=int(time.time()),
    )

    target_dir = _voice_dir(store, voice_id)
    if target_dir.exists():
        raise ValueError(f"Voice {voice_id!r} already exists")

    voices_root = _voices_root(store)
    if voices_root is None:
        raise TypeError("Store does not expose a concrete voices directory")
    voices_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=voices_root, prefix=f"{voice_id}.tmp.") as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "metadata.json").write_text(
            json.dumps(asdict(voice), indent=2),
            encoding="utf-8",
        )
        (tmpdir_path / "reference.wav").write_bytes(encode_wav(np.asarray(audio, dtype=np.float32), sample_rate))
        tmpdir_path.rename(target_dir)

    return voice


def delete_stored_voice(store: BlobStore, voice_id: str) -> bool:
    target_dir = _voice_dir(store, voice_id)
    if not target_dir.exists():
        return False
    shutil.rmtree(target_dir)
    return True


def load_reference_audio(
    store: BlobStore,
    voice_id: str,
    *,
    target_rate: int,
) -> tuple[NDArray[np.float32], str | None] | None:
    voice = get_stored_voice(store, voice_id)
    if voice is None:
        return None

    data = _audio_path(store, voice_id).read_bytes()
    audio, sample_rate = decode_audio(data, format_hint="wav")
    audio = to_mono(audio)
    audio = resample(audio, sample_rate, target_rate)
    return np.asarray(audio, dtype=np.float32), voice.reference_text
