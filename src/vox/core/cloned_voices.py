from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from vox.audio.codecs import decode_audio, encode_wav, to_mono
from vox.audio.resampler import resample
from vox.core.errors import (
    ReferenceAudioInvalidError,
    VoiceCloningUnsupportedError,
    VoiceNotFoundError,
)
from vox.core.store import BlobStore
from vox.core.types import VoiceInfo

REFERENCE_MIN_SECONDS = 1.0
REFERENCE_MAX_SECONDS = 30.0
REFERENCE_CLIP_THRESHOLD = 0.99
REFERENCE_CLIP_FRACTION_MAX = 0.01
REFERENCE_RMS_MIN = 1e-3


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


def validate_reference_audio(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    min_seconds: float = REFERENCE_MIN_SECONDS,
    max_seconds: float = REFERENCE_MAX_SECONDS,
) -> None:
    if sample_rate <= 0:
        raise ReferenceAudioInvalidError(f"Invalid sample rate: {sample_rate}")

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.reshape(-1) if samples.shape[0] == 1 else samples.mean(axis=-1 if samples.shape[-1] <= 2 else 0)

    if samples.size == 0:
        raise ReferenceAudioInvalidError("Reference audio is empty")

    duration = samples.size / float(sample_rate)
    if duration < min_seconds:
        raise ReferenceAudioInvalidError(
            f"Reference audio too short ({duration:.2f}s); minimum is {min_seconds:.1f}s"
        )
    if duration > max_seconds:
        raise ReferenceAudioInvalidError(
            f"Reference audio too long ({duration:.2f}s); maximum is {max_seconds:.1f}s"
        )

    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    if rms < REFERENCE_RMS_MIN:
        raise ReferenceAudioInvalidError(
            f"Reference audio is effectively silent (RMS {rms:.2e}); upload a clearer sample"
        )

    clipped = float(np.mean(np.abs(samples) >= REFERENCE_CLIP_THRESHOLD))
    if clipped > REFERENCE_CLIP_FRACTION_MAX:
        raise ReferenceAudioInvalidError(
            f"Reference audio is heavily clipped ({clipped * 100:.1f}% of samples at full scale); "
            "re-record at a lower input level"
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
    validate: bool = True,
) -> StoredVoice:
    if not name.strip():
        raise ValueError("Voice name is required")

    audio, sample_rate = decode_audio(audio_bytes, format_hint=_format_hint(content_type))
    if validate:
        validate_reference_audio(np.asarray(audio, dtype=np.float32), sample_rate)
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


def reference_audio_bytes(store: BlobStore, voice_id: str) -> bytes | None:
    path = _audio_path(store, voice_id)
    if not path.is_file():
        return None
    return path.read_bytes()


def resolve_voice_request(
    adapter: Any,
    store: BlobStore,
    voice_id: str | None,
    language: str | None,
) -> tuple[str | None, str | None, NDArray[np.float32] | None, str | None]:
    """Resolve a voice reference into synthesis kwargs.

    Returns (voice_to_pass, language, reference_audio, reference_text).
    - If voice_id is empty or no stored voice exists, the voice is passed through unchanged.
    - If a stored voice exists and the adapter doesn't support cloning, raises.
    """
    if not voice_id:
        return None, language, None, None

    stored = get_stored_voice(store, voice_id)
    if stored is None:
        return voice_id, language, None, None

    info = adapter.info()
    if not info.supports_voice_cloning:
        raise VoiceCloningUnsupportedError(info.name)

    reference = load_reference_audio(store, stored.id, target_rate=info.default_sample_rate)
    if reference is None:
        raise VoiceNotFoundError(voice_id)

    reference_audio, reference_text = reference
    return None, language or stored.language, reference_audio, reference_text


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
