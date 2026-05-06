from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import (
    create_stored_voice,
    delete_stored_voice,
    get_stored_voice,
    list_stored_voices,
    reference_audio_bytes,
)
from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.types import VoiceInfo
from vox.operations.errors import (
    VoiceAudioRequiredError,
    VoiceIdRequiredError,
    VoiceNameRequiredError,
    VoiceNotFoundOperationError,
    VoiceReferenceNotFoundError,
    WrongModelTypeError,
)


@dataclass(frozen=True)
class ListedVoice:
    voice: VoiceInfo
    model: str | None = None


def _voices_for_adapter(adapter: TTSAdapter, store: Any) -> list[VoiceInfo]:
    voices = list(adapter.list_voices())
    if adapter.info().supports_voice_cloning:
        voices.extend(voice.to_voice_info() for voice in list_stored_voices(store))
    return voices


async def list_voices(
    *,
    scheduler: Any,
    store: Any,
    model: str | None = None,
) -> list[ListedVoice]:
    if not model:
        listed: list[ListedVoice] = []
        for loaded in scheduler.list_loaded():
            if loaded.type.value != "tts":
                continue
            full_name = f"{loaded.name}:{loaded.tag}"
            async with scheduler.acquire(full_name) as adapter:
                if not isinstance(adapter, TTSAdapter):
                    continue
                for v in _voices_for_adapter(adapter, store):
                    listed.append(ListedVoice(voice=v, model=full_name))
        return listed

    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, TTSAdapter):
                raise WrongModelTypeError(model, "TTS")
            return [ListedVoice(voice=v) for v in _voices_for_adapter(adapter, store)]
    except (ModelNotFoundError, VoxError, WrongModelTypeError):
        raise


@dataclass(frozen=True)
class CreateVoiceRequest:
    name: str
    audio: bytes
    content_type: str | None = None
    language: str | None = None
    gender: str | None = None
    reference_text: str | None = None


def create_voice(*, store: Any, request: CreateVoiceRequest):
    if not request.name or not request.name.strip():
        raise VoiceNameRequiredError()
    if not request.audio:
        raise VoiceAudioRequiredError()
    from vox.core.cloned_voices import generate_voice_id

    return create_stored_voice(
        store,
        voice_id=generate_voice_id(store),
        name=request.name,
        audio_bytes=request.audio,
        content_type=request.content_type,
        language=request.language,
        gender=request.gender,
        reference_text=request.reference_text,
    )


def delete_voice(*, store: Any, voice_id: str) -> None:
    if not voice_id:
        raise VoiceIdRequiredError()
    if not delete_stored_voice(store, voice_id):
        raise VoiceNotFoundOperationError(voice_id)


def get_voice_reference(*, store: Any, voice_id: str) -> bytes:
    stored = get_stored_voice(store, voice_id)
    if stored is None:
        raise VoiceNotFoundOperationError(voice_id)
    data = reference_audio_bytes(store, voice_id)
    if data is None:
        raise VoiceReferenceNotFoundError(voice_id)
    return data
