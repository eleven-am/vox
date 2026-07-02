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
from vox.core.errors import ModelNotFoundError, ReferenceAudioInvalidError, VoxError
from vox.core.types import VoiceInfo
from vox.operations.errors import (
    InternalOperationError,
    InvalidConfigError,
    StoredModelNotFoundError,
    VoiceAudioRequiredError,
    VoiceIdRequiredError,
    VoiceNameRequiredError,
    VoiceNotFoundOperationError,
    VoiceReferenceInvalidError,
    VoiceReferenceNotFoundError,
    WrongModelTypeError,
)


@dataclass(frozen=True)
class ListedVoice:
    voice: VoiceInfo
    model: str | None = None


def voice_payload(voice: VoiceInfo) -> dict[str, Any]:
    return {
        "id": voice.id,
        "name": voice.name,
        "language": voice.language,
        "gender": voice.gender,
        "description": voice.description,
        "is_cloned": voice.is_cloned,
    }


def listed_voice_payload(listed: ListedVoice, *, include_model: bool) -> dict[str, Any]:
    payload = voice_payload(listed.voice)
    if include_model:
        payload = {"model": listed.model, **payload}
    return payload


def list_voices_payload(listed: list[ListedVoice], *, include_model: bool) -> dict[str, Any]:
    return {
        "voices": [
            listed_voice_payload(voice, include_model=include_model)
            for voice in listed
        ]
    }


def created_voice_payload(voice: Any) -> dict[str, Any]:
    return {
        "id": voice.id,
        "name": voice.name,
        "language": voice.language,
        "gender": voice.gender,
        "is_cloned": True,
        "created_at": voice.created_at,
    }


def deleted_voice_payload(voice_id: str) -> dict[str, Any]:
    return {"id": voice_id, "deleted": True}


def _voices_for_adapter(adapter: TTSAdapter, store: Any) -> list[VoiceInfo]:
    voices = list(adapter.list_voices())
    if adapter.info().supports_voice_cloning:
        voices.extend(voice.to_voice_info() for voice in list_stored_voices(store))
    return voices


async def _listed_voices_for_model(
    *,
    scheduler: Any,
    store: Any,
    model: str,
    include_model: bool,
) -> list[ListedVoice]:
    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, TTSAdapter):
                raise WrongModelTypeError(model, "TTS")
            listed_model = model if include_model else None
            return [
                ListedVoice(voice=voice, model=listed_model)
                for voice in _voices_for_adapter(adapter, store)
            ]
    except ModelNotFoundError as exc:
        raise StoredModelNotFoundError(exc.model) from exc
    except VoxError as exc:
        raise InternalOperationError(str(exc)) from exc


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
            try:
                listed.extend(
                    await _listed_voices_for_model(
                        scheduler=scheduler,
                        store=store,
                        model=full_name,
                        include_model=True,
                    )
                )
            except WrongModelTypeError:
                continue
        return listed

    return await _listed_voices_for_model(
        scheduler=scheduler,
        store=store,
        model=model,
        include_model=False,
    )


@dataclass(frozen=True)
class CreateVoiceRequest:
    name: str
    audio: bytes
    content_type: str | None = None
    language: str | None = None
    gender: str | None = None
    reference_text: str | None = None


def create_voice_request_from_fields(
    *,
    name: str,
    audio: bytes,
    content_type: str | None = None,
    format_hint: str | None = None,
    language: str | None = None,
    gender: str | None = None,
    reference_text: str | None = None,
) -> CreateVoiceRequest:
    return CreateVoiceRequest(
        name=name,
        audio=audio,
        content_type=(content_type or format_hint or None),
        language=language or None,
        gender=gender or None,
        reference_text=reference_text or None,
    )


def create_voice(*, store: Any, request: CreateVoiceRequest):
    if not request.name or not request.name.strip():
        raise VoiceNameRequiredError()
    if not request.audio:
        raise VoiceAudioRequiredError()
    from vox.core.cloned_voices import generate_voice_id

    try:
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
    except ReferenceAudioInvalidError as exc:
        raise VoiceReferenceInvalidError(str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise InvalidConfigError(str(exc)) from exc


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
