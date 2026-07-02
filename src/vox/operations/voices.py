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
from vox.core.errors import ReferenceAudioInvalidError, VoxError
from vox.core.types import VoiceInfo
from vox.operations.errors import (
    InternalOperationError,
    InvalidConfigError,
    VoiceAudioRequiredError,
    VoiceIdRequiredError,
    VoiceNameRequiredError,
    VoiceNotFoundOperationError,
    VoiceReferenceInvalidError,
    VoiceReferenceNotFoundError,
    WrongModelTypeError,
)
from vox.operations.model_acquisition import acquire_typed_adapter


@dataclass(frozen=True)
class ListedVoice:
    voice: VoiceInfo
    model: str | None = None


@dataclass(frozen=True)
class ListVoicesRequest:
    model: str | None = None


def list_voices_request_from_fields(*, model: str | None = None) -> ListVoicesRequest:
    return ListVoicesRequest(model=model or None)


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


@dataclass(frozen=True)
class DeleteVoiceResult:
    voice_id: str
    deleted: bool = True


def deleted_voice_payload(result: DeleteVoiceResult) -> dict[str, Any]:
    return {"id": result.voice_id, "deleted": result.deleted}


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
        async with acquire_typed_adapter(
            scheduler,
            model=model,
            adapter_type=TTSAdapter,
            expected_type="TTS",
        ) as adapter:
            listed_model = model if include_model else None
            return [
                ListedVoice(voice=voice, model=listed_model)
                for voice in _voices_for_adapter(adapter, store)
            ]
    except VoxError as exc:
        raise InternalOperationError(str(exc)) from exc


async def list_voices(
    *,
    scheduler: Any,
    store: Any,
    request: ListVoicesRequest,
) -> list[ListedVoice]:
    if not request.model:
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
        model=request.model,
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
    except TypeError as exc:
        raise InternalOperationError(str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise InvalidConfigError(str(exc)) from exc


@dataclass(frozen=True)
class DeleteVoiceRequest:
    voice_id: str


def delete_voice_request_from_fields(*, voice_id: str) -> DeleteVoiceRequest:
    return DeleteVoiceRequest(voice_id=voice_id)


def delete_voice(*, store: Any, request: DeleteVoiceRequest) -> DeleteVoiceResult:
    if not request.voice_id:
        raise VoiceIdRequiredError()
    if not delete_stored_voice(store, request.voice_id):
        raise VoiceNotFoundOperationError(request.voice_id)
    return DeleteVoiceResult(voice_id=request.voice_id)


@dataclass(frozen=True)
class GetVoiceReferenceRequest:
    voice_id: str


@dataclass(frozen=True)
class VoiceReferenceResult:
    voice_id: str
    audio: bytes
    content_type: str = "audio/wav"

    @property
    def filename(self) -> str:
        return f"{self.voice_id}.wav"


def get_voice_reference_request_from_fields(*, voice_id: str) -> GetVoiceReferenceRequest:
    return GetVoiceReferenceRequest(voice_id=voice_id)


def get_voice_reference(*, store: Any, request: GetVoiceReferenceRequest) -> VoiceReferenceResult:
    stored = get_stored_voice(store, request.voice_id)
    if stored is None:
        raise VoiceNotFoundOperationError(request.voice_id)
    data = reference_audio_bytes(store, request.voice_id)
    if data is None:
        raise VoiceReferenceNotFoundError(request.voice_id)
    return VoiceReferenceResult(voice_id=request.voice_id, audio=data)
