from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import (
    create_stored_voice,
    delete_stored_voice,
    generate_voice_id,
    list_stored_voices,
)
from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.types import VoiceInfo

router = APIRouter()
AUDIO_SAMPLE_FILE = File(...)


@router.get("/api/voices")
@router.get("/v1/audio/voices")
async def list_voices(request: Request, model: str = ""):
    scheduler = request.app.state.scheduler
    store = request.app.state.store

    try:
        if not model:
            all_voices = []
            for loaded in scheduler.list_loaded():
                if loaded.type.value == "tts":
                    full_name = f"{loaded.name}:{loaded.tag}"
                    async with scheduler.acquire(full_name) as adapter:
                        if isinstance(adapter, TTSAdapter):
                            for v in _voices_for_adapter(adapter, store):
                                all_voices.append({"model": full_name, **_voice_dict(v)})
            return {"voices": all_voices}

        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, TTSAdapter):
                raise HTTPException(status_code=400, detail=f"Model '{model}' is not a TTS model")
            voices = _voices_for_adapter(adapter, store)
            return {"voices": [_voice_dict(v) for v in voices]}
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except VoxError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/voices")
@router.post("/v1/audio/voices")
async def create_voice(
    request: Request,
    audio_sample: UploadFile = AUDIO_SAMPLE_FILE,
    name: str = Form(""),
    language: str | None = Form(None),
    gender: str | None = Form(None),
    reference_text: str | None = Form(None),
):
    store = request.app.state.store

    if not name.strip():
        raise HTTPException(status_code=400, detail="Voice name is required")

    data = await audio_sample.read()
    if not data:
        raise HTTPException(status_code=400, detail="Audio sample is required")

    try:
        voice = create_stored_voice(
            store,
            voice_id=generate_voice_id(store),
            name=name,
            audio_bytes=data,
            content_type=audio_sample.content_type,
            language=language,
            gender=gender,
            reference_text=reference_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "id": voice.id,
        "name": voice.name,
        "language": voice.language,
        "gender": voice.gender,
        "created_at": voice.created_at,
    }


@router.delete("/api/voices/{voice_id}")
@router.delete("/v1/audio/voices/{voice_id}")
async def delete_voice(request: Request, voice_id: str):
    store = request.app.state.store
    if not delete_stored_voice(store, voice_id):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    return {"id": voice_id, "deleted": True}


def _voice_dict(v: VoiceInfo) -> dict[str, Any]:
    return {
        "id": v.id,
        "name": v.name,
        "language": v.language,
        "gender": v.gender,
        "description": v.description,
        "is_cloned": v.is_cloned,
    }


def _voices_for_adapter(adapter: TTSAdapter, store) -> list[VoiceInfo]:
    voices = list(adapter.list_voices())
    if adapter.info().supports_voice_cloning:
        voices.extend(voice.to_voice_info() for voice in list_stored_voices(store))
    return voices
