from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from vox.core.adapter import TTSAdapter

router = APIRouter()


@router.get("/api/voices")
async def list_voices(request: Request, model: str = ""):
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry

    if not model:
        # List voices across all loaded TTS models
        all_voices = []
        for loaded in scheduler.list_loaded():
            if loaded.type.value == "tts":
                full_name = f"{loaded.name}:{loaded.tag}"
                async with scheduler.acquire(full_name) as adapter:
                    if isinstance(adapter, TTSAdapter):
                        for v in adapter.list_voices():
                            all_voices.append({"model": full_name, **_voice_dict(v)})
        return {"voices": all_voices}

    async with scheduler.acquire(model) as adapter:
        if not isinstance(adapter, TTSAdapter):
            raise HTTPException(status_code=400, detail=f"Model '{model}' is not a TTS model")
        voices = adapter.list_voices()
        return {"voices": [_voice_dict(v) for v in voices]}


def _voice_dict(v):
    return {
        "id": v.id,
        "name": v.name,
        "language": v.language,
        "gender": v.gender,
        "description": v.description,
        "is_cloned": v.is_cloned,
    }
