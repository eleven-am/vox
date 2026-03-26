from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vox.audio.pipeline import AudioPipeline
from vox.core.adapter import TTSAdapter
from vox.core.errors import ModelNotFoundError, VoxError

logger = logging.getLogger(__name__)
router = APIRouter()
audio_pipeline = AudioPipeline()


class SynthesizeRequest(BaseModel):
    model: str = ""
    input: str
    voice: str | None = None
    speed: float = 1.0
    language: str | None = None
    response_format: str = "wav"
    stream: bool = False


class OpenAISpeechRequest(BaseModel):
    model: str = ""
    input: str
    voice: str = "default"
    speed: float = 1.0
    response_format: str = "wav"


@router.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest, request: Request):
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry

    model = req.model or _get_default_tts_model(registry)

    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, TTSAdapter):
                raise HTTPException(status_code=400, detail=f"Model '{model}' is not a TTS model")

            if req.stream:
                return await _stream_synthesis(adapter, req)
            else:
                return await _full_synthesis(adapter, req)

    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VoxError as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _full_synthesis(adapter: TTSAdapter, req: SynthesizeRequest):
    """Collect all audio chunks and return as a single response."""
    import numpy as np

    chunks = []
    sample_rate = 0
    async for chunk in adapter.synthesize(
        req.input, voice=req.voice, speed=req.speed, language=req.language,
    ):
        chunks.append(np.frombuffer(chunk.audio, dtype=np.float32))
        sample_rate = chunk.sample_rate

    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    audio = np.concatenate(chunks)
    encoded, content_type = audio_pipeline.prepare_for_output(audio, sample_rate, req.response_format)
    return StreamingResponse(
        iter([encoded]),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename=speech.{req.response_format}"},
    )


async def _stream_synthesis(adapter: TTSAdapter, req: SynthesizeRequest):
    """Stream audio chunks as they are generated."""
    import numpy as np

    content_type = AudioPipeline.get_content_type(req.response_format)

    async def audio_stream():
        async for chunk in adapter.synthesize(
            req.input, voice=req.voice, speed=req.speed, language=req.language,
        ):
            audio = np.frombuffer(chunk.audio, dtype=np.float32)
            encoded, _ = audio_pipeline.prepare_for_output(audio, chunk.sample_rate, req.response_format)
            yield encoded

    return StreamingResponse(audio_stream(), media_type=content_type)


# OpenAI-compatible endpoint
@router.post("/v1/audio/speech")
async def openai_speech(req: OpenAISpeechRequest, request: Request):
    """OpenAI-compatible speech synthesis endpoint."""
    synth_req = SynthesizeRequest(
        model=req.model,
        input=req.input,
        voice=req.voice,
        speed=req.speed,
        response_format=req.response_format,
    )
    return await synthesize(synth_req, request)


def _get_default_tts_model(registry) -> str:
    """Get the first available TTS model from catalog."""
    for name, tags in registry.available_models().items():
        for tag, entry in tags.items():
            if entry.get("type") == "tts":
                return f"{name}:{tag}"
    raise HTTPException(status_code=400, detail="No model specified and no default TTS model available")
