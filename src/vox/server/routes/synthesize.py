from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vox.audio.pipeline import get_content_type, prepare_for_output
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import get_stored_voice, load_reference_audio
from vox.core.errors import ModelNotFoundError, VoxError
from vox.server.routes import get_default_model

logger = logging.getLogger(__name__)
router = APIRouter()


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
    store = request.app.state.store

    model = req.model or get_default_model("tts", registry, request.app.state.store)

    try:
        if req.stream:
            return await _stream_synthesis(scheduler, store, model, req)
        return await _full_synthesis(scheduler, store, model, req)
    except HTTPException:
        raise
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except VoxError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Synthesis failed for model {model}")
        raise HTTPException(status_code=500, detail="Internal synthesis error") from e


async def _full_synthesis(scheduler, store, model: str, req: SynthesizeRequest):
    """Collect all audio chunks and return as a single response."""
    async with scheduler.acquire(model) as adapter:
        if not isinstance(adapter, TTSAdapter):
            raise HTTPException(status_code=400, detail=f"Model '{model}' is not a TTS model")
        voice, language, reference_audio, reference_text = _resolve_voice_request(adapter, store, req)
        chunks = []
        sample_rate = 0
        async for chunk in adapter.synthesize(
            req.input,
            voice=voice,
            speed=req.speed,
            language=language,
            reference_audio=reference_audio,
            reference_text=reference_text,
        ):
            audio_data = np.frombuffer(chunk.audio, dtype=np.float32)
            if audio_data.size > 0:
                chunks.append(audio_data)
            sample_rate = chunk.sample_rate

        if not chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        audio = np.concatenate(chunks)
        if audio.size == 0:
            raise HTTPException(status_code=500, detail="No audio generated")

        encoded, content_type = prepare_for_output(audio, sample_rate, req.response_format)
        return StreamingResponse(
            iter([encoded]),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename=speech.{req.response_format}"},
        )


async def _stream_synthesis(scheduler, store, model: str, req: SynthesizeRequest):
    """Stream audio chunks as they are generated.

    Single acquire is held for the entire stream duration to prevent
    the model from being evicted mid-synthesis.
    """
    content_type = get_content_type(req.response_format)

    async def audio_stream():
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, TTSAdapter):
                return
            voice, language, reference_audio, reference_text = _resolve_voice_request(adapter, store, req)
            async for chunk in adapter.synthesize(
                req.input,
                voice=voice,
                speed=req.speed,
                language=language,
                reference_audio=reference_audio,
                reference_text=reference_text,
            ):
                audio_data = np.frombuffer(chunk.audio, dtype=np.float32)
                if audio_data.size > 0:
                    encoded, _ = prepare_for_output(audio_data, chunk.sample_rate, req.response_format)
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


def _resolve_voice_request(adapter: TTSAdapter, store, req: SynthesizeRequest):
    if not req.voice:
        return None, req.language, None, None

    stored_voice = get_stored_voice(store, req.voice)
    if stored_voice is None:
        return req.voice, req.language, None, None

    if not adapter.info().supports_voice_cloning:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{adapter.info().name}' does not support cloned voices",
        )

    reference = load_reference_audio(
        store,
        stored_voice.id,
        target_rate=adapter.info().default_sample_rate,
    )
    if reference is None:
        raise HTTPException(status_code=404, detail=f"Voice '{stored_voice.id}' not found")

    reference_audio, reference_text = reference
    return None, req.language or stored_voice.language, reference_audio, reference_text
