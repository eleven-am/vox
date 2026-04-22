from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vox.audio.pipeline import get_content_type, prepare_for_output
from vox.conversation.text_buffer import split_for_tts
from vox.core.adapter import TTSAdapter
from vox.core.cloned_voices import resolve_voice_request
from vox.core.errors import (
    ModelNotFoundError,
    VoiceCloningUnsupportedError,
    VoiceNotFoundError,
    VoxError,
)
from vox.server.routes import get_default_model


def _split_for_adapter(text: str, adapter: TTSAdapter) -> list[str]:
    """Respect the adapter's declared max_input_chars cap, falling back to the full text."""
    max_chars = int(getattr(adapter.info(), "max_input_chars", 0) or 0)
    if max_chars <= 0:
        return [text] if text.strip() else []
    return split_for_tts(text, max_chars=max_chars)


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
    voice: str | None = None
    speed: float = 1.0
    response_format: str = "wav"
    language: str | None = None
    stream: bool = False


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
    start_time = time.perf_counter()
    async with scheduler.acquire(model) as adapter:
        if not isinstance(adapter, TTSAdapter):
            raise HTTPException(status_code=400, detail=f"Model '{model}' is not a TTS model")
        voice, language, reference_audio, reference_text = _resolve_voice_request(adapter, store, req)
        text_chunks = _split_for_adapter(req.input, adapter)
        chunks = []
        sample_rate = 0
        for text_chunk in text_chunks:
            async for chunk in adapter.synthesize(
                text_chunk,
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
        processing_ms = int((time.perf_counter() - start_time) * 1000)
        audio_ms = int(1000 * audio.size / max(sample_rate, 1))
        logger.info(
            "synthesize %s chars=%d audio_ms=%d processing_ms=%d format=%s",
            model, len(req.input or ""), audio_ms, processing_ms, req.response_format,
        )
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
            for text_chunk in _split_for_adapter(req.input, adapter):
                async for chunk in adapter.synthesize(
                    text_chunk,
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



@router.post("/v1/audio/speech")
async def openai_speech(req: OpenAISpeechRequest, request: Request):
    synth_req = SynthesizeRequest(
        model=req.model,
        input=req.input,
        voice=req.voice,
        speed=req.speed,
        language=req.language,
        response_format=req.response_format,
        stream=req.stream,
    )
    return await synthesize(synth_req, request)


def _resolve_voice_request(adapter: TTSAdapter, store, req: SynthesizeRequest):
    try:
        return resolve_voice_request(adapter, store, req.voice, req.language)
    except VoiceCloningUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VoiceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
