from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from vox.audio.pipeline import AudioPipeline
from vox.core.adapter import STTAdapter
from vox.core.errors import ModelNotFoundError, VoxError

logger = logging.getLogger(__name__)
router = APIRouter()
audio_pipeline = AudioPipeline()


@router.post("/api/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str | None = Form(None),
    word_timestamps: bool = Form(False),
    temperature: float = Form(0.0),
    response_format: str = Form("json"),
    stream: bool = Form(False),
):
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry

    # Use default STT model if none specified
    if not model:
        model = _get_default_stt_model(registry)

    data = await file.read()
    audio = audio_pipeline.prepare_for_stt(data, format_hint=file.content_type)

    start_time = time.perf_counter()

    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                raise HTTPException(status_code=400, detail=f"Model '{model}' is not an STT model")
            result = adapter.transcribe(
                audio,
                language=language,
                word_timestamps=word_timestamps,
                temperature=temperature,
            )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VoxError as e:
        raise HTTPException(status_code=500, detail=str(e))

    processing_ms = int((time.perf_counter() - start_time) * 1000)
    result.model = model

    if response_format == "text":
        return PlainTextResponse(result.text)

    response = {
        "model": result.model,
        "text": result.text,
        "language": result.language,
        "duration_ms": result.duration_ms,
        "processing_ms": processing_ms,
    }

    if response_format == "verbose_json":
        response["segments"] = [
            {
                "text": s.text,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "words": [
                    {"word": w.word, "start_ms": w.start_ms, "end_ms": w.end_ms, "confidence": w.confidence}
                    for w in s.words
                ] if s.words else [],
            }
            for s in result.segments
        ]

    return response


# OpenAI-compatible endpoint
@router.post("/v1/audio/transcriptions")
async def openai_transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """OpenAI-compatible transcription endpoint."""
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry

    if not model:
        model = _get_default_stt_model(registry)

    data = await file.read()
    audio = audio_pipeline.prepare_for_stt(data, format_hint=file.content_type)

    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                raise HTTPException(status_code=400, detail=f"Model '{model}' is not an STT model")
            result = adapter.transcribe(audio, language=language, temperature=temperature)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VoxError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if response_format == "text":
        return PlainTextResponse(result.text)

    return {"text": result.text}


def _get_default_stt_model(registry) -> str:
    """Get the first available STT model from catalog."""
    for name, tags in registry.available_models().items():
        for tag, entry in tags.items():
            if entry.get("type") == "stt":
                return f"{name}:{tag}"
    raise HTTPException(status_code=400, detail="No model specified and no default STT model available")
