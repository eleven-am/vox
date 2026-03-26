from __future__ import annotations

import logging
import time

from dataclasses import replace

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from vox.audio.pipeline import prepare_for_stt
from vox.core.adapter import STTAdapter
from vox.core.errors import ModelNotFoundError, VoxError
from vox.server.routes import get_default_model

logger = logging.getLogger(__name__)
router = APIRouter()


def _mime_to_format(content_type: str | None) -> str | None:
    """Convert a MIME type like 'audio/wav' to a format hint like 'wav'."""
    if not content_type:
        return None
    fmt = content_type.split("/")[-1].lower()
    replacements = {"mpeg": "mp3", "x-wav": "wav", "x-flac": "flac", "ogg": "ogg", "webm": "webm"}
    return replacements.get(fmt, fmt)


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

    if not model:
        model = get_default_model("stt", registry, request.app.state.store)

    data = await file.read()
    audio = prepare_for_stt(data, format_hint=_mime_to_format(file.content_type))

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
    except HTTPException:
        raise
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VoxError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception(f"Transcription failed for model {model}")
        raise HTTPException(status_code=500, detail="Internal transcription error")

    processing_ms = int((time.perf_counter() - start_time) * 1000)
    result = replace(result, model=model)

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
        model = get_default_model("stt", registry, request.app.state.store)

    data = await file.read()
    audio = prepare_for_stt(data, format_hint=_mime_to_format(file.content_type))

    try:
        async with scheduler.acquire(model) as adapter:
            if not isinstance(adapter, STTAdapter):
                raise HTTPException(status_code=400, detail=f"Model '{model}' is not an STT model")
            result = adapter.transcribe(audio, language=language, temperature=temperature)
    except HTTPException:
        raise
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VoxError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception(f"Transcription failed for model {model}")
        raise HTTPException(status_code=500, detail="Internal transcription error")

    if response_format == "text":
        return PlainTextResponse(result.text)

    return {"text": result.text}
