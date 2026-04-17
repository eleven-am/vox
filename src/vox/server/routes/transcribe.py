from __future__ import annotations

import logging
import time

from dataclasses import replace

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from vox.audio.pipeline import prepare_for_stt
from vox.core.adapter import STTAdapter
from vox.core.errors import ModelNotFoundError, VoxError
from vox.core.ner import annotate, entity_to_dict
from vox.server.routes import get_default_model

logger = logging.getLogger(__name__)
router = APIRouter()


def _mime_to_format(content_type: str | None) -> str | None:
    if not content_type:
        return None
    fmt = content_type.split("/")[-1].lower()
    replacements = {"mpeg": "mp3", "x-wav": "wav", "x-flac": "flac", "ogg": "ogg", "webm": "webm"}
    return replacements.get(fmt, fmt)


async def _run_transcribe(
    *,
    request: Request,
    file: UploadFile,
    model: str,
    language: str | None,
    word_timestamps: bool,
    temperature: float,
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
    logger.info(
        "transcribe %s audio_ms=%d processing_ms=%d chars=%d",
        model, result.duration_ms, processing_ms, len(result.text or ""),
    )
    return result, processing_ms


def _rich_payload(result, processing_ms: int, language_hint: str | None) -> dict:
    response = {
        "model": result.model,
        "text": result.text,
        "language": result.language,
        "duration_ms": result.duration_ms,
        "processing_ms": processing_ms,
    }
    if result.text:
        entities, topics = annotate(result.text, language_hint or result.language or "en")
        if entities:
            response["entities"] = [entity_to_dict(e) for e in entities]
        if topics:
            response["topics"] = topics
    return response


def _segments_payload(result) -> list[dict]:
    return [
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


@router.post("/v1/audio/transcriptions")
async def openai_transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """OpenAI-compatible transcription.

    - response_format=json (default): {"text": ...} — strict OpenAI shape.
    - response_format=verbose_json: rich payload with segments/words + Vox extras (entities, topics).
    - response_format=text: plain text.
    """
    verbose = response_format == "verbose_json"
    result, processing_ms = await _run_transcribe(
        request=request, file=file, model=model, language=language,
        word_timestamps=verbose, temperature=temperature,
    )

    if response_format == "text":
        return PlainTextResponse(result.text)

    if verbose:
        response = _rich_payload(result, processing_ms, language)
        response["segments"] = _segments_payload(result)
        return response

    return {"text": result.text}
