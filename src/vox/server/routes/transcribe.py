from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from vox.core.errors import ModelNotFoundError, VoxError
from vox.operations.errors import (
    EmptyAudioError,
    NoDefaultModelError,
    OperationError,
    WrongModelTypeError,
)
from vox.operations.transcription import TranscriptionRequest, transcribe

logger = logging.getLogger(__name__)
router = APIRouter()


def _mime_to_format(content_type: str | None) -> str | None:
    if not content_type:
        return None
    fmt = content_type.split("/")[-1].lower()
    replacements = {"mpeg": "mp3", "x-wav": "wav", "x-flac": "flac", "ogg": "ogg", "webm": "webm"}
    return replacements.get(fmt, fmt)


def _operation_error_to_http(exc: OperationError) -> HTTPException:
    if isinstance(exc, NoDefaultModelError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, EmptyAudioError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, WrongModelTypeError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


async def _run_transcribe(
    *,
    request: Request,
    file: UploadFile,
    model: str,
    language: str | None,
    word_timestamps: bool,
    temperature: float,
    annotate_text: bool = False,
):
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry
    store = request.app.state.store

    data = await file.read()
    op_request = TranscriptionRequest(
        audio=data,
        model=model or "",
        format_hint=_mime_to_format(file.content_type),
        language=language,
        word_timestamps=word_timestamps,
        temperature=temperature,
        annotate_text=annotate_text,
    )
    try:
        bundle = await transcribe(
            scheduler=scheduler, registry=registry, store=store, request=op_request,
        )
    except OperationError as exc:
        raise _operation_error_to_http(exc) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except VoxError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Transcription failed for model {model}")
        raise HTTPException(status_code=500, detail="Internal transcription error") from exc

    return bundle.result, bundle.processing_ms, bundle.entities, bundle.topics


def _rich_payload(result, processing_ms: int, entities, topics) -> dict:
    response = {
        "model": result.model,
        "text": result.text,
        "language": result.language,
        "duration_ms": result.duration_ms,
        "processing_ms": processing_ms,
    }
    if entities:
        response["entities"] = [asdict(e) for e in entities]
    if topics:
        response["topics"] = list(topics)
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
    file: UploadFile = File(...),  # noqa: B008
    model: str = Form(""),  # noqa: B008
    language: str | None = Form(None),  # noqa: B008
    response_format: str = Form("json"),  # noqa: B008
    temperature: float = Form(0.0),  # noqa: B008
):
    verbose = response_format == "verbose_json"
    result, processing_ms, entities, topics = await _run_transcribe(
        request=request, file=file, model=model, language=language,
        word_timestamps=verbose, temperature=temperature, annotate_text=verbose,
    )

    if response_format == "text":
        return PlainTextResponse(result.text)

    if verbose:
        response = _rich_payload(result, processing_ms, entities, topics)
        response["segments"] = _segments_payload(result)
        return response

    return {"text": result.text}
