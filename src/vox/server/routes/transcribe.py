from __future__ import annotations

import json
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
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"application/octet-stream", "binary/octet-stream"}:
        return None
    if "/" not in media_type:
        return media_type or None
    fmt = media_type.split("/")[-1].lower()
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


def _ms_to_seconds(value: int | None) -> float:
    if value is None:
        return 0.0
    return value / 1000.0


def _rich_payload(result, processing_ms: int, entities, topics) -> dict:
    response = {
        "model": result.model,
        "text": result.text,
        "language": result.language,
        "duration": _ms_to_seconds(result.duration_ms),
        "processing_ms": processing_ms,
    }
    if entities:
        response["entities"] = [asdict(e) for e in entities]
    if topics:
        response["topics"] = list(topics)
    return response


def _word_payload(word) -> dict:
    return {
        "word": word.word,
        "start": _ms_to_seconds(word.start_ms),
        "end": _ms_to_seconds(word.end_ms),
    }


def _segments_payload(result) -> list[dict]:
    segments: list[dict] = []
    for idx, segment in enumerate(result.segments):
        segments.append(
            {
                "id": idx,
                "seek": segment.start_ms or 0,
                "start": _ms_to_seconds(segment.start_ms),
                "end": _ms_to_seconds(segment.end_ms),
                "text": segment.text,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
        )
    return segments


def _words_payload(result) -> list[dict]:
    return [
        _word_payload(word)
        for segment in result.segments
        for word in segment.words
    ]


async def _timestamp_granularities(request: Request) -> set[str]:
    form = await request.form()
    values: list[str] = []
    for key in ("timestamp_granularities", "timestamp_granularities[]"):
        for value in form.getlist(key):
            raw = str(value).strip()
            if not raw:
                continue
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    values.extend(str(item).strip().lower() for item in parsed if str(item).strip())
                    continue
            values.extend(part.strip().lower() for part in raw.split(",") if part.strip())
    if not values:
        return {"segment"}
    return set(values)


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
    granularities = await _timestamp_granularities(request) if verbose else set()
    result, processing_ms, entities, topics = await _run_transcribe(
        request=request, file=file, model=model, language=language,
        word_timestamps="word" in granularities, temperature=temperature, annotate_text=verbose,
    )

    if response_format == "text":
        return PlainTextResponse(result.text)

    if verbose:
        response = _rich_payload(result, processing_ms, entities, topics)
        if "segment" in granularities:
            response["segments"] = _segments_payload(result)
        if "word" in granularities:
            response["words"] = _words_payload(result)
        return response

    return {"text": result.text}
