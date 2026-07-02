from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from vox.operations.transcription import (
    TranscriptionRequest,
    openai_transcription_payload,
    transcribe,
)
from vox.server.operation_errors import map_operation_errors_to_http

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
        with map_operation_errors_to_http():
            bundle = await transcribe(
                scheduler=scheduler, registry=registry, store=store, request=op_request,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Transcription failed for model {model}")
        raise HTTPException(status_code=500, detail="Internal transcription error") from exc

    return bundle


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
    bundle = await _run_transcribe(
        request=request, file=file, model=model, language=language,
        word_timestamps="word" in granularities, temperature=temperature, annotate_text=verbose,
    )

    if response_format == "text":
        return PlainTextResponse(bundle.result.text)

    if verbose:
        return openai_transcription_payload(
            bundle,
            include_segments="segment" in granularities,
            include_words="word" in granularities,
        )

    return {"text": bundle.result.text}
