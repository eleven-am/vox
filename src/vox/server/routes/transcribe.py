from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from vox.operations.transcription import (
    format_hint_from_content_type,
    openai_transcription_payload,
    parse_timestamp_granularities,
    transcribe,
    transcription_request_from_fields,
)
from vox.server.operation_errors import map_operation_errors_to_http

logger = logging.getLogger(__name__)
router = APIRouter()


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
    op_request = transcription_request_from_fields(
        audio=data,
        model=model or "",
        format_hint=format_hint_from_content_type(file.content_type),
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
            values.append(str(value))
    return parse_timestamp_granularities(values)


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
