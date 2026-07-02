from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse

from vox.operations.transcription import (
    format_hint_from_content_type,
    openai_transcription_response,
    parse_timestamp_granularities,
    transcribe,
    transcription_request_from_fields,
)
from vox.server.app_services import app_services
from vox.server.operation_errors import map_route_errors_to_http

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
    services = app_services(request)

    data = await file.read()
    with map_route_errors_to_http(
        logger=logger,
        unexpected_detail="Internal transcription error",
        unexpected_log_message=f"Transcription failed for model {model}",
    ):
        op_request = transcription_request_from_fields(
            audio=data,
            model=model or "",
            format_hint=format_hint_from_content_type(file.content_type),
            language=language,
            word_timestamps=word_timestamps,
            temperature=temperature,
            annotate_text=annotate_text,
        )
        bundle = await transcribe(
            scheduler=services.scheduler,
            registry=services.registry,
            store=services.store,
            request=op_request,
        )

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
    response = openai_transcription_response(
        bundle,
        response_format=response_format,
        timestamp_granularities=granularities,
    )

    if response.is_text:
        return PlainTextResponse(str(response.payload))

    return response.payload
