from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vox.operations.synthesis import (
    synthesis_request_from_fields,
    synthesize_audio_response,
)
from vox.server.app_services import app_services
from vox.server.operation_errors import map_route_errors_to_http

logger = logging.getLogger(__name__)
router = APIRouter()


class SpeechRequest(BaseModel):
    model: str = ""
    input: str
    voice: str | None = None
    speed: float = 1.0
    language: str | None = None
    response_format: str = "wav"
    stream: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


async def synthesize(req: SpeechRequest, request: Request):
    services = app_services(request)

    with map_route_errors_to_http(
        logger=logger,
        unexpected_detail="Internal synthesis error",
        unexpected_log_message=f"Synthesis failed for model {req.model}",
    ):
        op_req = synthesis_request_from_fields(
            input=req.input,
            model=req.model,
            voice=req.voice,
            speed=req.speed,
            language=req.language,
            response_format=req.response_format,
            params=req.params,
        )
        result = await synthesize_audio_response(
            scheduler=services.scheduler,
            registry=services.registry,
            store=services.store,
            request=op_req,
            stream=req.stream,
        )
        return StreamingResponse(
            result.chunks,
            media_type=result.content_type,
            headers={"Content-Disposition": f"attachment; filename={result.filename}"},
        )


@router.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest, request: Request):
    return await synthesize(req, request)
