from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vox.core.errors import (
    ModelNotFoundError,
    VoiceCloningUnsupportedError,
    VoiceNotFoundError,
    VoxError,
)
from vox.operations.errors import OperationError
from vox.operations.synthesis import (
    SynthesisRequest,
    synthesize_audio_response,
)
from vox.server.operation_errors import operation_error_to_http

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


def _voice_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, VoiceCloningUnsupportedError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, VoiceNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


async def synthesize(req: SynthesizeRequest, request: Request):
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry
    store = request.app.state.store

    op_req = SynthesisRequest(
        input=req.input,
        model=req.model,
        voice=req.voice,
        speed=req.speed,
        language=req.language,
        response_format=req.response_format,
    )

    try:
        result = await synthesize_audio_response(
            scheduler=scheduler,
            registry=registry,
            store=store,
            request=op_req,
            stream=req.stream,
        )
        return StreamingResponse(
            result.chunks,
            media_type=result.content_type,
            headers={"Content-Disposition": f"attachment; filename={result.filename}"},
        )
    except HTTPException:
        raise
    except OperationError as exc:
        raise operation_error_to_http(exc) from exc
    except (VoiceCloningUnsupportedError, VoiceNotFoundError) as exc:
        raise _voice_error_to_http(exc) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except VoxError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Synthesis failed for model {req.model}")
        raise HTTPException(status_code=500, detail="Internal synthesis error") from exc


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
