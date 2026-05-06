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
from vox.operations.errors import (
    EmptyInputError,
    NoAudioGeneratedError,
    NoDefaultModelError,
    OperationError,
    WrongModelTypeError,
)
from vox.operations.synthesis import (
    SynthesisRequest,
    stream_content_type,
    synthesize_full,
    synthesize_stream,
)

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


def _operation_error_to_http(exc: OperationError) -> HTTPException:
    if isinstance(exc, NoDefaultModelError | EmptyInputError | WrongModelTypeError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, NoAudioGeneratedError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


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
        if req.stream:
            return await _stream_response(scheduler, registry, store, op_req)
        return await _full_response(scheduler, registry, store, op_req)
    except HTTPException:
        raise
    except OperationError as exc:
        raise _operation_error_to_http(exc) from exc
    except (VoiceCloningUnsupportedError, VoiceNotFoundError) as exc:
        raise _voice_error_to_http(exc) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except VoxError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Synthesis failed for model {req.model}")
        raise HTTPException(status_code=500, detail="Internal synthesis error") from exc


async def _full_response(scheduler, registry, store, op_req: SynthesisRequest):
    bundle = await synthesize_full(
        scheduler=scheduler, registry=registry, store=store, request=op_req,
    )
    return StreamingResponse(
        iter([bundle.audio]),
        media_type=bundle.content_type,
        headers={"Content-Disposition": f"attachment; filename=speech.{bundle.response_format}"},
    )


async def _stream_response(scheduler, registry, store, op_req: SynthesisRequest):
    iterator = await synthesize_stream(
        scheduler=scheduler, registry=registry, store=store, request=op_req,
    )
    content_type = stream_content_type(op_req.response_format)
    return StreamingResponse(iterator, media_type=content_type)


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
