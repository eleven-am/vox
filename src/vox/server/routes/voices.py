from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from vox.core.errors import ModelNotFoundError, ReferenceAudioInvalidError, VoxError
from vox.operations.errors import (
    OperationError,
    VoiceAudioRequiredError,
    VoiceIdRequiredError,
    VoiceNameRequiredError,
    VoiceNotFoundOperationError,
    VoiceReferenceNotFoundError,
    WrongModelTypeError,
)
from vox.operations.voices import (
    CreateVoiceRequest,
    create_voice,
    created_voice_payload,
    delete_voice,
    deleted_voice_payload,
    get_voice_reference,
    list_voices,
    list_voices_payload,
)

router = APIRouter()
AUDIO_SAMPLE_FILE = File(...)


def _voice_op_error_to_http(exc: OperationError) -> HTTPException:
    if isinstance(exc, (VoiceNameRequiredError, VoiceAudioRequiredError, VoiceIdRequiredError, WrongModelTypeError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (VoiceNotFoundOperationError, VoiceReferenceNotFoundError)):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.get("/v1/audio/voices")
async def list_voices_route(request: Request, model: str = ""):
    scheduler = request.app.state.scheduler
    store = request.app.state.store
    try:
        listed = await list_voices(scheduler=scheduler, store=store, model=model or None)
    except OperationError as exc:
        raise _voice_op_error_to_http(exc) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except VoxError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return list_voices_payload(listed, include_model=not model)


@router.post("/v1/audio/voices")
async def create_voice_route(
    request: Request,
    audio_sample: UploadFile = AUDIO_SAMPLE_FILE,
    name: str = Form(""),
    language: str | None = Form(None),
    gender: str | None = Form(None),
    reference_text: str | None = Form(None),
):
    store = request.app.state.store
    data = await audio_sample.read()
    op_req = CreateVoiceRequest(
        name=name,
        audio=data,
        content_type=audio_sample.content_type,
        language=language,
        gender=gender,
        reference_text=reference_text,
    )
    try:
        voice = create_voice(store=store, request=op_req)
    except OperationError as exc:
        raise _voice_op_error_to_http(exc) from exc
    except ReferenceAudioInvalidError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return created_voice_payload(voice)


@router.get("/v1/audio/voices/{voice_id}/reference")
async def get_voice_reference_route(request: Request, voice_id: str):
    store = request.app.state.store
    try:
        data = get_voice_reference(store=store, voice_id=voice_id)
    except OperationError as exc:
        raise _voice_op_error_to_http(exc) from exc
    return Response(
        content=data,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{voice_id}.wav"'},
    )


@router.delete("/v1/audio/voices/{voice_id}")
async def delete_voice_route(request: Request, voice_id: str):
    store = request.app.state.store
    try:
        delete_voice(store=store, voice_id=voice_id)
    except OperationError as exc:
        raise _voice_op_error_to_http(exc) from exc
    return deleted_voice_payload(voice_id)
