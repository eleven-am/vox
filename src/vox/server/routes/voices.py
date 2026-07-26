from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response

from vox.operations.voices import (
    create_voice,
    create_voice_request_from_fields,
    created_voice_payload,
    delete_voice,
    delete_voice_request_from_fields,
    deleted_voice_payload,
    get_voice_reference,
    get_voice_reference_request_from_fields,
    list_voices,
    list_voices_payload,
    list_voices_request_from_fields,
)
from vox.server.app_services import app_services, app_store
from vox.server.operation_errors import map_operation_errors_to_http
from vox.server.uploads import read_upload_limited

router = APIRouter()
AUDIO_SAMPLE_FILE = File(...)


@router.get("/v1/audio/voices")
async def list_voices_route(request: Request, model: str = ""):
    services = app_services(request)
    with map_operation_errors_to_http():
        listed = await list_voices(
            scheduler=services.scheduler,
            store=services.store,
            request=list_voices_request_from_fields(model=model),
        )

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
    store = app_store(request)
    data = await read_upload_limited(
        audio_sample,
        max_bytes=getattr(request.app.state, "max_upload_bytes", None),
    )
    with map_operation_errors_to_http():
        op_req = create_voice_request_from_fields(
            name=name,
            audio=data,
            content_type=audio_sample.content_type,
            language=language,
            gender=gender,
            reference_text=reference_text,
        )
        voice = create_voice(store=store, request=op_req)

    return created_voice_payload(voice)


@router.get("/v1/audio/voices/{voice_id}/reference")
async def get_voice_reference_route(request: Request, voice_id: str):
    store = app_store(request)
    with map_operation_errors_to_http():
        result = get_voice_reference(
            store=store,
            request=get_voice_reference_request_from_fields(voice_id=voice_id),
        )
    return Response(
        content=result.audio,
        media_type=result.content_type,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


@router.delete("/v1/audio/voices/{voice_id}")
async def delete_voice_route(request: Request, voice_id: str):
    store = app_store(request)
    with map_operation_errors_to_http():
        result = delete_voice(
            store=store,
            request=delete_voice_request_from_fields(voice_id=voice_id),
        )
    return deleted_voice_payload(result)
