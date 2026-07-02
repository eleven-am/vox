from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vox.operations.models import (
    delete_model,
    delete_model_payload,
    list_models,
    list_models_payload,
    model_reference_request_from_fields,
    pull_event_payload,
    pull_model,
    show_model,
    show_model_payload,
)
from vox.server.app_services import app_services, app_store
from vox.server.operation_errors import map_operation_errors_to_http

logger = logging.getLogger(__name__)
router = APIRouter()


class PullRequest(BaseModel):
    name: str
    variant: str | None = None


@router.post("/v1/models/pull")
async def pull_model_route(req: PullRequest, request: Request):
    services = app_services(request)

    with map_operation_errors_to_http():
        events = pull_model(
            store=services.store,
            scheduler=services.scheduler,
            registry=services.registry,
            request=model_reference_request_from_fields(name=req.name, variant=req.variant),
        )

    async def stream():
        async for event in events:
            yield json.dumps(pull_event_payload(event)) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.get("/v1/models")
async def list_models_route(request: Request):
    store = app_store(request)
    models = list_models(store=store)
    return list_models_payload(models)


@router.get("/v1/models/{name:path}")
async def show_model_route(name: str, request: Request):
    services = app_services(request)
    with map_operation_errors_to_http():
        result = show_model(
            store=services.store,
            registry=services.registry,
            request=model_reference_request_from_fields(name=name),
        )
    return show_model_payload(result)


@router.delete("/v1/models/{name:path}")
async def delete_model_route(name: str, request: Request):
    services = app_services(request)
    with map_operation_errors_to_http():
        await delete_model(
            store=services.store,
            scheduler=services.scheduler,
            registry=services.registry,
            request=model_reference_request_from_fields(name=name),
        )
    return delete_model_payload()
