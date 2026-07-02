from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vox.operations.errors import OperationError
from vox.operations.models import (
    delete_model,
    list_models,
    list_models_payload,
    pull_event_payload,
    pull_model,
    show_model,
    show_model_payload,
)
from vox.server.operation_errors import operation_error_to_http

logger = logging.getLogger(__name__)
router = APIRouter()


class PullRequest(BaseModel):
    name: str


@router.post("/v1/models/pull")
async def pull_model_route(req: PullRequest, request: Request):
    store = request.app.state.store
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry

    try:
        events = pull_model(store=store, scheduler=scheduler, registry=registry, name=req.name)
    except OperationError as exc:
        raise operation_error_to_http(exc) from exc

    async def stream():
        async for event in events:
            yield json.dumps(pull_event_payload(event)) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.get("/v1/models")
async def list_models_route(request: Request):
    store = request.app.state.store
    models = list_models(store=store)
    return list_models_payload(models)


@router.get("/v1/models/{name:path}")
async def show_model_route(name: str, request: Request):
    store = request.app.state.store
    registry = request.app.state.registry
    try:
        result = show_model(store=store, registry=registry, name=name)
    except OperationError as exc:
        raise operation_error_to_http(exc) from exc
    return show_model_payload(result)


@router.delete("/v1/models/{name:path}")
async def delete_model_route(name: str, request: Request):
    store = request.app.state.store
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry
    try:
        await delete_model(store=store, scheduler=scheduler, registry=registry, name=name)
    except OperationError as exc:
        raise operation_error_to_http(exc) from exc
    return {"status": "success"}
