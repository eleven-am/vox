from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vox.operations.errors import (
    CatalogEntryNotFoundError,
    ModelInUseError,
    OperationError,
    StoredModelNotFoundError,
)
from vox.operations.models import (
    delete_model,
    list_models,
    pull_model,
    show_model,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class PullRequest(BaseModel):
    name: str


def _model_op_error_to_http(exc: OperationError) -> HTTPException:
    if isinstance(exc, (CatalogEntryNotFoundError, StoredModelNotFoundError)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ModelInUseError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/v1/models/pull")
async def pull_model_route(req: PullRequest, request: Request):
    store = request.app.state.store
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry

    try:
        events = pull_model(store=store, scheduler=scheduler, registry=registry, name=req.name)
    except OperationError as exc:
        raise _model_op_error_to_http(exc) from exc

    async def stream():
        async for event in events:
            payload = {"status": event.status}
            if event.total > 0:
                payload["completed"] = event.completed
                payload["total"] = event.total
            if event.error:
                payload["error"] = event.error
            yield json.dumps(payload) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.get("/v1/models")
async def list_models_route(request: Request):
    store = request.app.state.store
    models = list_models(store=store)
    return {
        "models": [
            {
                "name": m.full_name,
                "type": m.type.value,
                "format": m.format.value,
                "architecture": m.architecture,
                "size_bytes": m.size_bytes,
                "description": m.description,
            }
            for m in models
        ]
    }


@router.get("/v1/models/{name:path}")
async def show_model_route(name: str, request: Request):
    store = request.app.state.store
    registry = request.app.state.registry
    try:
        result = show_model(store=store, registry=registry, name=name)
    except OperationError as exc:
        raise _model_op_error_to_http(exc) from exc
    return {
        "name": result.name,
        "config": result.config,
        "layers": [
            {
                "media_type": layer.media_type,
                "digest": layer.digest,
                "size": layer.size,
                "filename": layer.filename,
            }
            for layer in result.layers
        ],
    }


@router.delete("/v1/models/{name:path}")
async def delete_model_route(name: str, request: Request):
    store = request.app.state.store
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry
    try:
        await delete_model(store=store, scheduler=scheduler, registry=registry, name=name)
    except OperationError as exc:
        raise _model_op_error_to_http(exc) from exc
    return {"status": "success"}
