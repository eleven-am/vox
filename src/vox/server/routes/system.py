from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from vox.operations import system as system_operations
from vox.server.app_services import app_scheduler
from vox.server.operation_errors import map_operation_errors_to_http

router = APIRouter()


class TrimIdleBody(BaseModel):
    min_idle_seconds: int = 0


class EnforceMemoryBudgetBody(BaseModel):
    additional_vram_bytes: int = 0


@router.get("/v1/system/memory")
async def memory_status(request: Request):
    scheduler = app_scheduler(request)
    result = system_operations.get_memory_status(
        scheduler=scheduler,
        request=system_operations.memory_status_request_from_fields(),
    )
    return system_operations.memory_status_payload(result)


@router.post("/v1/system/trim")
async def trim_idle(req: TrimIdleBody, request: Request):
    scheduler = app_scheduler(request)
    with map_operation_errors_to_http():
        result = await system_operations.trim_idle_memory(
            scheduler=scheduler,
            request=system_operations.trim_idle_memory_request_from_fields(
                min_idle_seconds=req.min_idle_seconds,
            ),
        )
    return system_operations.trim_idle_payload(result)


@router.post("/v1/system/enforce-memory-budget")
async def enforce_memory_budget(req: EnforceMemoryBudgetBody, request: Request):
    scheduler = app_scheduler(request)
    with map_operation_errors_to_http():
        result = await system_operations.enforce_memory_budget(
            scheduler=scheduler,
            request=system_operations.enforce_memory_budget_request_from_fields(
                additional_vram_bytes=req.additional_vram_bytes,
            ),
        )
    return system_operations.enforce_memory_budget_payload(result)


@router.post("/v1/models/{name:path}/trim")
async def trim_model(name: str, request: Request):
    scheduler = app_scheduler(request)
    with map_operation_errors_to_http():
        result = await system_operations.trim_model(
            scheduler=scheduler,
            request=system_operations.trim_model_request_from_fields(model_name=name),
        )
    return system_operations.trim_model_payload(result)


@router.post("/v1/models/unload_idle")
async def unload_idle(request: Request):
    scheduler = app_scheduler(request)
    with map_operation_errors_to_http():
        result = await system_operations.unload_idle_models(
            scheduler=scheduler,
            request=system_operations.unload_idle_models_request_from_fields(),
        )
    return system_operations.unload_idle_payload(result)
