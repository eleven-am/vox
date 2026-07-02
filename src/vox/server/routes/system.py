from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from vox.operations import system as system_operations
from vox.server.operation_errors import map_operation_errors_to_http

router = APIRouter()


class TrimIdleBody(BaseModel):
    min_idle_seconds: int = 0


class EnforceMemoryBudgetBody(BaseModel):
    additional_vram_bytes: int = 0


@router.get("/v1/system/memory")
async def memory_status(request: Request):
    result = system_operations.get_memory_status(
        scheduler=request.app.state.scheduler,
        request=system_operations.MemoryStatusRequest(),
    )
    return system_operations.memory_status_payload(result)


@router.post("/v1/system/trim")
async def trim_idle(req: TrimIdleBody, request: Request):
    result = await system_operations.trim_idle_memory(
        scheduler=request.app.state.scheduler,
        request=system_operations.TrimIdleMemoryRequest(
            min_idle_seconds=req.min_idle_seconds,
        ),
    )
    return system_operations.trim_idle_payload(result)


@router.post("/v1/system/enforce-memory-budget")
async def enforce_memory_budget(req: EnforceMemoryBudgetBody, request: Request):
    with map_operation_errors_to_http():
        result = await system_operations.enforce_memory_budget(
            scheduler=request.app.state.scheduler,
            request=system_operations.EnforceMemoryBudgetRequest(
                additional_vram_bytes=req.additional_vram_bytes,
            ),
        )
    return system_operations.enforce_memory_budget_payload(result)


@router.post("/v1/models/{name:path}/trim")
async def trim_model(name: str, request: Request):
    with map_operation_errors_to_http():
        result = await system_operations.trim_model(
            scheduler=request.app.state.scheduler,
            request=system_operations.TrimModelRequest(model_name=name),
        )
    return system_operations.trim_model_payload(result)


@router.post("/v1/models/unload_idle")
async def unload_idle(request: Request):
    result = await system_operations.unload_idle_models(
        scheduler=request.app.state.scheduler,
        request=system_operations.UnloadIdleModelsRequest(),
    )
    return system_operations.unload_idle_payload(result)
