from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from vox.operations import system as system_operations
from vox.operations.errors import ModelInUseError
from vox.operations.system import MemoryBudgetExceededError

router = APIRouter()


class TrimIdleRequest(BaseModel):
    min_idle_seconds: int = 0


class EnforceMemoryBudgetRequest(BaseModel):
    additional_vram_bytes: int = 0


@router.get("/v1/system/memory")
async def memory_status(request: Request):
    result = system_operations.get_memory_status(scheduler=request.app.state.scheduler)
    return system_operations.memory_status_payload(result)


@router.post("/v1/system/trim")
async def trim_idle(req: TrimIdleRequest, request: Request):
    result = await system_operations.trim_idle_memory(
        scheduler=request.app.state.scheduler,
        min_idle_seconds=req.min_idle_seconds,
    )
    return system_operations.trim_idle_payload(result)


@router.post("/v1/system/enforce-memory-budget")
async def enforce_memory_budget(req: EnforceMemoryBudgetRequest, request: Request):
    try:
        result = await system_operations.enforce_memory_budget(
            scheduler=request.app.state.scheduler,
            additional_vram_bytes=req.additional_vram_bytes,
        )
    except MemoryBudgetExceededError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    return system_operations.enforce_memory_budget_payload(result)


@router.post("/v1/models/{name:path}/trim")
async def trim_model(name: str, request: Request):
    try:
        result = await system_operations.trim_model(
            scheduler=request.app.state.scheduler,
            model_name=name,
        )
    except ModelInUseError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return system_operations.trim_model_payload(result)


@router.post("/v1/models/unload_idle")
async def unload_idle(request: Request):
    result = await system_operations.unload_idle_models(scheduler=request.app.state.scheduler)
    return system_operations.unload_idle_payload(result)
