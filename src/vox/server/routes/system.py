from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from vox.core.errors import ModelLoadError
from vox.core.types import LoadedModelInfo, VramSnapshot

router = APIRouter()


class TrimIdleRequest(BaseModel):
    min_idle_seconds: int = 0


class EnforceMemoryBudgetRequest(BaseModel):
    additional_vram_bytes: int = 0


def _loaded_model_payload(model: LoadedModelInfo) -> dict:
    return {
        "name": model.name,
        "tag": model.tag,
        "type": model.type.value,
        "device": model.device,
        "vram_bytes": model.vram_bytes,
        "loaded_at": model.loaded_at,
        "last_used": model.last_used,
        "ref_count": model.ref_count,
        "is_evictable": model.is_evictable,
        "is_trimmable": model.is_trimmable,
        "backend_memory": model.backend_memory,
    }


def _snapshot_payload(snapshot: VramSnapshot) -> dict:
    return {
        "policy": {
            "max_vram_bytes": snapshot.policy.max_vram_bytes,
            "headroom_bytes": snapshot.policy.headroom_bytes,
            "idle_trim_seconds": snapshot.policy.idle_trim_seconds,
            "over_budget": snapshot.policy.over_budget,
        },
        "device": {
            "device": snapshot.device.device,
            "free_bytes": snapshot.device.free_bytes,
            "total_bytes": snapshot.device.total_bytes,
            "torch_allocated_bytes": snapshot.device.torch_allocated_bytes,
            "torch_reserved_bytes": snapshot.device.torch_reserved_bytes,
        },
        "estimated_loaded_vram_bytes": snapshot.estimated_loaded_vram_bytes,
        "active_model_count": snapshot.active_model_count,
        "models": [_loaded_model_payload(model) for model in snapshot.loaded_models],
    }


@router.get("/v1/system/memory")
async def memory_status(request: Request):
    return _snapshot_payload(request.app.state.scheduler.memory_snapshot())


@router.post("/v1/system/trim")
async def trim_idle(req: TrimIdleRequest, request: Request):
    trimmed = await request.app.state.scheduler.trim_idle(min_idle_seconds=max(0, req.min_idle_seconds))
    return {"trimmed": trimmed, "memory": _snapshot_payload(request.app.state.scheduler.memory_snapshot())}


@router.post("/v1/system/enforce-memory-budget")
async def enforce_memory_budget(req: EnforceMemoryBudgetRequest, request: Request):
    try:
        await request.app.state.scheduler.enforce_memory_budget(
            additional_vram_bytes=max(0, req.additional_vram_bytes)
        )
    except ModelLoadError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    return {"status": "ok", "memory": _snapshot_payload(request.app.state.scheduler.memory_snapshot())}


@router.post("/v1/models/{name:path}/trim")
async def trim_model(name: str, request: Request):
    trimmed = await request.app.state.scheduler.trim(name)
    if not trimmed:
        raise HTTPException(status_code=409, detail=f"Model {name} is currently in use")
    return {"status": "success"}


@router.post("/v1/models/unload_idle")
async def unload_idle(request: Request):
    unloaded: list[str] = []
    for model in request.app.state.scheduler.list_loaded():
        if model.ref_count > 0:
            continue
        ref = f"{model.name}:{model.tag}"
        if await request.app.state.scheduler.unload(ref):
            unloaded.append(ref)
    return {"unloaded": unloaded, "memory": _snapshot_payload(request.app.state.scheduler.memory_snapshot())}
