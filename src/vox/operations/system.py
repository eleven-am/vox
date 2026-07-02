from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from vox.core.errors import ModelLoadError
from vox.core.types import LoadedModelInfo, VramSnapshot
from vox.operations.errors import MemoryBudgetExceededError, ModelInUseError


class SystemScheduler(Protocol):
    def list_loaded(self) -> list[LoadedModelInfo]: ...
    def memory_snapshot(self) -> VramSnapshot: ...
    async def trim(self, model_name: str) -> bool: ...
    async def trim_idle(self, *, min_idle_seconds: int = 0) -> list[str]: ...
    async def unload(self, model_name: str) -> bool: ...
    async def enforce_memory_budget(self, *, additional_vram_bytes: int = 0) -> None: ...


@dataclass(frozen=True)
class MemoryStatus:
    snapshot: VramSnapshot


@dataclass(frozen=True)
class TrimIdleResult:
    trimmed: list[str]
    snapshot: VramSnapshot


@dataclass(frozen=True)
class EnforceMemoryBudgetResult:
    snapshot: VramSnapshot


@dataclass(frozen=True)
class TrimModelResult:
    status: str = "success"


@dataclass(frozen=True)
class UnloadIdleResult:
    unloaded: list[str]
    snapshot: VramSnapshot


def loaded_model_payload(model: LoadedModelInfo) -> dict[str, Any]:
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


def memory_snapshot_payload(snapshot: VramSnapshot) -> dict[str, Any]:
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
        "models": [loaded_model_payload(model) for model in snapshot.loaded_models],
    }


def memory_status_payload(result: MemoryStatus) -> dict[str, Any]:
    return memory_snapshot_payload(result.snapshot)


def trim_idle_payload(result: TrimIdleResult) -> dict[str, Any]:
    return {
        "trimmed": result.trimmed,
        "memory": memory_snapshot_payload(result.snapshot),
    }


def enforce_memory_budget_payload(result: EnforceMemoryBudgetResult) -> dict[str, Any]:
    return {
        "status": "ok",
        "memory": memory_snapshot_payload(result.snapshot),
    }


def trim_model_payload(result: TrimModelResult) -> dict[str, Any]:
    return {"status": result.status}


def unload_idle_payload(result: UnloadIdleResult) -> dict[str, Any]:
    return {
        "unloaded": result.unloaded,
        "memory": memory_snapshot_payload(result.snapshot),
    }


def get_memory_status(*, scheduler: SystemScheduler) -> MemoryStatus:
    return MemoryStatus(snapshot=scheduler.memory_snapshot())


async def trim_idle_memory(
    *,
    scheduler: SystemScheduler,
    min_idle_seconds: int = 0,
) -> TrimIdleResult:
    trimmed = await scheduler.trim_idle(min_idle_seconds=max(0, min_idle_seconds))
    return TrimIdleResult(trimmed=trimmed, snapshot=scheduler.memory_snapshot())


async def enforce_memory_budget(
    *,
    scheduler: SystemScheduler,
    additional_vram_bytes: int = 0,
) -> EnforceMemoryBudgetResult:
    try:
        await scheduler.enforce_memory_budget(
            additional_vram_bytes=max(0, additional_vram_bytes),
        )
    except ModelLoadError as exc:
        raise MemoryBudgetExceededError(str(exc)) from exc
    return EnforceMemoryBudgetResult(snapshot=scheduler.memory_snapshot())


async def trim_model(*, scheduler: SystemScheduler, model_name: str) -> TrimModelResult:
    trimmed = await scheduler.trim(model_name)
    if not trimmed:
        raise ModelInUseError(model_name)
    return TrimModelResult()


async def unload_idle_models(*, scheduler: SystemScheduler) -> UnloadIdleResult:
    unloaded: list[str] = []
    for model in scheduler.list_loaded():
        if model.ref_count > 0:
            continue
        ref = f"{model.name}:{model.tag}"
        if await scheduler.unload(ref):
            unloaded.append(ref)
    return UnloadIdleResult(unloaded=unloaded, snapshot=scheduler.memory_snapshot())
