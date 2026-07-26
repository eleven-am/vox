from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from vox.core.errors import ModelTrimUnsupportedError
from vox.core.types import LoadedModelInfo, VramSnapshot
from vox.operations.errors import InvalidConfigError, ModelInUseError


class SystemScheduler(Protocol):
    def list_loaded(self) -> list[LoadedModelInfo]: ...
    def memory_snapshot(self) -> VramSnapshot: ...
    async def trim(self, model_name: str) -> bool: ...
    async def trim_idle(self, *, min_idle_seconds: int = 0) -> list[str]: ...
    async def unload(self, model_name: str) -> bool: ...


@dataclass(frozen=True)
class MemoryStatus:
    snapshot: VramSnapshot


@dataclass(frozen=True)
class HealthStatusRequest:
    pass


@dataclass(frozen=True)
class HealthStatusResult:
    status: str = "ok"


@dataclass(frozen=True)
class ListLoadedModelsRequest:
    pass


@dataclass(frozen=True)
class ListLoadedModelsResult:
    models: list[LoadedModelInfo]


@dataclass(frozen=True)
class MemoryStatusRequest:
    pass


@dataclass(frozen=True)
class TrimIdleMemoryRequest:
    min_idle_seconds: int = 0


@dataclass(frozen=True)
class TrimModelRequest:
    model_name: str


@dataclass(frozen=True)
class UnloadIdleModelsRequest:
    pass


def memory_status_request_from_fields() -> MemoryStatusRequest:
    return MemoryStatusRequest()


def health_status_request_from_fields() -> HealthStatusRequest:
    return HealthStatusRequest()


def list_loaded_models_request_from_fields() -> ListLoadedModelsRequest:
    return ListLoadedModelsRequest()


def trim_idle_memory_request_from_fields(*, min_idle_seconds: int = 0) -> TrimIdleMemoryRequest:
    return TrimIdleMemoryRequest(min_idle_seconds=min_idle_seconds)


def trim_model_request_from_fields(*, model_name: str) -> TrimModelRequest:
    return TrimModelRequest(model_name=model_name)


def unload_idle_models_request_from_fields() -> UnloadIdleModelsRequest:
    return UnloadIdleModelsRequest()


@dataclass(frozen=True)
class TrimIdleResult:
    trimmed: list[str]
    snapshot: VramSnapshot


@dataclass(frozen=True)
class TrimModelResult:
    status: str = "success"


@dataclass(frozen=True)
class UnloadIdleResult:
    unloaded: list[str]
    snapshot: VramSnapshot


def _loaded_model_base_payload(model: LoadedModelInfo) -> dict[str, Any]:
    return {
        "name": model.name,
        "tag": model.tag,
        "type": model.type.value,
        "device": model.device,
        "vram_bytes": model.vram_bytes,
        "loaded_at": model.loaded_at,
        "last_used": model.last_used,
        "ref_count": model.ref_count,
    }


def loaded_model_payload(model: LoadedModelInfo) -> dict[str, Any]:
    return {
        **_loaded_model_base_payload(model),
        "is_evictable": model.is_evictable,
        "is_trimmable": model.is_trimmable,
        "backend_memory": model.backend_memory,
    }


def loaded_model_summary_payload(model: LoadedModelInfo) -> dict[str, Any]:
    return _loaded_model_base_payload(model)


def list_loaded_models_payload(result: ListLoadedModelsResult) -> dict[str, Any]:
    return {"models": [loaded_model_summary_payload(model) for model in result.models]}


def memory_snapshot_payload(snapshot: VramSnapshot) -> dict[str, Any]:
    return {
        "device": {
            "device": snapshot.device.device,
            "free_bytes": snapshot.device.free_bytes,
            "total_bytes": snapshot.device.total_bytes,
            "torch_allocated_bytes": snapshot.device.torch_allocated_bytes,
            "torch_reserved_bytes": snapshot.device.torch_reserved_bytes,
        },
        "idle_trim_seconds": snapshot.idle_trim_seconds,
        "estimated_loaded_vram_bytes": snapshot.estimated_loaded_vram_bytes,
        "active_model_count": snapshot.active_model_count,
        "models": [loaded_model_payload(model) for model in snapshot.loaded_models],
    }


def memory_status_payload(result: MemoryStatus) -> dict[str, Any]:
    return memory_snapshot_payload(result.snapshot)


def health_status_payload(result: HealthStatusResult) -> dict[str, str]:
    return {"status": result.status}


def trim_idle_payload(result: TrimIdleResult) -> dict[str, Any]:
    return {
        "trimmed": result.trimmed,
        "memory": memory_snapshot_payload(result.snapshot),
    }


def trim_model_payload(result: TrimModelResult) -> dict[str, Any]:
    return {"status": result.status}


def unload_idle_payload(result: UnloadIdleResult) -> dict[str, Any]:
    return {
        "unloaded": result.unloaded,
        "memory": memory_snapshot_payload(result.snapshot),
    }


def get_memory_status(
    *,
    scheduler: SystemScheduler,
    request: MemoryStatusRequest,
) -> MemoryStatus:
    _ = request
    return MemoryStatus(snapshot=scheduler.memory_snapshot())


def get_health_status(*, request: HealthStatusRequest) -> HealthStatusResult:
    _ = request
    return HealthStatusResult()


def list_loaded_models(
    *,
    scheduler: SystemScheduler,
    request: ListLoadedModelsRequest,
) -> ListLoadedModelsResult:
    _ = request
    return ListLoadedModelsResult(models=scheduler.list_loaded())


async def trim_idle_memory(
    *,
    scheduler: SystemScheduler,
    request: TrimIdleMemoryRequest,
) -> TrimIdleResult:
    trimmed = await scheduler.trim_idle(min_idle_seconds=max(0, request.min_idle_seconds))
    return TrimIdleResult(trimmed=trimmed, snapshot=scheduler.memory_snapshot())


async def trim_model(*, scheduler: SystemScheduler, request: TrimModelRequest) -> TrimModelResult:
    try:
        trimmed = await scheduler.trim(request.model_name)
    except ModelTrimUnsupportedError as exc:
        raise InvalidConfigError(str(exc)) from exc
    if not trimmed:
        raise ModelInUseError(request.model_name)
    return TrimModelResult()


async def unload_idle_models(
    *,
    scheduler: SystemScheduler,
    request: UnloadIdleModelsRequest,
) -> UnloadIdleResult:
    _ = request
    unloaded: list[str] = []
    for model in scheduler.list_loaded():
        if model.ref_count > 0:
            continue
        ref = f"{model.name}:{model.tag}"
        if await scheduler.unload(ref):
            unloaded.append(ref)
    return UnloadIdleResult(unloaded=unloaded, snapshot=scheduler.memory_snapshot())
