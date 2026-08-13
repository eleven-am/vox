from __future__ import annotations

import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from vox.core.adapter import STTAdapter, TTSAdapter, TurnDetectorAdapter
from vox.core.adapter_runtime import run_with_adapter_runtime_lock
from vox.core.device_placement import (
    LoadedModelView,
    Placement,
    decide_placement,
    detect_capabilities,
)
from vox.core.errors import ModelLoadError, ModelTrimUnsupportedError
from vox.core.process_memory import cgroup_memory_status, process_memory_status
from vox.core.runtime import detect_runtime_capabilities
from vox.core.tasks import reap_task
from vox.core.types import (
    DeviceMemoryInfo,
    LoadedModelInfo,
    ModelInfo,
    ProcessMemoryInfo,
    VramSnapshot,
    parse_model_name,
)

logger = logging.getLogger(__name__)

Adapter = STTAdapter | TTSAdapter | TurnDetectorAdapter


def _detect_device() -> str:
    """Auto-detect best available device."""
    capabilities = detect_runtime_capabilities()
    if capabilities.torch_cuda or capabilities.onnx_cuda or capabilities.nvidia_device:
        device = "cuda"
    elif capabilities.mps:
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Auto-detected device: {device}")
    return device


def _clear_gpu_cache() -> None:
    """Clear CUDA/MPS cache and run garbage collection."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    except RuntimeError as e:
        logger.warning(f"Failed to clear GPU cache: {e}")
    gc.collect()


def _teardown_adapters_blocking(items: list[tuple[str, Any, str]]) -> dict[str, str | None]:
    """Unload adapters and clear the GPU cache. Blocking; run off the event loop."""
    results: dict[str, str | None] = {}
    for full_name, adapter, reason in items:
        results[full_name] = _unload_adapter_blocking(full_name, adapter, reason)
    _clear_gpu_cache()
    return results


def _unload_adapter_blocking(full_name: str, adapter: Any, reason: str) -> str | None:
    try:
        adapter.unload()
    except Exception as error:
        logger.error("Error unloading %s during %s: %s", full_name, reason, error)
        return str(error)
    try:
        adapter.close_execution_lane()
    except Exception as error:
        logger.error("Error closing execution lane for %s during %s: %s", full_name, reason, error)
        return str(error)
    return None


def _reset_adapter_after_failed_load(
    full_name: str,
    adapter: Any,
    reason: str,
) -> str | None:
    try:
        adapter.unload()
    except Exception as error:
        logger.error("Error resetting %s during %s: %s", full_name, reason, error)
        return str(error)
    _clear_gpu_cache()
    return None


def _device_memory_snapshot(device: str) -> DeviceMemoryInfo:
    """Return a best-effort memory snapshot for an accelerator device."""
    free_bytes = _available_device_memory_bytes(device)
    total_bytes = _total_device_memory_bytes(device)
    allocated_bytes = None
    reserved_bytes = None

    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                allocated_bytes = int(torch.cuda.memory_allocated())
                reserved_bytes = int(torch.cuda.memory_reserved())
        except ImportError:
            pass
        except RuntimeError as error:
            logger.warning("Failed to query torch %s memory: %s", device, error)

    return DeviceMemoryInfo(
        device=device,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        torch_allocated_bytes=allocated_bytes,
        torch_reserved_bytes=reserved_bytes,
    )


def _is_oom_error(error: Exception) -> bool:
    """Check if an exception is an out-of-memory error."""
    oom_keywords = ["out of memory", "cuda oom", "onnxruntime oom", "failed to allocate"]
    msg = str(error).lower()
    return any(kw in msg for kw in oom_keywords)


def _available_device_memory_bytes(device: str) -> int | None:
    """Return free accelerator memory for *device* when the backend exposes it."""
    if device != "cuda":
        return None

    try:
        import torch

        if not torch.cuda.is_available() or not hasattr(torch.cuda, "mem_get_info"):
            return None
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes)
    except ImportError:
        return None
    except RuntimeError as error:
        logger.warning("Failed to query free %s memory: %s", device, error)
        return None


def _total_device_memory_bytes(device: str) -> int | None:
    if device != "cuda":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if hasattr(torch.cuda, "mem_get_info"):
            _free, total = torch.cuda.mem_get_info()
            return int(total)
        properties = torch.cuda.get_device_properties(0)
        total_memory = getattr(properties, "total_memory", None)
        return int(total_memory) if total_memory is not None else None
    except ImportError:
        return None
    except RuntimeError as error:
        logger.warning("Failed to query total %s memory: %s", device, error)
        return None


class RegistryProtocol(Protocol):
    def resolve(self, name: str, tag: str) -> tuple[ModelInfo, Path]: ...
    def resolve_model_ref(self, name: str, tag: str = "latest", *, explicit_tag: bool = False) -> tuple[str, str]: ...
    def get_adapter_class(self, adapter_name: str) -> type: ...


@dataclass
class _LoadedModel:
    """Internal state for a loaded model."""

    full_name: str
    info: ModelInfo
    adapter: Adapter
    device: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    ref_count: int = 0
    vram_bytes: int = 0
    trimmed: bool = False
    maintenance: str | None = None
    lifecycle_error: str | None = None
    maintenance_done: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.maintenance_done.set()

    @property
    def physical_work_count(self) -> int:
        return self.adapter.physical_work_count

    @property
    def active_count(self) -> int:
        return self.ref_count + self.physical_work_count + int(self.maintenance is not None)

    @property
    def is_busy(self) -> bool:
        return self.active_count > 0

    @property
    def has_work(self) -> bool:
        return self.ref_count > 0 or self.physical_work_count > 0


class Scheduler:
    def __init__(
        self,
        registry: RegistryProtocol,
        *,
        default_device: str = "auto",
        max_loaded: int = 3,
        ttl_seconds: int = 300,
        cleanup_interval: int = 30,
        idle_trim_seconds: int = 0,
        shutdown_timeout_seconds: float = 30.0,
    ) -> None:
        self._registry = registry
        self._requested_device = default_device
        self._default_device = default_device
        self._max_loaded = max_loaded
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._idle_trim_seconds = max(0, int(idle_trim_seconds))
        self._shutdown_timeout_seconds = max(0.1, float(shutdown_timeout_seconds))
        self._models: dict[str, _LoadedModel] = {}
        self._orphaned_models: dict[int, _LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._load_tasks: dict[str, asyncio.Task[_LoadedModel]] = {}
        self._maintenance_tasks: set[asyncio.Task[Any]] = set()
        self._cleanup_task: asyncio.Task | None = None
        self._stopping = False

    def _normalize_model_ref(self, model_name: str) -> str:
        """Resolve aliases so all cache keys use the canonical registry ref."""
        explicit_tag = ":" in model_name
        name, tag = parse_model_name(model_name)
        resolved_name, resolved_tag = self._registry.resolve_model_ref(name, tag, explicit_tag=explicit_tag)
        return f"{resolved_name}:{resolved_tag}"

    def _infer_loaded_device(self, adapter: Adapter, info: ModelInfo, requested_device: str) -> str:
        """Report the actual device used by an adapter after load."""
        actual_device = getattr(adapter, "_device", None)
        if isinstance(actual_device, str) and actual_device and actual_device != "auto":
            return actual_device

        if requested_device == "auto":
            from vox.core.device_placement import auto_device_for_model

            candidate = auto_device_for_model(info, detect_capabilities())
            return candidate if candidate != "auto" else "cpu"

        return requested_device

    async def start(self) -> None:
        """Start the background TTL cleanup loop."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._ttl_cleanup_loop())

    async def stop(self, *, deadline: float | None = None) -> None:
        deadline = time.monotonic() + self._shutdown_timeout_seconds if deadline is None else deadline
        async with self._lock:
            self._stopping = True
        if self._cleanup_task:
            remaining = self._shutdown_remaining(deadline, "cleanup")
            await reap_task(self._cleanup_task, timeout=remaining)
            self._cleanup_task = None
        async with self._lock:
            load_tasks = tuple(self._load_tasks.values())
        await self._wait_for_shutdown_tasks(load_tasks, deadline, "model loads")
        await self._drain_maintenance_tasks(deadline)
        await self._wait_for_idle_models(deadline)
        await self._drain_maintenance_tasks(deadline)
        await self.unload_all(deadline=deadline)
        await self._unload_orphans(deadline=deadline)
        if self._models or self._orphaned_models:
            names = ", ".join(
                sorted(
                    {
                        *self._models,
                        *(loaded.full_name for loaded in self._orphaned_models.values()),
                    }
                )
            )
            raise RuntimeError(f"scheduler shutdown left loaded models: {names}")

    @staticmethod
    def _shutdown_remaining(deadline: float, phase: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"scheduler shutdown timed out waiting for {phase}")
        return remaining

    async def _wait_for_shutdown_tasks(
        self,
        tasks: tuple[asyncio.Task[Any], ...],
        deadline: float,
        phase: str,
    ) -> None:
        if not tasks:
            return
        remaining = self._shutdown_remaining(deadline, phase)
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        if pending:
            raise RuntimeError(f"scheduler shutdown timed out waiting for {phase}")

    async def _drain_maintenance_tasks(self, deadline: float) -> None:
        while self._maintenance_tasks:
            tasks = tuple(self._maintenance_tasks)
            await self._wait_for_shutdown_tasks(tasks, deadline, "model maintenance")
            await asyncio.sleep(0)

    async def _wait_for_idle_models(self, deadline: float) -> None:
        while True:
            async with self._lock:
                busy = tuple(
                    model for model in (*self._models.values(), *self._orphaned_models.values()) if model.has_work
                )
            if not busy:
                return
            names = ", ".join(sorted(model.full_name for model in busy))
            remaining = self._shutdown_remaining(deadline, f"active models: {names}")
            physical = tuple(model.adapter for model in busy if model.physical_work_count > 0)
            if physical:
                await asyncio.gather(
                    *(adapter.wait_execution_idle(timeout=min(remaining, 0.25)) for adapter in physical),
                    return_exceptions=True,
                )
            else:
                await asyncio.sleep(min(0.01, remaining))

    def _estimate_model_memory_bytes(self, adapter: Adapter, info: ModelInfo, model_path: Path) -> int:
        """Estimate accelerator memory required for *info* before loading."""
        estimate_kwargs = {**info.parameters, "model_path": str(model_path)}
        try:
            estimate = adapter.estimate_vram_bytes(**estimate_kwargs)
        except TypeError:
            estimate = adapter.estimate_vram_bytes()
        return max(int(estimate or 0), 0)

    def _loaded_model_views(self) -> list[LoadedModelView]:
        return [
            LoadedModelView(
                full_name=m.full_name,
                device=m.device,
                vram_bytes=m.vram_bytes,
                ref_count=1 if m.is_busy else 0,
                last_used=m.last_used,
            )
            for m in self._models.values()
        ]

    @staticmethod
    def _begin_maintenance(loaded: _LoadedModel, action: str) -> None:
        loaded.maintenance = action
        loaded.maintenance_done.clear()

    @staticmethod
    def _finish_maintenance(loaded: _LoadedModel) -> None:
        loaded.maintenance = None
        loaded.maintenance_done.set()

    def _start_maintenance_task(self, coroutine) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        self._maintenance_tasks.add(task)

        def completed(done: asyncio.Task[Any]) -> None:
            self._maintenance_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)
        return task

    @staticmethod
    async def _await_owned_task(task: asyncio.Task[Any]) -> Any:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise

    async def _complete_unloads(
        self,
        items: list[tuple[str, _LoadedModel, str]],
    ) -> dict[str, str | None]:
        results = await self._teardown_off_loop(items)
        async with self._lock:
            for name, loaded, _reason in items:
                if self._models.get(name) is not loaded:
                    continue
                error = results.get(name)
                if error is None:
                    self._finish_maintenance(loaded)
                    del self._models[name]
                else:
                    loaded.lifecycle_error = error
                    self._finish_maintenance(loaded)
        return results

    async def _complete_orphan_teardown(
        self,
        full_name: str,
        loaded: _LoadedModel,
        reason: str,
    ) -> str | None:
        try:
            await loaded.adapter.wait_execution_idle()
        except Exception as error:
            result = str(error)
        else:
            result = (await self._teardown_off_loop([(full_name, loaded, reason)])).get(full_name)
        async with self._lock:
            if self._orphaned_models.get(id(loaded)) is loaded:
                if result is None:
                    self._orphaned_models.pop(id(loaded), None)
                else:
                    loaded.lifecycle_error = result
                self._finish_maintenance(loaded)
        return result

    def _schedule_orphan_teardown_locked(
        self,
        full_name: str,
        loaded: _LoadedModel,
        reason: str,
    ) -> asyncio.Task[Any] | None:
        if loaded.maintenance is not None:
            return None
        self._begin_maintenance(loaded, reason)
        return self._start_maintenance_task(
            self._complete_orphan_teardown(
                full_name,
                loaded,
                reason,
            )
        )

    async def _complete_trims(
        self,
        items: list[tuple[str, _LoadedModel]],
    ) -> list[str]:
        trimmed: list[str] = []
        try:
            trimmed = await self._trim_off_loop(items)
            return trimmed
        finally:
            async with self._lock:
                for full_name, loaded in items:
                    if self._models.get(full_name) is loaded:
                        if full_name in trimmed:
                            loaded.trimmed = True
                        self._finish_maintenance(loaded)

    async def _execute_evictions(self, names: list[str]) -> None:
        for full_name in names:
            async with self._lock:
                loaded = self._models.get(full_name)
                if loaded is None:
                    continue
                if loaded.is_busy:
                    raise ModelLoadError(f"Cannot evict {full_name}: model became active")
                self._begin_maintenance(loaded, "memory eviction")
            logger.info(
                "Evicting %s to free %s memory for a new load (idle for %.0fs)",
                full_name,
                loaded.device,
                time.time() - loaded.last_used,
            )
            task = self._start_maintenance_task(self._complete_unloads([(full_name, loaded, "memory eviction")]))
            results = await self._await_owned_task(task)
            error = results.get(full_name)
            if error is not None:
                raise ModelLoadError(f"Failed to evict {full_name}: {error}")

    def _trim_loaded_model(self, full_name: str, loaded: _LoadedModel) -> bool:
        if not loaded.adapter.supports_trim:
            return False
        logger.info("Trimming non-essential memory for %s", full_name)
        before = self._adapter_memory_status(loaded)

        def trim_and_clear() -> None:
            loaded.adapter.trim()
            _clear_gpu_cache()

        try:
            loaded.adapter.run_exclusive(trim_and_clear)
        except Exception as error:
            logger.error("Error trimming %s: %s", full_name, error)
            return False
        after = self._adapter_memory_status(loaded)
        logger.info(
            "Trimmed non-essential memory for %s rss_bytes=%s->%s torch_reserved_bytes=%s->%s",
            full_name,
            before.get("rss_bytes"),
            after.get("rss_bytes"),
            before.get("torch_reserved_bytes"),
            after.get("torch_reserved_bytes"),
        )
        return True

    def _select_trimmable_locked(self, *, min_idle_seconds: int = 0) -> list[tuple[str, _LoadedModel]]:
        now = time.time()
        eligible: list[tuple[str, _LoadedModel]] = []
        for full_name, loaded in list(self._models.items()):
            if loaded.is_busy:
                continue
            if loaded.trimmed:
                continue
            if not loaded.adapter.supports_trim:
                continue
            if min_idle_seconds > 0 and now - loaded.last_used < min_idle_seconds:
                continue
            eligible.append((full_name, loaded))
        return eligible

    def _trim_models_blocking(self, items: list[tuple[str, _LoadedModel]]) -> list[str]:
        return [full_name for full_name, loaded in items if self._trim_loaded_model(full_name, loaded)]

    async def _trim_off_loop(self, items: list[tuple[str, _LoadedModel]]) -> list[str]:
        if not items:
            return []
        return await asyncio.to_thread(self._trim_models_blocking, items)

    def _adapter_memory_status(self, loaded: _LoadedModel) -> dict:
        try:
            status = dict(loaded.adapter.memory_status())
            status["physical_work_count"] = loaded.physical_work_count
            status["maintenance"] = loaded.maintenance
            status["lifecycle_error"] = loaded.lifecycle_error
            return status
        except Exception as error:
            logger.warning("Failed to query memory status for %s: %s", loaded.full_name, error)
            return {"error": str(error)}

    def _decide_placement(self, adapter: Adapter, info: ModelInfo, estimated_vram_bytes: int) -> Placement:
        capabilities = detect_capabilities()
        tiers = adapter.placement_tiers()
        return decide_placement(
            info,
            requested_device=self._requested_device,
            capabilities=capabilities,
            loaded_models=self._loaded_model_views(),
            estimated_vram_bytes=estimated_vram_bytes,
            free_memory_query=_available_device_memory_bytes,
            total_memory_query=_total_device_memory_bytes,
            tiers=tiers,
        )

    async def _load_model(self, full_name: str) -> _LoadedModel:
        """Load a model by name. Handles eviction and OOM fallback."""
        full_name = self._normalize_model_ref(full_name)
        name, tag = parse_model_name(full_name)

        info, model_path = self._registry.resolve(name, tag)
        adapter_cls = self._registry.get_adapter_class(info.adapter)

        adapter = adapter_cls()
        estimated_vram_bytes = self._estimate_model_memory_bytes(adapter, info, model_path)
        placement = self._decide_placement(adapter, info, estimated_vram_bytes)
        if placement.evict:
            await self._execute_evictions(placement.evict)
        device = placement.device

        if len(self._models) >= self._max_loaded:
            await self._evict_lru()
            if len(self._models) >= self._max_loaded:
                raise ModelLoadError(
                    f"Cannot load {full_name}: all {self._max_loaded} model slots are in use. "
                    "Wait for an active request to finish or increase --max-loaded."
                )

        load_kwargs = {**info.parameters}
        if placement.tier is not None:
            load_kwargs["_placement_tier"] = placement.tier
        if placement.notes:
            load_kwargs["_placement_extras"] = dict(placement.notes)
        try:
            logger.info(f"Loading {full_name} on {device}")
            start = time.perf_counter()
            await asyncio.to_thread(
                run_with_adapter_runtime_lock,
                adapter.load,
                str(model_path),
                device,
                **load_kwargs,
            )
            elapsed = time.perf_counter() - start
            logger.info(f"Loaded {full_name} in {elapsed:.2f}s on {device}")
        except Exception as e:
            if _is_oom_error(e) and device != "cpu":
                logger.warning(f"OOM loading {full_name} on {device}, falling back to CPU")
                reset_error = await asyncio.to_thread(
                    run_with_adapter_runtime_lock,
                    _reset_adapter_after_failed_load,
                    full_name,
                    adapter,
                    "accelerator OOM fallback",
                )
                if reset_error is not None:
                    cleanup_error = await asyncio.to_thread(
                        run_with_adapter_runtime_lock,
                        _unload_adapter_blocking,
                        full_name,
                        adapter,
                        "failed load cleanup",
                    )
                    detail = f"; cleanup failed: {cleanup_error}" if cleanup_error else ""
                    raise ModelLoadError(
                        f"Failed to reset {full_name} after accelerator OOM: {reset_error}{detail}"
                    ) from e
                device = "cpu"
                load_kwargs.pop("_placement_tier", None)
                load_kwargs.pop("_placement_extras", None)
                try:
                    await asyncio.to_thread(
                        run_with_adapter_runtime_lock,
                        adapter.load,
                        str(model_path),
                        device,
                        **load_kwargs,
                    )
                except Exception as e2:
                    cleanup_error = await asyncio.to_thread(
                        run_with_adapter_runtime_lock,
                        _unload_adapter_blocking,
                        full_name,
                        adapter,
                        "failed load cleanup",
                    )
                    detail = f"; cleanup failed: {cleanup_error}" if cleanup_error else ""
                    raise ModelLoadError(f"Failed to load {full_name}: {e2}{detail}") from e2
            else:
                cleanup_error = await asyncio.to_thread(
                    run_with_adapter_runtime_lock,
                    _unload_adapter_blocking,
                    full_name,
                    adapter,
                    "failed load cleanup",
                )
                detail = f"; cleanup failed: {cleanup_error}" if cleanup_error else ""
                raise ModelLoadError(f"Failed to load {full_name}: {e}{detail}") from e

        actual_device = self._infer_loaded_device(adapter, info, device)
        loaded = _LoadedModel(
            full_name=full_name,
            info=info,
            adapter=adapter,
            device=actual_device,
            vram_bytes=estimated_vram_bytes if actual_device != "cpu" else 0,
        )
        async with self._lock:
            stopped_while_loading = self._stopping
            if not stopped_while_loading:
                self._models[full_name] = loaded
        if stopped_while_loading:
            cleanup_error = await asyncio.to_thread(
                run_with_adapter_runtime_lock,
                _unload_adapter_blocking,
                full_name,
                adapter,
                "load completed after scheduler stop",
            )
            detail = f"; cleanup failed: {cleanup_error}" if cleanup_error else ""
            raise ModelLoadError(f"Scheduler stopped while loading {full_name}{detail}")
        return loaded

    async def _evict_lru(self) -> None:
        """Evict the least-recently-used model with no logical or physical work."""
        async with self._lock:
            candidates = [(name, m) for name, m in self._models.items() if not m.is_busy]
            if not candidates:
                logger.warning("Cannot evict: all loaded models are in use")
                return
            candidates.sort(key=lambda x: x[1].last_used)
            lru_name, lru_model = candidates[0]
            self._begin_maintenance(lru_model, "eviction")
        logger.info(f"Evicting {lru_name} (idle since {time.time() - lru_model.last_used:.0f}s ago)")
        task = self._start_maintenance_task(self._complete_unloads([(lru_name, lru_model, "eviction")]))
        results = await self._await_owned_task(task)
        error = results.get(lru_name)
        if error is not None:
            raise ModelLoadError(f"Failed to evict {lru_name}: {error}")

    async def _load_model_owned(self, full_name: str) -> _LoadedModel:
        task = asyncio.current_task()
        assert task is not None
        try:
            async with self._lifecycle_lock:
                async with self._lock:
                    loaded = self._models.get(full_name)
                if loaded is not None:
                    return loaded
                return await self._load_model(full_name)
        finally:
            async with self._lock:
                if self._load_tasks.get(full_name) is task:
                    del self._load_tasks[full_name]

    async def _ensure_loaded(self, full_name: str) -> _LoadedModel:
        async with self._lock:
            if self._stopping:
                raise ModelLoadError(f"Scheduler is stopping; cannot load {full_name}")
            loaded = self._models.get(full_name)
            if loaded is not None:
                return loaded
            task = self._load_tasks.get(full_name)
            if task is None:
                task = asyncio.create_task(self._load_model_owned(full_name))
                self._load_tasks[full_name] = task

                def completed(done: asyncio.Task[_LoadedModel]) -> None:
                    if not done.cancelled():
                        done.exception()

                task.add_done_callback(completed)
        return await asyncio.shield(task)

    async def _acquire_loaded(self, full_name: str) -> _LoadedModel:
        while True:
            maintenance_done: asyncio.Event | None = None
            async with self._lock:
                if self._stopping:
                    raise ModelLoadError(f"Scheduler is stopping; cannot acquire {full_name}")
                loaded = self._models.get(full_name)
                if loaded is not None and loaded.maintenance is not None:
                    maintenance_done = loaded.maintenance_done
                    loaded = None
                if loaded is not None and loaded.lifecycle_error is not None:
                    raise ModelLoadError(
                        f"Model {full_name} has an unresolved lifecycle failure: {loaded.lifecycle_error}"
                    )
                if loaded is not None and not loaded.adapter.is_loaded:
                    logger.warning(
                        "Retiring %s from cache: adapter reports unloaded with %d active references",
                        full_name,
                        loaded.ref_count,
                    )
                    del self._models[full_name]
                    self._orphaned_models[id(loaded)] = loaded
                    if loaded.ref_count == 0:
                        self._schedule_orphan_teardown_locked(
                            full_name,
                            loaded,
                            "dead adapter retirement",
                        )
                    loaded = None
                if loaded is not None:
                    loaded.ref_count += 1
                    loaded.last_used = time.time()
                    loaded.trimmed = False
                    return loaded
            if maintenance_done is not None:
                await maintenance_done.wait()
                continue
            await self._ensure_loaded(full_name)

    @asynccontextmanager
    async def acquire(self, model_name: str):
        """Acquire a loaded model adapter. Loads on first use, ref-counted."""
        full_name = self._normalize_model_ref(model_name)

        loaded = await self._acquire_loaded(full_name)

        try:
            yield loaded.adapter
        finally:
            async with self._lock:
                loaded.ref_count -= 1
                if self._models.get(full_name) is loaded:
                    loaded.last_used = time.time()
                elif self._orphaned_models.get(id(loaded)) is loaded and loaded.ref_count == 0:
                    self._schedule_orphan_teardown_locked(
                        full_name,
                        loaded,
                        "orphan release",
                    )

    async def preload(self, model_name: str) -> None:
        """Pre-load a model into memory."""
        full_name = self._normalize_model_ref(model_name)
        await self._ensure_loaded(full_name)

    async def _teardown_off_loop(
        self,
        items: list[tuple[str, _LoadedModel, str]],
    ) -> dict[str, str | None]:
        if not items:
            return {}
        return cast(
            dict[str, str | None],
            await asyncio.to_thread(
                run_with_adapter_runtime_lock,
                _teardown_adapters_blocking,
                [(name, loaded.adapter, reason) for name, loaded, reason in items],
            ),
        )

    async def unload(self, model_name: str) -> bool:
        """Unload a specific model. Returns True if unloaded, False if skipped."""
        full_name = self._normalize_model_ref(model_name)
        async with self._lifecycle_lock:
            async with self._lock:
                loaded = self._models.get(full_name)
                if loaded is None:
                    return True
                if loaded.is_busy:
                    logger.warning(
                        "Cannot unload %s: %d active references or physical jobs",
                        full_name,
                        loaded.active_count,
                    )
                    return False
                self._begin_maintenance(loaded, "explicit unload")
            task = self._start_maintenance_task(self._complete_unloads([(full_name, loaded, "explicit unload")]))
            results = await self._await_owned_task(task)
            error = results.get(full_name)
            return error is None

    async def trim(self, model_name: str) -> bool:
        """Trim a loaded model without unloading weights. Returns False when active."""
        full_name = self._normalize_model_ref(model_name)
        async with self._lifecycle_lock:
            async with self._lock:
                loaded = self._models.get(full_name)
                if loaded is None:
                    return True
                if not loaded.adapter.supports_trim:
                    raise ModelTrimUnsupportedError(full_name)
                if loaded.is_busy:
                    logger.warning(
                        "Cannot trim %s: %d active references or physical jobs",
                        full_name,
                        loaded.active_count,
                    )
                    return False
                self._begin_maintenance(loaded, "trim")
            task = self._start_maintenance_task(self._complete_trims([(full_name, loaded)]))
            trimmed = await self._await_owned_task(task)
            return full_name in trimmed

    async def trim_idle(self, *, min_idle_seconds: int = 0) -> list[str]:
        """Trim all idle models and return the canonical model refs that were trimmed."""
        async with self._lifecycle_lock:
            async with self._lock:
                eligible = self._select_trimmable_locked(min_idle_seconds=min_idle_seconds)
                for _full_name, loaded in eligible:
                    self._begin_maintenance(loaded, "idle trim")
            task = self._start_maintenance_task(self._complete_trims(eligible))
            trimmed = await self._await_owned_task(task)
            return trimmed

    async def unload_all(self, *, deadline: float | None = None) -> None:
        """Unload all idle models. Models with active references are left in place."""
        async with self._lifecycle_lock:
            async with self._lock:
                removable = [(name, m) for name, m in self._models.items() if not m.is_busy]
                busy = [name for name, m in self._models.items() if m.is_busy]
                for _name, loaded in removable:
                    self._begin_maintenance(loaded, "unload_all")
            if busy:
                logger.warning(
                    "unload_all skipped %d model(s) with active references: %s",
                    len(busy),
                    ", ".join(busy),
                )
            task = self._start_maintenance_task(
                self._complete_unloads([(name, loaded, "unload_all") for name, loaded in removable])
            )
            if deadline is None:
                await self._await_owned_task(task)
            else:
                await self._wait_for_shutdown_tasks((task,), deadline, "model unload")

    async def _unload_orphans(self, *, deadline: float | None = None) -> None:
        async with self._lifecycle_lock:
            async with self._lock:
                removable = [loaded for loaded in self._orphaned_models.values() if not loaded.is_busy]
                tasks: list[asyncio.Task[Any]] = []
                for loaded in removable:
                    task = self._schedule_orphan_teardown_locked(
                        loaded.full_name,
                        loaded,
                        "orphan shutdown",
                    )
                    if task is not None:
                        tasks.append(task)
            if deadline is None:
                if tasks:
                    await asyncio.gather(*(self._await_owned_task(task) for task in tasks))
            else:
                await self._wait_for_shutdown_tasks(tuple(tasks), deadline, "orphan unload")

    def list_loaded(self) -> list[LoadedModelInfo]:
        """List currently loaded models."""
        return [
            LoadedModelInfo(
                name=m.info.name,
                tag=m.info.tag,
                type=m.info.type,
                device=m.device,
                vram_bytes=m.vram_bytes,
                loaded_at=m.loaded_at,
                last_used=m.last_used,
                ref_count=m.ref_count,
                is_evictable=not m.is_busy,
                is_trimmable=not m.is_busy and m.adapter.supports_trim,
                backend_memory=self._adapter_memory_status(m),
            )
            for m in self._models.values()
        ]

    def memory_snapshot(self) -> VramSnapshot:
        loaded = tuple(self.list_loaded())
        process = process_memory_status()
        cgroup = cgroup_memory_status()
        return VramSnapshot(
            device=_device_memory_snapshot("cuda"),
            process=ProcessMemoryInfo(
                rss_bytes=process["rss_bytes"],
                peak_rss_bytes=process["peak_rss_bytes"],
                cgroup_current_bytes=cgroup["current_bytes"],
                cgroup_peak_bytes=cgroup["peak_bytes"],
                cgroup_limit_bytes=cgroup["limit_bytes"],
            ),
            idle_trim_seconds=self._idle_trim_seconds,
            loaded_models=loaded,
            estimated_loaded_vram_bytes=sum(max(int(m.vram_bytes), 0) for m in loaded if m.device == "cuda"),
            active_model_count=sum(1 for m in self._models.values() if m.is_busy),
        )

    async def _ttl_cleanup_loop(self) -> None:
        """Periodically unload idle models past TTL."""
        while True:
            await asyncio.sleep(self._cleanup_interval)
            now = time.time()
            to_unload: list[tuple[str, _LoadedModel, str]] = []
            async with self._lock:
                if self._ttl_seconds > 0:
                    to_evict = [
                        name
                        for name, m in self._models.items()
                        if not m.is_busy and (now - m.last_used) > self._ttl_seconds
                    ]
                    for name in to_evict:
                        logger.info(f"TTL expired for {name}, unloading")
                        to_unload.append((name, self._models[name], "TTL cleanup"))
            for name, _loaded, _reason in to_unload:
                await self.unload(name)
            if self._idle_trim_seconds > 0:
                await self.trim_idle(min_idle_seconds=self._idle_trim_seconds)
