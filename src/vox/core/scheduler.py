from __future__ import annotations

import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.errors import ModelLoadError
from vox.core.runtime import detect_runtime_capabilities
from vox.core.types import LoadedModelInfo, ModelFormat, ModelInfo, parse_model_name

logger = logging.getLogger(__name__)

Adapter = STTAdapter | TTSAdapter
DEVICE_MEMORY_HEADROOM_BYTES = 512 * 1024 * 1024


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


class RegistryProtocol(Protocol):
    def resolve(self, name: str, tag: str) -> tuple[ModelInfo, Path]: ...
    def resolve_model_ref(
        self, name: str, tag: str = "latest", *, explicit_tag: bool = False
    ) -> tuple[str, str]: ...
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


class Scheduler:
    def __init__(
        self,
        registry: RegistryProtocol,
        *,
        default_device: str = "auto",
        max_loaded: int = 3,
        ttl_seconds: int = 300,
        cleanup_interval: int = 30,
    ) -> None:
        self._registry = registry
        self._requested_device = default_device
        self._default_device = default_device
        self._max_loaded = max_loaded
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._models: dict[str, _LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def _normalize_model_ref(self, model_name: str) -> str:
        """Resolve aliases so all cache keys use the canonical registry ref."""
        explicit_tag = ":" in model_name
        name, tag = parse_model_name(model_name)
        resolved_name, resolved_tag = self._registry.resolve_model_ref(
            name, tag, explicit_tag=explicit_tag
        )
        return f"{resolved_name}:{resolved_tag}"

    def _auto_device_for_model(self, info: ModelInfo) -> str:
        """Choose the best candidate device for *info* in auto mode."""
        capabilities = detect_runtime_capabilities()

        if info.format == ModelFormat.ONNX:
            if capabilities.onnx_cuda:
                return "cuda"
            return "auto" if capabilities.onnx_coreml else "cpu"

        if info.format in {ModelFormat.PYTORCH, ModelFormat.CT2}:
            if capabilities.torch_cuda:
                return "cuda"
            if capabilities.mps:
                return "mps"
            return "cpu"

        return "cpu"

    def _infer_loaded_device(self, adapter: Adapter, info: ModelInfo, requested_device: str) -> str:
        """Report the actual device used by an adapter after load."""
        actual_device = getattr(adapter, "_device", None)
        if isinstance(actual_device, str) and actual_device and actual_device != "auto":
            return actual_device

        if requested_device == "auto":
            candidate = self._auto_device_for_model(info)
            return candidate if candidate != "auto" else "cpu"

        return requested_device

    async def start(self) -> None:
        """Start the background TTL cleanup loop."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._ttl_cleanup_loop())

    async def stop(self) -> None:
        """Stop cleanup and unload all models."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None
        await self.unload_all()

    def _estimate_model_memory_bytes(self, adapter: Adapter, info: ModelInfo, model_path: Path) -> int:
        """Estimate accelerator memory required for *info* before loading."""
        estimate_kwargs = {**info.parameters, "model_path": str(model_path)}
        try:
            estimate = adapter.estimate_vram_bytes(**estimate_kwargs)
        except TypeError:
            estimate = adapter.estimate_vram_bytes()
        return max(int(estimate or 0), 0)

    def _evict_idle_models_for_memory(self, device: str, required_bytes: int) -> bool:
        """Evict idle models already occupying *device* until *required_bytes* is available."""
        if device != "cuda":
            return False

        while True:
            free_bytes = _available_device_memory_bytes(device)
            if free_bytes is None:
                return False
            if required_bytes <= free_bytes:
                return True

            candidates = [
                (name, model)
                for name, model in self._models.items()
                if model.ref_count == 0 and model.device == device
            ]
            if not candidates:
                return False

            candidates.sort(key=lambda item: item[1].last_used)
            lru_name, lru_model = candidates[0]
            logger.info(
                "Evicting %s to free %s memory for a new load (idle for %.0fs)",
                lru_name,
                device,
                time.time() - lru_model.last_used,
            )
            try:
                lru_model.adapter.unload()
            except Exception as error:
                logger.error("Error unloading %s during memory eviction: %s", lru_name, error)
            del self._models[lru_name]
            _clear_gpu_cache()

    def _select_device(self, adapter: Adapter, info: ModelInfo, estimated_vram_bytes: int) -> str:
        """Select device for a model, considering free accelerator memory."""
        if self._requested_device != "auto":
            return self._default_device

        candidate_device = self._auto_device_for_model(info)
        if candidate_device != "cuda" or estimated_vram_bytes <= 0:
            return "auto"

        device = candidate_device
        if estimated_vram_bytes <= 0:
            return device

        free_bytes = _available_device_memory_bytes(device)
        if free_bytes is None:
            return "auto"

        required_bytes = estimated_vram_bytes + DEVICE_MEMORY_HEADROOM_BYTES
        if required_bytes > free_bytes and self._evict_idle_models_for_memory(device, required_bytes):
            return "auto"
        if required_bytes <= free_bytes:
            return "auto"

        logger.info(
            "Routing %s to CPU: estimated %d bytes plus headroom exceeds free %s memory (%d bytes)",
            info.full_name,
            estimated_vram_bytes,
            device,
            free_bytes,
        )
        return "cpu"

    async def _load_model(self, full_name: str) -> _LoadedModel:
        """Load a model by name. Handles eviction and OOM fallback."""
        full_name = self._normalize_model_ref(full_name)
        name, tag = parse_model_name(full_name)


        info, model_path = self._registry.resolve(name, tag)
        adapter_cls = self._registry.get_adapter_class(info.adapter)


        adapter = adapter_cls()
        estimated_vram_bytes = self._estimate_model_memory_bytes(adapter, info, model_path)
        device = self._select_device(adapter, info, estimated_vram_bytes)


        if len(self._models) >= self._max_loaded:
            self._evict_lru()
            if len(self._models) >= self._max_loaded:
                raise ModelLoadError(
                    f"Cannot load {full_name}: all {self._max_loaded} model slots are in use. "
                    "Wait for an active request to finish or increase --max-loaded."
                )


        load_kwargs = {**info.parameters}
        try:
            logger.info(f"Loading {full_name} on {device}")
            start = time.perf_counter()
            await asyncio.to_thread(adapter.load, str(model_path), device, **load_kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"Loaded {full_name} in {elapsed:.2f}s on {device}")
        except Exception as e:
            if _is_oom_error(e) and device != "cpu":
                logger.warning(f"OOM loading {full_name} on {device}, falling back to CPU")
                _clear_gpu_cache()
                device = "cpu"
                try:
                    await asyncio.to_thread(adapter.load, str(model_path), device, **load_kwargs)
                except Exception as e2:
                    raise ModelLoadError(f"Failed to load {full_name}: {e2}") from e2
            else:
                raise ModelLoadError(f"Failed to load {full_name}: {e}") from e

        actual_device = self._infer_loaded_device(adapter, info, device)
        loaded = _LoadedModel(
            full_name=full_name,
            info=info,
            adapter=adapter,
            device=actual_device,
            vram_bytes=estimated_vram_bytes if actual_device != "cpu" else 0,
        )
        self._models[full_name] = loaded
        return loaded

    def _evict_lru(self) -> None:
        """Evict the least-recently-used model with ref_count == 0."""
        candidates = [
            (name, m) for name, m in self._models.items() if m.ref_count == 0
        ]
        if not candidates:
            logger.warning("Cannot evict: all loaded models are in use")
            return

        candidates.sort(key=lambda x: x[1].last_used)
        lru_name, lru_model = candidates[0]
        logger.info(f"Evicting {lru_name} (idle since {time.time() - lru_model.last_used:.0f}s ago)")
        try:
            lru_model.adapter.unload()
        except Exception as e:
            logger.error(f"Error unloading {lru_name} during eviction: {e}")
        del self._models[lru_name]
        _clear_gpu_cache()

    @asynccontextmanager
    async def acquire(self, model_name: str):
        """Acquire a loaded model adapter. Loads on first use, ref-counted."""
        full_name = self._normalize_model_ref(model_name)

        async with self._lock:
            if full_name not in self._models:
                await self._load_model(full_name)
            loaded = self._models[full_name]
            loaded.ref_count += 1
            loaded.last_used = time.time()

        try:
            yield loaded.adapter
        finally:
            async with self._lock:
                if full_name in self._models:
                    self._models[full_name].ref_count -= 1

    async def preload(self, model_name: str) -> None:
        """Pre-load a model into memory."""
        full_name = self._normalize_model_ref(model_name)
        async with self._lock:
            if full_name not in self._models:
                await self._load_model(full_name)

    async def unload(self, model_name: str) -> bool:
        """Unload a specific model. Returns True if unloaded, False if skipped."""
        full_name = self._normalize_model_ref(model_name)
        async with self._lock:
            if full_name in self._models:
                loaded = self._models[full_name]
                if loaded.ref_count > 0:
                    logger.warning(f"Cannot unload {full_name}: {loaded.ref_count} active references")
                    return False
                try:
                    loaded.adapter.unload()
                except Exception as e:
                    logger.error(f"Error unloading {full_name}: {e}")
                del self._models[full_name]
                _clear_gpu_cache()
        return True

    async def unload_all(self) -> None:
        """Unload all models."""
        async with self._lock:
            for name, loaded in list(self._models.items()):
                try:
                    loaded.adapter.unload()
                except Exception as e:
                    logger.error(f"Error unloading {name}: {e}")
            self._models.clear()
            _clear_gpu_cache()

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
            )
            for m in self._models.values()
        ]

    async def _ttl_cleanup_loop(self) -> None:
        """Periodically unload idle models past TTL."""
        while True:
            await asyncio.sleep(self._cleanup_interval)
            if self._ttl_seconds <= 0:
                continue
            now = time.time()
            async with self._lock:
                to_evict = [
                    name for name, m in self._models.items()
                    if m.ref_count == 0 and (now - m.last_used) > self._ttl_seconds
                ]
                for name in to_evict:
                    logger.info(f"TTL expired for {name}, unloading")
                    try:
                        self._models[name].adapter.unload()
                    except Exception as e:
                        logger.error(f"Error unloading {name} during TTL cleanup: {e}")
                    del self._models[name]
                if to_evict:
                    _clear_gpu_cache()
