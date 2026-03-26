from __future__ import annotations

import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.errors import AdapterNotFoundError, ModelLoadError, ModelNotFoundError, OOMError
from vox.core.types import LoadedModelInfo, ModelInfo, ModelType

logger = logging.getLogger(__name__)

Adapter = STTAdapter | TTSAdapter


def _detect_device() -> str:
    """Auto-detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _clear_gpu_cache() -> None:
    """Clear CUDA/MPS cache and run garbage collection."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    gc.collect()


def _is_oom_error(error: Exception) -> bool:
    """Check if an exception is an out-of-memory error."""
    oom_keywords = ["out of memory", "oom", "cuda out of memory", "alloc", "memory"]
    msg = str(error).lower()
    return any(kw in msg for kw in oom_keywords)


class RegistryProtocol(Protocol):
    def resolve(self, name: str, tag: str) -> tuple[ModelInfo, Path]: ...
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
        self._default_device = default_device if default_device != "auto" else _detect_device()
        self._max_loaded = max_loaded
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._models: dict[str, _LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background TTL cleanup loop."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._ttl_cleanup_loop())

    async def stop(self) -> None:
        """Stop cleanup and unload all models."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        await self.unload_all()

    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        """Parse 'name:tag' into (name, tag). Default tag is 'latest'."""
        if ":" in model_name:
            name, tag = model_name.split(":", 1)
            return name, tag
        return model_name, "latest"

    def _select_device(self, adapter: Adapter) -> str:
        """Select device for a model, considering VRAM estimates."""
        return self._default_device

    async def _load_model(self, full_name: str) -> _LoadedModel:
        """Load a model by name. Handles eviction and OOM fallback."""
        name, tag = self._parse_model_name(full_name)

        # Resolve model info and path
        info, model_path = self._registry.resolve(name, tag)
        adapter_cls = self._registry.get_adapter_class(info.adapter)

        # Instantiate adapter
        adapter = adapter_cls()
        device = self._select_device(adapter)

        # Evict LRU if at capacity
        if len(self._models) >= self._max_loaded:
            self._evict_lru()

        # Try loading on preferred device, fallback to CPU on OOM
        try:
            logger.info(f"Loading {full_name} on {device}")
            start = time.perf_counter()
            await asyncio.to_thread(adapter.load, str(model_path), device, **info.parameters)
            elapsed = time.perf_counter() - start
            logger.info(f"Loaded {full_name} in {elapsed:.2f}s on {device}")
        except Exception as e:
            if _is_oom_error(e) and device != "cpu":
                logger.warning(f"OOM loading {full_name} on {device}, falling back to CPU")
                _clear_gpu_cache()
                device = "cpu"
                try:
                    await asyncio.to_thread(adapter.load, str(model_path), device, **info.parameters)
                except Exception as e2:
                    raise ModelLoadError(f"Failed to load {full_name}: {e2}") from e2
            else:
                raise ModelLoadError(f"Failed to load {full_name}: {e}") from e

        loaded = _LoadedModel(
            full_name=full_name,
            info=info,
            adapter=adapter,
            device=device,
            vram_bytes=adapter.estimate_vram_bytes(),
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
        # Sort by last_used ascending
        candidates.sort(key=lambda x: x[1].last_used)
        lru_name, lru_model = candidates[0]
        logger.info(f"Evicting {lru_name} (idle since {time.time() - lru_model.last_used:.0f}s ago)")
        lru_model.adapter.unload()
        del self._models[lru_name]
        _clear_gpu_cache()

    @asynccontextmanager
    async def acquire(self, model_name: str):
        """Acquire a loaded model adapter. Loads on first use, ref-counted."""
        name, tag = self._parse_model_name(model_name)
        full_name = f"{name}:{tag}"

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
        name, tag = self._parse_model_name(model_name)
        full_name = f"{name}:{tag}"
        async with self._lock:
            if full_name not in self._models:
                await self._load_model(full_name)

    async def unload(self, model_name: str) -> None:
        """Unload a specific model."""
        name, tag = self._parse_model_name(model_name)
        full_name = f"{name}:{tag}"
        async with self._lock:
            if full_name in self._models:
                loaded = self._models[full_name]
                if loaded.ref_count > 0:
                    logger.warning(f"Cannot unload {full_name}: {loaded.ref_count} active references")
                    return
                loaded.adapter.unload()
                del self._models[full_name]
                _clear_gpu_cache()

    async def unload_all(self) -> None:
        """Unload all models."""
        async with self._lock:
            for name, loaded in list(self._models.items()):
                loaded.adapter.unload()
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
                    self._models[name].adapter.unload()
                    del self._models[name]
                if to_evict:
                    _clear_gpu_cache()
