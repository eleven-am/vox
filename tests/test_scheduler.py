"""Comprehensive tests for vox.core.scheduler.Scheduler."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.errors import ModelLoadError
from vox.core.runtime import RuntimeCapabilities
from vox.core.scheduler import Scheduler, _detect_device, _is_oom_error, _LoadedModel
from vox.core.types import (
    AdapterInfo,
    LoadedModelInfo,
    ModelFormat,
    ModelInfo,
    ModelType,
    SynthesizeChunk,
    TranscribeResult,
    parse_model_name,
)


class FakeSTTAdapter(STTAdapter):
    """Minimal STT adapter for testing."""

    def __init__(self, *, oom_on_device: str | None = None, fail_with: Exception | None = None):
        self._loaded = False
        self._device: str | None = None
        self._oom_on_device = oom_on_device
        self._fail_with = fail_with
        self.load_calls: list[tuple[str, str]] = []
        self.unload_calls: int = 0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt",
            type=ModelType.STT,
            architectures=("fake",),
            default_sample_rate=16000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        self.load_calls.append((model_path, device))
        if self._oom_on_device and device == self._oom_on_device:
            raise RuntimeError("CUDA out of memory")
        if self._fail_with is not None:
            raise self._fail_with
        self._loaded = True
        self._device = device

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> TranscribeResult:
        return TranscribeResult(text="hello")


class FakeTTSAdapter(TTSAdapter):
    """Minimal TTS adapter for testing."""

    def __init__(self):
        self._loaded = False

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts",
            type=ModelType.TTS,
            architectures=("fake",),
            default_sample_rate=24000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: np.ndarray | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        yield SynthesizeChunk(audio=b"\x00" * 4, sample_rate=24000, is_final=True)






class FakeRegistry:
    """Implements RegistryProtocol for testing."""

    def __init__(self):
        self._models: dict[str, tuple[ModelInfo, Path]] = {}
        self._adapter_classes: dict[str, type] = {}
        self._aliases: dict[str, tuple[str, str]] = {}

    def add_model(
        self,
        name: str,
        tag: str,
        *,
        adapter_name: str = "fake-stt",
        adapter_cls: type | None = None,
        model_type: ModelType = ModelType.STT,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        info = ModelInfo(
            name=name,
            tag=tag,
            type=model_type,
            format=ModelFormat.ONNX,
            architecture="fake",
            adapter=adapter_name,
            parameters=parameters or {},
        )
        self._models[f"{name}:{tag}"] = (info, Path(f"/fake/models/{name}/{tag}"))
        if adapter_cls is not None:
            self._adapter_classes[adapter_name] = adapter_cls

    def add_alias(self, alias: str, resolved_name: str, resolved_tag: str) -> None:
        self._aliases[alias] = (resolved_name, resolved_tag)

    def resolve(self, name: str, tag: str) -> tuple[ModelInfo, Path]:
        key = f"{name}:{tag}"
        if key not in self._models:
            raise KeyError(f"Model {key} not registered in fake registry")
        return self._models[key]

    def resolve_model_ref(
        self, name: str, tag: str = "latest", *, explicit_tag: bool = False
    ) -> tuple[str, str]:
        if not explicit_tag and name in self._aliases:
            return self._aliases[name]
        return name, tag

    def get_adapter_class(self, adapter_name: str) -> type:
        if adapter_name not in self._adapter_classes:
            raise KeyError(f"Adapter {adapter_name} not registered in fake registry")
        return self._adapter_classes[adapter_name]






def _make_registry_with_model(
    name: str = "whisper",
    tag: str = "large-v3",
    adapter_cls: type | None = None,
    parameters: dict[str, Any] | None = None,
) -> FakeRegistry:
    """Create a FakeRegistry pre-populated with one model."""
    registry = FakeRegistry()
    registry.add_model(name, tag, adapter_cls=adapter_cls or FakeSTTAdapter, parameters=parameters)
    return registry


@pytest.fixture
def registry() -> FakeRegistry:
    return _make_registry_with_model()


@pytest.fixture
def scheduler(registry: FakeRegistry) -> Scheduler:
    return Scheduler(registry, default_device="cpu", max_loaded=3)



@pytest.fixture(autouse=True)
def _patch_gpu_cache():
    with patch("vox.core.scheduler._clear_gpu_cache"):
        yield







@pytest.mark.asyncio
async def test_acquire_loads_model_on_first_use(scheduler: Scheduler):
    """First acquire should trigger a load; adapter must be usable inside the context."""
    async with scheduler.acquire("whisper:large-v3") as adapter:
        assert isinstance(adapter, FakeSTTAdapter)
        assert adapter.is_loaded
        assert len(adapter.load_calls) == 1


@pytest.mark.asyncio
async def test_acquire_reuses_already_loaded_model(scheduler: Scheduler):
    """Second acquire for the same model must not reload it."""
    async with scheduler.acquire("whisper:large-v3") as adapter_first:
        first_id = id(adapter_first)

    async with scheduler.acquire("whisper:large-v3") as adapter_second:
        second_id = id(adapter_second)

    assert first_id == second_id

    assert len(adapter_second.load_calls) == 1


@pytest.mark.asyncio
async def test_acquire_increments_and_decrements_ref_count(scheduler: Scheduler):
    """ref_count should be 1 inside the context and 0 after exiting."""
    async with scheduler.acquire("whisper:large-v3"):
        loaded_list = scheduler.list_loaded()
        assert len(loaded_list) == 1
        assert loaded_list[0].ref_count == 1

    loaded_list = scheduler.list_loaded()
    assert loaded_list[0].ref_count == 0


@pytest.mark.asyncio
async def test_acquire_ref_count_decremented_on_exception(scheduler: Scheduler):
    """ref_count must be decremented even when the caller raises inside the context."""
    with pytest.raises(ValueError, match="boom"):
        async with scheduler.acquire("whisper:large-v3"):
            raise ValueError("boom")

    loaded_list = scheduler.list_loaded()
    assert len(loaded_list) == 1
    assert loaded_list[0].ref_count == 0


@pytest.mark.asyncio
async def test_evict_lru_removes_least_recently_used():
    """When at capacity, the least-recently-used model with ref_count 0 should be evicted."""
    registry = FakeRegistry()
    registry.add_model("m1", "latest", adapter_cls=FakeSTTAdapter)
    registry.add_model("m2", "latest", adapter_cls=FakeSTTAdapter)
    registry.add_model("m3", "latest", adapter_cls=FakeSTTAdapter)

    sched = Scheduler(registry, default_device="cpu", max_loaded=2)


    async with sched.acquire("m1:latest"):
        pass
    async with sched.acquire("m2:latest"):
        pass


    async with sched.acquire("m1:latest"):
        pass


    async with sched.acquire("m3:latest"):
        loaded_names = {f"{m.name}:{m.tag}" for m in sched.list_loaded()}
        assert "m1:latest" in loaded_names
        assert "m3:latest" in loaded_names
        assert "m2:latest" not in loaded_names


@pytest.mark.asyncio
async def test_evict_lru_skips_models_with_active_refs():
    """Models with active refs must not be evicted."""
    registry = FakeRegistry()
    registry.add_model("m1", "latest", adapter_cls=FakeSTTAdapter)
    registry.add_model("m2", "latest", adapter_cls=FakeSTTAdapter)
    registry.add_model("m3", "latest", adapter_cls=FakeSTTAdapter)

    sched = Scheduler(registry, default_device="cpu", max_loaded=2)

    async with sched.acquire("m1:latest"):
        pass
    async with sched.acquire("m2:latest"):
        pass


    async with sched.acquire("m1:latest"), sched.acquire("m3:latest"):
        loaded_names = {f"{m.name}:{m.tag}" for m in sched.list_loaded()}
        assert "m1:latest" in loaded_names
        assert "m3:latest" in loaded_names
        assert "m2:latest" not in loaded_names


@pytest.mark.asyncio
async def test_max_loaded_raises_when_all_in_use():
    """When all slots are occupied by active refs, loading another model must raise."""
    registry = FakeRegistry()
    registry.add_model("m1", "latest", adapter_cls=FakeSTTAdapter)
    registry.add_model("m2", "latest", adapter_cls=FakeSTTAdapter)
    registry.add_model("m3", "latest", adapter_cls=FakeSTTAdapter)

    sched = Scheduler(registry, default_device="cpu", max_loaded=2)

    async with sched.acquire("m1:latest"), sched.acquire("m2:latest"):
        with pytest.raises(ModelLoadError, match="all 2 model slots are in use"):
            async with sched.acquire("m3:latest"):
                pass


@pytest.mark.asyncio
async def test_load_falls_back_to_cpu_on_oom():
    """If load raises OOM on GPU, scheduler should retry on CPU."""

    class OOMAdapter(FakeSTTAdapter):
        def __init__(self):
            super().__init__(oom_on_device="cuda")

    registry = _make_registry_with_model(adapter_cls=OOMAdapter)
    sched = Scheduler(registry, default_device="cuda", max_loaded=3)

    async with sched.acquire("whisper:large-v3") as adapter:
        assert isinstance(adapter, OOMAdapter)

        assert adapter.load_calls == [
            ("/fake/models/whisper/large-v3", "cuda"),
            ("/fake/models/whisper/large-v3", "cpu"),
        ]

    loaded = sched.list_loaded()
    assert loaded[0].device == "cpu"
    assert loaded[0].vram_bytes == 0


def _make_mock_torch(*, free_bytes: int, total_bytes: int = 16_000_000_000):
    class _MockCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return free_bytes, total_bytes

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def synchronize() -> None:
            return None

    class _MockBackends:
        pass

    class _MockTorch:
        cuda = _MockCuda()
        backends = _MockBackends()

    return _MockTorch()


@pytest.mark.asyncio
async def test_select_device_uses_estimate_and_routes_to_cpu_when_cuda_is_tight():
    class EstimatedAdapter(FakeSTTAdapter):
        estimate_kwargs: dict[str, Any] | None = None

        def estimate_vram_bytes(self, **kwargs: Any) -> int:
            type(self).estimate_kwargs = kwargs
            return 4_000_000_000

    registry = _make_registry_with_model(
        adapter_cls=EstimatedAdapter,
        parameters={"_source": "Qwen/Qwen3-ASR-1.7B"},
    )
    sched = Scheduler(registry, default_device="auto", max_loaded=3)
    mock_torch = _make_mock_torch(free_bytes=3_500_000_000)
    capabilities = RuntimeCapabilities(
        system="linux",
        machine="x86_64",
        torch_cuda=True,
        onnx_cuda=True,
        onnx_coreml=False,
        mps=False,
        nvidia_device=True,
    )

    with (
        patch.dict("sys.modules", {"torch": mock_torch}),
        patch("vox.core.scheduler.detect_runtime_capabilities", return_value=capabilities),
    ):
        async with sched.acquire("whisper:large-v3") as adapter:
            assert isinstance(adapter, EstimatedAdapter)
            assert adapter.load_calls == [
                ("/fake/models/whisper/large-v3", "cpu"),
            ]

    loaded = sched.list_loaded()
    assert loaded[0].device == "cpu"
    assert loaded[0].vram_bytes == 0
    assert EstimatedAdapter.estimate_kwargs == {
        "_source": "Qwen/Qwen3-ASR-1.7B",
        "model_path": "/fake/models/whisper/large-v3",
    }


@pytest.mark.asyncio
async def test_select_device_evicts_idle_cuda_model_before_falling_back_to_cpu():
    class ExistingAdapter(FakeSTTAdapter):
        def estimate_vram_bytes(self, **kwargs: Any) -> int:
            return 2_000_000_000

    class EstimatedAdapter(FakeSTTAdapter):
        def estimate_vram_bytes(self, **kwargs: Any) -> int:
            return 4_000_000_000

    registry = FakeRegistry()
    registry.add_model("existing", "latest", adapter_cls=ExistingAdapter)
    registry.add_model("candidate", "latest", adapter_cls=EstimatedAdapter)
    sched = Scheduler(registry, default_device="auto", max_loaded=3)
    mock_torch = _make_mock_torch(free_bytes=3_500_000_000)
    capabilities = RuntimeCapabilities(
        system="linux",
        machine="x86_64",
        torch_cuda=True,
        onnx_cuda=True,
        onnx_coreml=False,
        mps=False,
        nvidia_device=True,
    )

    existing_info, _ = registry.resolve("existing", "latest")
    existing_adapter = ExistingAdapter()
    existing_adapter._loaded = True
    existing_adapter._device = "cuda"
    sched._models["existing:latest"] = _LoadedModel(
        full_name="existing:latest",
        info=existing_info,
        adapter=existing_adapter,
        device="cuda",
        loaded_at=time.time() - 120,
        last_used=time.time() - 120,
    )

    free_bytes = iter([3_500_000_000, 3_500_000_000, 5_500_000_000])

    with (
        patch.dict("sys.modules", {"torch": mock_torch}),
        patch("vox.core.scheduler.detect_runtime_capabilities", return_value=capabilities),
        patch("vox.core.scheduler._available_device_memory_bytes", side_effect=lambda device: next(free_bytes)),
    ):
        async with sched.acquire("candidate:latest") as adapter:
            assert isinstance(adapter, EstimatedAdapter)
            assert adapter.load_calls == [
                ("/fake/models/candidate/latest", "auto"),
            ]

    assert existing_adapter.unload_calls == 1


@pytest.mark.asyncio
async def test_scheduler_passes_source_hint_to_adapter_load():
    class SourceAwareAdapter(FakeSTTAdapter):
        load_kwargs: dict[str, Any] | None = None

        def load(self, model_path: str, device: str, **kwargs: Any) -> None:
            type(self).load_kwargs = kwargs
            super().load(model_path, device, **kwargs)

    registry = _make_registry_with_model(
        adapter_cls=SourceAwareAdapter,
        parameters={
            "sample_rate": 16000,
            "_source": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        },
    )
    sched = Scheduler(registry, default_device="cpu", max_loaded=3)

    async with sched.acquire("whisper:large-v3") as adapter:
        assert isinstance(adapter, SourceAwareAdapter)

    assert SourceAwareAdapter.load_kwargs == {
        "sample_rate": 16000,
        "_source": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    }


@pytest.mark.asyncio
async def test_select_device_keeps_cuda_when_estimate_fits():
    class EstimatedAdapter(FakeSTTAdapter):
        def estimate_vram_bytes(self, **kwargs: Any) -> int:
            return 2_000_000_000

    registry = _make_registry_with_model(adapter_cls=EstimatedAdapter)
    sched = Scheduler(registry, default_device="auto", max_loaded=3)
    mock_torch = _make_mock_torch(free_bytes=4_500_000_000)
    capabilities = RuntimeCapabilities(
        system="linux",
        machine="x86_64",
        torch_cuda=True,
        onnx_cuda=True,
        onnx_coreml=False,
        mps=False,
        nvidia_device=True,
    )

    with (
        patch.dict("sys.modules", {"torch": mock_torch}),
        patch("vox.core.scheduler.detect_runtime_capabilities", return_value=capabilities),
    ):
        async with sched.acquire("whisper:large-v3") as adapter:
            assert isinstance(adapter, EstimatedAdapter)
            assert adapter.load_calls == [
                ("/fake/models/whisper/large-v3", "auto"),
            ]

    loaded = sched.list_loaded()
    assert loaded[0].device == "cuda"
    assert loaded[0].vram_bytes == 2_000_000_000


@pytest.mark.asyncio
async def test_select_device_respects_explicit_cuda_even_when_estimate_exceeds_free_memory():
    class EstimatedAdapter(FakeSTTAdapter):
        def estimate_vram_bytes(self, **kwargs: Any) -> int:
            return 16_000_000_000

    registry = _make_registry_with_model(adapter_cls=EstimatedAdapter)
    sched = Scheduler(registry, default_device="cuda", max_loaded=3)
    mock_torch = _make_mock_torch(free_bytes=3_500_000_000, total_bytes=130_000_000_000)

    with patch.dict("sys.modules", {"torch": mock_torch}):
        async with sched.acquire("whisper:large-v3") as adapter:
            assert isinstance(adapter, EstimatedAdapter)
            assert adapter.load_calls == [
                ("/fake/models/whisper/large-v3", "cuda"),
            ]

    loaded = sched.list_loaded()
    assert loaded[0].device == "cuda"
    assert loaded[0].vram_bytes == 16_000_000_000


@pytest.mark.asyncio
async def test_load_non_oom_error_raises_model_load_error():
    """A non-OOM load failure must propagate as ModelLoadError."""

    class BadAdapter(FakeSTTAdapter):
        def __init__(self):
            super().__init__(fail_with=RuntimeError("corrupted weights"))

    registry = _make_registry_with_model(adapter_cls=BadAdapter)
    sched = Scheduler(registry, default_device="cpu", max_loaded=3)

    with pytest.raises(ModelLoadError, match="corrupted weights"):
        async with sched.acquire("whisper:large-v3"):
            pass


@pytest.mark.asyncio
async def test_unload_returns_false_when_refs_active(scheduler: Scheduler):
    """unload should return False when the model has active references."""
    async with scheduler.acquire("whisper:large-v3"):
        result = await scheduler.unload("whisper:large-v3")
        assert result is False


    result = await scheduler.unload("whisper:large-v3")
    assert result is True


@pytest.mark.asyncio
async def test_unload_returns_true_and_removes_model(scheduler: Scheduler):
    """unload should return True, remove the model, and call adapter.unload()."""
    async with scheduler.acquire("whisper:large-v3") as adapter:
        pass

    assert len(scheduler.list_loaded()) == 1
    result = await scheduler.unload("whisper:large-v3")
    assert result is True
    assert len(scheduler.list_loaded()) == 0
    assert adapter.unload_calls == 1


@pytest.mark.asyncio
async def test_unload_nonexistent_model_returns_true(scheduler: Scheduler):
    """Unloading a model that was never loaded should return True (no-op)."""
    result = await scheduler.unload("nonexistent:latest")
    assert result is True


@pytest.mark.asyncio
async def test_unload_all(scheduler: Scheduler, registry: FakeRegistry):
    """unload_all should unload every loaded model."""
    registry.add_model("m2", "latest", adapter_cls=FakeSTTAdapter)

    async with scheduler.acquire("whisper:large-v3"):
        pass
    async with scheduler.acquire("m2:latest"):
        pass

    assert len(scheduler.list_loaded()) == 2
    await scheduler.unload_all()
    assert len(scheduler.list_loaded()) == 0


@pytest.mark.asyncio
async def test_preload(scheduler: Scheduler):
    """preload should load the model without bumping the ref_count."""
    await scheduler.preload("whisper:large-v3")
    loaded = scheduler.list_loaded()
    assert len(loaded) == 1
    assert loaded[0].name == "whisper"
    assert loaded[0].tag == "large-v3"
    assert loaded[0].ref_count == 0


@pytest.mark.asyncio
async def test_alias_acquire_reuses_preloaded_canonical_model():
    registry = FakeRegistry()
    registry.add_model("canonical-model", "v1", adapter_cls=FakeSTTAdapter)
    registry.add_alias("friendly-model", "canonical-model", "v1")
    sched = Scheduler(registry, default_device="cpu", max_loaded=3)

    await sched.preload("friendly-model")

    assert set(sched._models) == {"canonical-model:v1"}
    preloaded = sched._models["canonical-model:v1"].adapter
    assert isinstance(preloaded, FakeSTTAdapter)
    assert len(preloaded.load_calls) == 1

    async with sched.acquire("friendly-model") as adapter:
        assert adapter is preloaded

    assert len(preloaded.load_calls) == 1


@pytest.mark.asyncio
async def test_list_loaded(scheduler: Scheduler, registry: FakeRegistry):
    """list_loaded should return info for every loaded model."""
    registry.add_model("tts-model", "v1", adapter_cls=FakeTTSAdapter, model_type=ModelType.TTS)

    await scheduler.preload("whisper:large-v3")
    await scheduler.preload("tts-model:v1")

    loaded = scheduler.list_loaded()
    assert len(loaded) == 2
    names = {m.name for m in loaded}
    assert names == {"whisper", "tts-model"}

    for info in loaded:
        assert isinstance(info, LoadedModelInfo)
        assert info.device == "cpu"
        assert info.ref_count == 0






def test_parse_model_name_with_tag():
    name, tag = parse_model_name("whisper:large-v3")
    assert name == "whisper"
    assert tag == "large-v3"


def test_parse_model_name_without_tag():
    name, tag = parse_model_name("whisper")
    assert name == "whisper"
    assert tag == "latest"







def test_detect_device_returns_cpu_when_no_torch():
    """When torch is not importable, _detect_device should return 'cpu'."""
    capabilities = RuntimeCapabilities(
        system="linux",
        machine="x86_64",
        torch_cuda=False,
        onnx_cuda=False,
        onnx_coreml=False,
        mps=False,
        nvidia_device=False,
    )
    with patch("vox.core.scheduler.detect_runtime_capabilities", return_value=capabilities):
        assert _detect_device() == "cpu"


def test_detect_device_returns_cuda_when_available():
    """When torch.cuda.is_available() returns True, device should be 'cuda'."""
    capabilities = RuntimeCapabilities(
        system="linux",
        machine="x86_64",
        torch_cuda=True,
        onnx_cuda=False,
        onnx_coreml=False,
        mps=False,
        nvidia_device=True,
    )
    with patch("vox.core.scheduler.detect_runtime_capabilities", return_value=capabilities):
        assert _detect_device() == "cuda"







def test_is_oom_error_matches_keywords():
    """_is_oom_error should return True for known OOM messages."""
    assert _is_oom_error(RuntimeError("CUDA out of memory")) is True
    assert _is_oom_error(RuntimeError("failed to allocate 2 GiB")) is True
    assert _is_oom_error(RuntimeError("onnxruntime OOM during inference")) is True


def test_is_oom_error_no_match():
    """_is_oom_error should return False for unrelated errors."""
    assert _is_oom_error(RuntimeError("file not found")) is False
    assert _is_oom_error(ValueError("invalid shape")) is False







@pytest.mark.asyncio
async def test_start_and_stop_lifecycle(scheduler: Scheduler):
    """start() should create a cleanup task; stop() should cancel it and unload models."""
    await scheduler.preload("whisper:large-v3")
    assert len(scheduler.list_loaded()) == 1

    await scheduler.start()
    assert scheduler._cleanup_task is not None
    assert not scheduler._cleanup_task.done()

    await scheduler.stop()
    assert scheduler._cleanup_task is None
    assert len(scheduler.list_loaded()) == 0







@pytest.mark.asyncio
async def test_ttl_cleanup_evicts_idle_models():
    """Models idle beyond TTL should be evicted by the cleanup loop."""
    registry = _make_registry_with_model()
    sched = Scheduler(
        registry,
        default_device="cpu",
        max_loaded=3,
        ttl_seconds=0,
        cleanup_interval=0,
    )

    await sched.preload("whisper:large-v3")
    assert len(sched.list_loaded()) == 1


    for m in sched._models.values():
        m.last_used = time.time() - 10

    await sched.start()

    await asyncio.sleep(0.05)
    await sched.stop()

    assert len(sched.list_loaded()) == 0







@pytest.mark.asyncio
async def test_unload_handles_adapter_unload_exception():
    """If adapter.unload() raises, Scheduler.unload() should log and not crash."""

    class ExplodingUnloadAdapter(FakeSTTAdapter):
        def unload(self) -> None:
            raise RuntimeError("boom during unload")

    registry = _make_registry_with_model(adapter_cls=ExplodingUnloadAdapter)
    sched = Scheduler(registry, default_device="cpu", max_loaded=3)

    await sched.preload("whisper:large-v3")
    assert len(sched.list_loaded()) == 1


    result = await sched.unload("whisper:large-v3")
    assert result is True
    assert len(sched.list_loaded()) == 0







@pytest.mark.asyncio
async def test_cpu_fallback_also_fails():
    """If OOM triggers CPU fallback and that also fails, raise ModelLoadError."""

    class AlwaysFailAdapter(FakeSTTAdapter):
        def __init__(self):
            super().__init__()

        def load(self, model_path: str, device: str, **kwargs) -> None:
            self.load_calls.append((model_path, device))
            if device != "cpu":
                raise RuntimeError("CUDA out of memory")
            raise RuntimeError("CPU also broken")

    registry = _make_registry_with_model(adapter_cls=AlwaysFailAdapter)
    sched = Scheduler(registry, default_device="cuda", max_loaded=3)

    with pytest.raises(ModelLoadError, match="CPU also broken"):
        async with sched.acquire("whisper:large-v3"):
            pass
