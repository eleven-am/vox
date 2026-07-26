from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from vox.core.adapter import TTSAdapter
from vox.core.adapter_execution import AdapterExecutionLane
from vox.core.errors import AdapterExecutionBusyError
from vox.core.scheduler import Scheduler
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk


class BlockingTTSAdapter(TTSAdapter):
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    state_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0
    started_texts: list[str] = []
    thread_ids: set[int] = set()

    @classmethod
    def reset(cls) -> None:
        cls.first_started = threading.Event()
        cls.first_release = threading.Event()
        cls.second_started = threading.Event()
        cls.state_lock = threading.Lock()
        cls.active_calls = 0
        cls.max_active_calls = 0
        cls.started_texts = []
        cls.thread_ids = set()

    def __init__(self) -> None:
        self._loaded = False
        self.trim_calls = 0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="blocking-tts",
            type=ModelType.TTS,
            architectures=("fake",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def trim(self) -> None:
        self.trim_calls += 1

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def synthesize(self, text: str, **kwargs: Any):
        with type(self).state_lock:
            type(self).active_calls += 1
            type(self).max_active_calls = max(
                type(self).max_active_calls,
                type(self).active_calls,
            )
            type(self).started_texts.append(text)
            type(self).thread_ids.add(threading.get_ident())
        try:
            if text == "first":
                type(self).first_started.set()
                type(self).first_release.wait(5.0)
            else:
                type(self).second_started.set()
            yield SynthesizeChunk(
                audio=text.encode(),
                sample_rate=24_000,
                is_final=False,
            )
        finally:
            with type(self).state_lock:
                type(self).active_calls -= 1


class FakeRegistry:
    def resolve_model_ref(
        self,
        name: str,
        tag: str = "latest",
        *,
        explicit_tag: bool = False,
    ) -> tuple[str, str]:
        return name, tag

    def resolve(self, name: str, tag: str):
        from pathlib import Path

        from vox.core.types import ModelInfo

        return (
            ModelInfo(
                name=name,
                tag=tag,
                type=ModelType.TTS,
                format=ModelFormat.ONNX,
                architecture="fake",
                adapter="blocking",
            ),
            Path("/tmp/blocking"),
        )

    def get_adapter_class(self, adapter_name: str) -> type:
        return BlockingTTSAdapter


async def _consume(adapter: BlockingTTSAdapter, text: str, output: list[bytes]) -> None:
    chunks = adapter.synthesize(text)
    async for chunk in adapter.iterate_synthesis(chunks):
        output.append(chunk.audio)


async def _wait_until_idle(adapter: BlockingTTSAdapter) -> None:
    for _ in range(100):
        if adapter.physical_work_count == 0:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("adapter physical work did not finish")


@pytest.mark.asyncio
async def test_cancelled_tts_remains_owned_and_serializes_replacement() -> None:
    BlockingTTSAdapter.reset()
    adapter = BlockingTTSAdapter()
    first_output: list[bytes] = []
    second_output: list[bytes] = []

    first = asyncio.create_task(_consume(adapter, "first", first_output))
    assert await asyncio.to_thread(BlockingTTSAdapter.first_started.wait, 1.0)

    started = time.perf_counter()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert time.perf_counter() - started < 0.1
    assert adapter.physical_work_count == 1

    second = asyncio.create_task(_consume(adapter, "second", second_output))
    await asyncio.sleep(0.05)

    assert BlockingTTSAdapter.second_started.is_set() is False
    assert BlockingTTSAdapter.max_active_calls == 1

    BlockingTTSAdapter.first_release.set()
    await second
    await _wait_until_idle(adapter)

    assert first_output == []
    assert second_output == [b"second"]
    assert BlockingTTSAdapter.max_active_calls == 1
    assert len(BlockingTTSAdapter.thread_ids) == 1
    adapter.close_execution_lane()


@pytest.mark.asyncio
async def test_cancelled_nested_thread_remains_physically_owned() -> None:
    adapter = BlockingTTSAdapter()
    physical_started = threading.Event()
    physical_release = threading.Event()
    physical_finished = threading.Event()

    async def nested_thread_synthesis():
        def physical_work() -> None:
            physical_started.set()
            physical_release.wait(5.0)
            physical_finished.set()

        await asyncio.to_thread(physical_work)
        yield SynthesizeChunk(audio=b"late", sample_rate=24_000, is_final=False)

    output: list[bytes] = []

    async def consume() -> None:
        async for chunk in adapter.iterate_synthesis(nested_thread_synthesis()):
            output.append(chunk.audio)

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(physical_started.wait, 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.05)
    assert physical_finished.is_set() is False
    assert adapter.physical_work_count == 1

    physical_release.set()
    await _wait_until_idle(adapter)

    assert physical_finished.is_set() is True
    assert output == []
    adapter.close_execution_lane()


@pytest.mark.asyncio
async def test_execution_lane_rejects_work_beyond_bounded_capacity() -> None:
    lane = AdapterExecutionLane(max_pending=2)
    first_started = threading.Event()
    release = threading.Event()

    def blocking() -> None:
        first_started.set()
        release.wait(5.0)

    first = asyncio.create_task(lane.run(blocking))
    assert await asyncio.to_thread(first_started.wait, 1.0)
    second = asyncio.create_task(lane.run(lambda: None))
    await asyncio.sleep(0)

    with pytest.raises(AdapterExecutionBusyError):
        await lane.run(lambda: None)

    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    release.set()
    await first
    await lane.wait_idle(timeout=1.0)
    lane.close()


@pytest.mark.asyncio
async def test_execution_lane_applies_output_backpressure() -> None:
    lane = AdapterExecutionLane(output_queue_size=1)
    third_yielded = threading.Event()
    third_completed = threading.Event()

    async def generate():
        yield b"one"
        yield b"two"
        third_yielded.set()
        yield b"three"
        third_completed.set()

    stream = lane.iterate(generate())
    assert await anext(stream) == b"one"
    assert await asyncio.to_thread(third_yielded.wait, 1.0)
    await asyncio.sleep(0.05)
    assert third_completed.is_set() is False

    assert await anext(stream) == b"two"
    assert await asyncio.to_thread(third_completed.wait, 1.0)
    assert await anext(stream) == b"three"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    await lane.wait_idle(timeout=1.0)
    lane.close()


@pytest.mark.asyncio
async def test_closed_adapter_execution_lane_cannot_be_resurrected() -> None:
    adapter = BlockingTTSAdapter()
    lane = adapter._get_execution_lane()
    adapter.close_execution_lane()

    with pytest.raises(RuntimeError, match="closed"):
        await adapter.execute_sync(lambda: None)

    assert adapter._get_execution_lane() is lane


@pytest.mark.asyncio
async def test_cancelled_queued_tts_never_enters_model_code() -> None:
    BlockingTTSAdapter.reset()
    adapter = BlockingTTSAdapter()
    first = asyncio.create_task(_consume(adapter, "first", []))
    assert await asyncio.to_thread(BlockingTTSAdapter.first_started.wait, 1.0)

    cancelled = asyncio.create_task(_consume(adapter, "cancelled", []))
    await asyncio.sleep(0.01)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    BlockingTTSAdapter.first_release.set()
    await first
    await _wait_until_idle(adapter)

    third_output: list[bytes] = []
    third = asyncio.create_task(_consume(adapter, "third", third_output))

    await third
    await _wait_until_idle(adapter)

    assert BlockingTTSAdapter.started_texts == ["first", "third"]
    assert third_output == [b"third"]
    assert BlockingTTSAdapter.max_active_calls == 1
    assert len(BlockingTTSAdapter.thread_ids) == 1
    adapter.close_execution_lane()


@pytest.mark.asyncio
async def test_scheduler_rejects_trim_and_unload_during_detached_physical_work() -> None:
    BlockingTTSAdapter.reset()
    scheduler = Scheduler(FakeRegistry(), default_device="cpu", max_loaded=1)
    first_output: list[bytes] = []

    async def run_request() -> None:
        async with scheduler.acquire("blocking:latest") as adapter:
            assert isinstance(adapter, BlockingTTSAdapter)
            await _consume(adapter, "first", first_output)

    request = asyncio.create_task(run_request())
    assert await asyncio.to_thread(BlockingTTSAdapter.first_started.wait, 1.0)

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    loaded = scheduler._models["blocking:latest"]
    adapter = loaded.adapter
    assert isinstance(adapter, BlockingTTSAdapter)
    assert loaded.ref_count == 0
    assert adapter.physical_work_count == 1
    loaded_info = scheduler.list_loaded()[0]
    assert loaded_info.ref_count == 0
    assert loaded_info.backend_memory["physical_work_count"] == 1
    assert loaded_info.is_evictable is False
    assert loaded_info.is_trimmable is False
    assert scheduler.memory_snapshot().active_model_count == 1
    assert await scheduler.trim("blocking:latest") is False
    assert await scheduler.unload("blocking:latest") is False
    assert adapter.trim_calls == 0

    BlockingTTSAdapter.first_release.set()
    await _wait_until_idle(adapter)

    assert await scheduler.trim("blocking:latest") is True
    assert adapter.trim_calls == 1
    assert await scheduler.unload("blocking:latest") is True
    assert first_output == []


@pytest.mark.asyncio
async def test_tts_inference_cannot_overlap_trim() -> None:
    trim_started = threading.Event()
    trim_release = threading.Event()
    synthesis_started = threading.Event()

    class BlockingTrimTTSAdapter(BlockingTTSAdapter):
        def trim(self) -> None:
            trim_started.set()
            trim_release.wait(5.0)
            self.trim_calls += 1

        async def synthesize(self, text: str, **kwargs: Any):
            synthesis_started.set()
            yield SynthesizeChunk(audio=b"ready", sample_rate=24_000, is_final=False)

    class TrimRegistry(FakeRegistry):
        def get_adapter_class(self, adapter_name: str) -> type:
            return BlockingTrimTTSAdapter

    scheduler = Scheduler(TrimRegistry(), default_device="cpu", max_loaded=1)
    async with scheduler.acquire("blocking:latest"):
        pass

    trim_task = asyncio.create_task(scheduler.trim("blocking:latest"))
    assert await asyncio.to_thread(trim_started.wait, 1.0)

    output: list[bytes] = []

    async def run_request() -> None:
        async with scheduler.acquire("blocking:latest") as adapter:
            assert isinstance(adapter, BlockingTrimTTSAdapter)
            await _consume(adapter, "after-trim", output)

    request = asyncio.create_task(run_request())
    await asyncio.sleep(0.05)
    assert synthesis_started.is_set() is False

    trim_release.set()
    assert await trim_task is True
    await request

    assert synthesis_started.is_set() is True
    assert output == [b"ready"]
    assert await scheduler.unload("blocking:latest") is True
