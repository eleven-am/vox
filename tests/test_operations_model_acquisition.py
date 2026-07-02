from __future__ import annotations

import pytest

from tests.fakes import FakeScheduler, FakeSTTAdapter, FakeTTSAdapter
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.errors import ModelNotFoundError
from vox.operations.errors import StoredModelNotFoundError, WrongModelTypeError
from vox.operations.model_acquisition import (
    acquire_typed_adapter,
    enter_typed_adapter,
    release_entered_adapter,
    release_entered_adapter_suppressing,
)


class TrackingManager:
    def __init__(self, adapter):
        self.adapter = adapter
        self.closed = False

    async def __aenter__(self):
        return self.adapter

    async def __aexit__(self, _exc_type, _exc, _tb):
        self.closed = True
        return False


class TrackingScheduler:
    def __init__(self, adapter):
        self.manager = TrackingManager(adapter)

    def acquire(self, _model: str):
        return self.manager


class RaisingExitManager(TrackingManager):
    async def __aexit__(self, _exc_type, _exc, _tb):
        self.closed = True
        raise RuntimeError("close failed")


class RaisingExitScheduler(TrackingScheduler):
    def __init__(self, adapter):
        self.manager = RaisingExitManager(adapter)


@pytest.mark.asyncio
async def test_acquire_typed_adapter_yields_matching_adapter():
    scheduler = FakeScheduler(FakeSTTAdapter())

    async with acquire_typed_adapter(
        scheduler,
        model="fake-stt:latest",
        adapter_type=STTAdapter,
        expected_type="STT",
    ) as adapter:
        assert isinstance(adapter, STTAdapter)


@pytest.mark.asyncio
async def test_acquire_typed_adapter_maps_missing_model_to_operation_error():
    scheduler = FakeScheduler()

    with pytest.raises(StoredModelNotFoundError, match="missing:latest"):
        async with acquire_typed_adapter(
            scheduler,
            model="missing:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        ):
            raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_acquire_typed_adapter_maps_immediate_acquire_failure():
    class ImmediateMissingScheduler:
        def acquire(self, model: str):
            raise ModelNotFoundError(model)

    with pytest.raises(StoredModelNotFoundError, match="missing:latest"):
        async with acquire_typed_adapter(
            ImmediateMissingScheduler(),
            model="missing:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        ):
            raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_acquire_typed_adapter_rejects_wrong_adapter_type():
    scheduler = FakeScheduler(FakeTTSAdapter())

    with pytest.raises(WrongModelTypeError, match="not an STT model"):
        async with acquire_typed_adapter(
            scheduler,
            model="fake-tts:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        ):
            raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_acquire_typed_adapter_accepts_tts_when_requested():
    scheduler = FakeScheduler(FakeTTSAdapter())

    async with acquire_typed_adapter(
        scheduler,
        model="fake-tts:latest",
        adapter_type=TTSAdapter,
        expected_type="TTS",
    ) as adapter:
        assert isinstance(adapter, TTSAdapter)


@pytest.mark.asyncio
async def test_acquire_typed_adapter_does_not_rewrite_body_model_errors():
    scheduler = FakeScheduler(FakeSTTAdapter())

    with pytest.raises(ModelNotFoundError, match="body:latest"):
        async with acquire_typed_adapter(
            scheduler,
            model="fake-stt:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        ):
            raise ModelNotFoundError("body:latest")


@pytest.mark.asyncio
async def test_enter_typed_adapter_returns_manager_and_adapter():
    scheduler = TrackingScheduler(FakeSTTAdapter())

    entered = await enter_typed_adapter(
        scheduler,
        model="fake-stt:latest",
        adapter_type=STTAdapter,
        expected_type="STT",
    )

    assert isinstance(entered.adapter, STTAdapter)
    assert entered.manager is scheduler.manager
    assert not scheduler.manager.closed

    await release_entered_adapter(entered)
    assert scheduler.manager.closed


@pytest.mark.asyncio
async def test_release_entered_adapter_propagates_close_failure():
    scheduler = RaisingExitScheduler(FakeSTTAdapter())
    entered = await enter_typed_adapter(
        scheduler,
        model="fake-stt:latest",
        adapter_type=STTAdapter,
        expected_type="STT",
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await release_entered_adapter(entered)

    assert scheduler.manager.closed


@pytest.mark.asyncio
async def test_release_entered_adapter_suppressing_closes_without_leaking_failure():
    scheduler = RaisingExitScheduler(FakeSTTAdapter())
    entered = await enter_typed_adapter(
        scheduler,
        model="fake-stt:latest",
        adapter_type=STTAdapter,
        expected_type="STT",
    )

    await release_entered_adapter_suppressing(entered)

    assert scheduler.manager.closed


@pytest.mark.asyncio
async def test_enter_typed_adapter_maps_immediate_acquire_failure():
    class ImmediateMissingScheduler:
        def acquire(self, model: str):
            raise ModelNotFoundError(model)

    with pytest.raises(StoredModelNotFoundError, match="missing:latest"):
        await enter_typed_adapter(
            ImmediateMissingScheduler(),
            model="missing:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        )


@pytest.mark.asyncio
async def test_enter_typed_adapter_rejects_wrong_adapter_type_and_closes_manager():
    scheduler = TrackingScheduler(FakeTTSAdapter())

    with pytest.raises(WrongModelTypeError, match="not an STT model"):
        await enter_typed_adapter(
            scheduler,
            model="fake-tts:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        )

    assert scheduler.manager.closed
