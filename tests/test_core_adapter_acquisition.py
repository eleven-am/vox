from __future__ import annotations

import pytest

from tests.fakes import FakeSTTAdapter, FakeTTSAdapter
from vox.core.adapter import STTAdapter
from vox.core.adapter_acquisition import (
    AdapterTypeMismatchError,
    acquire_typed_adapter,
    enter_typed_adapter,
    release_entered_adapter,
    release_entered_adapter_suppressing,
)
from vox.core.errors import ModelNotFoundError


class TrackingManager:
    def __init__(self, adapter):
        self.adapter = adapter
        self.closed = False
        self.exit_exc_type = None

    async def __aenter__(self):
        return self.adapter

    async def __aexit__(self, exc_type, _exc, _tb):
        self.closed = True
        self.exit_exc_type = exc_type
        return False


class TrackingScheduler:
    def __init__(self, adapter):
        self.manager = TrackingManager(adapter)

    def acquire(self, _model: str):
        return self.manager


class MissingScheduler:
    def acquire(self, model: str):
        raise ModelNotFoundError(model)


@pytest.mark.asyncio
async def test_acquire_typed_adapter_yields_adapter_and_closes_manager():
    scheduler = TrackingScheduler(FakeSTTAdapter())

    async with acquire_typed_adapter(
        scheduler,
        model="fake-stt:latest",
        adapter_type=STTAdapter,
        expected_type="STT",
    ) as adapter:
        assert isinstance(adapter, STTAdapter)
        assert not scheduler.manager.closed

    assert scheduler.manager.closed
    assert scheduler.manager.exit_exc_type is None


@pytest.mark.asyncio
async def test_acquire_typed_adapter_passes_body_exception_to_manager_exit():
    scheduler = TrackingScheduler(FakeSTTAdapter())

    with pytest.raises(RuntimeError, match="body failed"):
        async with acquire_typed_adapter(
            scheduler,
            model="fake-stt:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        ):
            raise RuntimeError("body failed")

    assert scheduler.manager.closed
    assert scheduler.manager.exit_exc_type is RuntimeError


@pytest.mark.asyncio
async def test_enter_typed_adapter_rejects_wrong_type_and_closes_manager():
    scheduler = TrackingScheduler(FakeTTSAdapter())

    with pytest.raises(AdapterTypeMismatchError, match="fake-tts:latest"):
        await enter_typed_adapter(
            scheduler,
            model="fake-tts:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        )

    assert scheduler.manager.closed


@pytest.mark.asyncio
async def test_enter_typed_adapter_propagates_missing_model_from_scheduler():
    with pytest.raises(ModelNotFoundError, match="missing:latest"):
        await enter_typed_adapter(
            MissingScheduler(),
            model="missing:latest",
            adapter_type=STTAdapter,
            expected_type="STT",
        )


@pytest.mark.asyncio
async def test_release_helpers_close_or_suppress_close_failure():
    class RaisingManager(TrackingManager):
        async def __aexit__(self, _exc_type, _exc, _tb):
            self.closed = True
            raise RuntimeError("close failed")

    entered = await enter_typed_adapter(
        TrackingScheduler(FakeSTTAdapter()),
        model="fake-stt:latest",
        adapter_type=STTAdapter,
        expected_type="STT",
    )
    await release_entered_adapter(entered)

    scheduler = TrackingScheduler(FakeSTTAdapter())
    scheduler.manager = RaisingManager(FakeSTTAdapter())
    entered = await enter_typed_adapter(
        scheduler,
        model="fake-stt:latest",
        adapter_type=STTAdapter,
        expected_type="STT",
    )
    await release_entered_adapter_suppressing(entered)
    assert scheduler.manager.closed
