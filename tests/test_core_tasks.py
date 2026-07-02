import asyncio
from contextlib import suppress

import pytest

from vox.core.tasks import drain_task, reap_task


@pytest.mark.asyncio
async def test_reap_task_swallows_cancelled_error():
    started = asyncio.Event()

    async def hang():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(hang())
    await started.wait()
    await reap_task(task)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_reap_task_swallows_task_exception():
    async def boom():
        raise ValueError("boom")

    task = asyncio.create_task(boom())
    await asyncio.sleep(0)
    await reap_task(task)
    assert task.done()


@pytest.mark.asyncio
async def test_reap_task_accepts_none():
    await reap_task(None)


@pytest.mark.asyncio
async def test_drain_task_returns_result_before_timeout():
    async def quick():
        return 42

    task = asyncio.create_task(quick())
    await drain_task(task, timeout=1.0)
    assert task.done()
    assert task.result() == 42


@pytest.mark.asyncio
async def test_drain_task_cancels_after_timeout_without_raising():
    started = asyncio.Event()

    async def hang():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(hang())
    await started.wait()
    await drain_task(task, timeout=0.01)
    assert task.done()


@pytest.mark.asyncio
async def test_drain_task_swallows_task_exception():
    async def boom():
        raise RuntimeError("boom")

    task = asyncio.create_task(boom())
    await drain_task(task, timeout=1.0)
    assert task.done()


@pytest.mark.asyncio
async def test_drain_task_returns_even_if_task_ignores_cancellation():
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn():
        started.set()
        while True:
            try:
                await release.wait()
                return
            except asyncio.CancelledError:
                pass

    task = asyncio.create_task(stubborn())
    await started.wait()
    async with asyncio.timeout(2):
        await drain_task(task, timeout=0.01)
    release.set()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert task.done()
