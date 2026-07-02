import asyncio

import pytest

from vox.server.rtc_tasks import (
    cancel_and_drain_media_tasks,
    cancel_media_tasks,
    track_media_task,
    track_task,
)


async def _wait_forever() -> None:
    await asyncio.Event().wait()


async def _return_value(value: str) -> str:
    return value


@pytest.mark.asyncio
async def test_track_task_registers_and_discards_completed_task():
    tasks = set()

    task = track_task(tasks, _return_value("done"))

    assert task in tasks
    assert await task == "done"
    await asyncio.sleep(0)
    assert task not in tasks


@pytest.mark.asyncio
async def test_track_media_task_registers_and_discards_completed_task():
    class Record:
        media_tasks = set()

    task = track_media_task(Record(), _return_value("done"))

    assert task in Record.media_tasks
    assert await task == "done"
    await asyncio.sleep(0)
    assert task not in Record.media_tasks


@pytest.mark.asyncio
async def test_track_media_task_allows_records_without_media_task_set():
    class Record:
        pass

    task = track_media_task(Record(), _return_value("done"))

    assert await task == "done"


@pytest.mark.asyncio
async def test_cancel_media_tasks_cancels_and_clears_tracked_tasks():
    task = asyncio.create_task(_wait_forever())

    class Record:
        media_tasks = {task}

    cancelled = cancel_media_tasks(Record())
    await asyncio.gather(*cancelled, return_exceptions=True)

    assert cancelled == [task]
    assert task.cancelled()
    assert Record.media_tasks == set()


@pytest.mark.asyncio
async def test_cancel_media_tasks_is_noop_without_tracked_tasks():
    class Record:
        pass

    assert cancel_media_tasks(Record()) == []


@pytest.mark.asyncio
async def test_cancel_and_drain_media_tasks_awaits_cancelled_tasks():
    task = asyncio.create_task(_wait_forever())

    class Record:
        media_tasks = {task}

    await cancel_and_drain_media_tasks(Record())

    assert task.cancelled()
    assert Record.media_tasks == set()
