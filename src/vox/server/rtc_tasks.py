from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


def track_task(tasks: set[asyncio.Task], coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    task = asyncio.create_task(coro)
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


def track_media_task(record: Any, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    media_tasks = getattr(record, "media_tasks", None)
    if media_tasks is not None:
        return track_task(media_tasks, coro)
    task = asyncio.create_task(coro)
    return task


def cancel_media_tasks(record: Any) -> list[asyncio.Task]:
    tasks = list(getattr(record, "media_tasks", ()))
    for task in tasks:
        task.cancel()
    media_tasks = getattr(record, "media_tasks", None)
    if media_tasks is not None:
        media_tasks.clear()
    return tasks


async def cancel_and_drain_media_tasks(record: Any) -> None:
    tasks = cancel_media_tasks(record)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
