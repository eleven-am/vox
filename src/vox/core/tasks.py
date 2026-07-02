from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any


def _consume_result(task: asyncio.Task[Any]) -> None:
    if task.done() and not task.cancelled():
        with suppress(Exception):
            task.exception()


async def reap_task(task: asyncio.Task[Any] | None, *, timeout: float = 5.0) -> None:
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await asyncio.wait({task}, timeout=timeout)
    _consume_result(task)


async def drain_task(task: asyncio.Task[Any] | None, *, timeout: float = 5.0) -> None:
    if task is None:
        return
    with suppress(asyncio.CancelledError, Exception):
        await asyncio.wait({task}, timeout=timeout)
    if not task.done():
        await reap_task(task, timeout=timeout)
    else:
        _consume_result(task)
