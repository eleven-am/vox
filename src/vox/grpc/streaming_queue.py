from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import TypeVar

from vox.core.tasks import reap_task

EventT = TypeVar("EventT")
MessageT = TypeVar("MessageT")


async def iter_grpc_output_queue(
    queue: asyncio.Queue[MessageT | None],
) -> AsyncIterator[MessageT]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def close_grpc_output_queue(queue: asyncio.Queue[MessageT | None]) -> None:
    await queue.put(None)


async def pump_events_to_grpc_queue(
    events: AsyncIterable[EventT],
    queue: asyncio.Queue[MessageT | None],
    *,
    message: Callable[[EventT], MessageT | None],
    terminal_types: tuple[type, ...] = (),
) -> None:
    try:
        async for event in events:
            item = message(event)
            if item is not None:
                await queue.put(item)
            if terminal_types and isinstance(event, terminal_types):
                break
    finally:
        await close_grpc_output_queue(queue)


def start_grpc_event_pump(
    events: AsyncIterable[EventT],
    queue: asyncio.Queue[MessageT | None],
    *,
    message: Callable[[EventT], MessageT | None],
    terminal_types: tuple[type, ...] = (),
) -> asyncio.Task[None]:
    return asyncio.create_task(
        pump_events_to_grpc_queue(
            events,
            queue,
            message=message,
            terminal_types=terminal_types,
        )
    )


async def iter_grpc_stream_lifecycle(
    queue: asyncio.Queue[MessageT | None],
    *tasks: asyncio.Task,
    cleanup: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[MessageT]:
    try:
        async for item in iter_grpc_output_queue(queue):
            yield item
    finally:
        for task in tasks:
            await reap_task(task)
        if cleanup is not None:
            await cleanup()
