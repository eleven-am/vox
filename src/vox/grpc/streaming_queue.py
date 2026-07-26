from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import TypeVar

from vox.core.tasks import reap_task

EventT = TypeVar("EventT")
MessageT = TypeVar("MessageT")
GRPC_OUTPUT_QUEUE_MAX = 64


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


async def put_grpc_output_queue(
    queue: asyncio.Queue[MessageT | None],
    item: MessageT,
    *,
    consumer_closed: asyncio.Event,
) -> bool:
    if consumer_closed.is_set():
        return False
    put_task = asyncio.create_task(queue.put(item))
    close_task = asyncio.create_task(consumer_closed.wait())
    try:
        done, _pending = await asyncio.wait(
            (put_task, close_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if close_task in done:
            await close_task
            return False
        await put_task
        return True
    finally:
        if not put_task.done():
            await reap_task(put_task)
        if not close_task.done():
            await reap_task(close_task)


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
    on_consumer_close: Callable[[], None] | None = None,
) -> AsyncIterator[MessageT]:
    try:
        async for item in iter_grpc_output_queue(queue):
            yield item
    finally:
        if on_consumer_close is not None:
            on_consumer_close()
        for task in tasks:
            await reap_task(task)
        if cleanup is not None:
            await cleanup()
