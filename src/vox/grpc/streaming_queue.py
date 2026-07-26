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


async def close_grpc_output_queue(
    queue: asyncio.Queue[MessageT | None],
    *,
    consumer_closed: asyncio.Event | None = None,
) -> None:
    if consumer_closed is None:
        await queue.put(None)
        return
    if consumer_closed.is_set():
        return
    put_task = asyncio.create_task(queue.put(None))
    close_task = asyncio.create_task(consumer_closed.wait())
    try:
        done, _pending = await asyncio.wait(
            (put_task, close_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if put_task in done:
            await put_task
    finally:
        if not put_task.done():
            await reap_task(put_task)
        if not close_task.done():
            await reap_task(close_task)


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
    consumer_closed: asyncio.Event | None = None,
) -> None:
    try:
        async for event in events:
            item = message(event)
            if item is not None:
                if consumer_closed is None:
                    await queue.put(item)
                elif not await put_grpc_output_queue(
                    queue,
                    item,
                    consumer_closed=consumer_closed,
                ):
                    return
            if terminal_types and isinstance(event, terminal_types):
                break
    finally:
        task = asyncio.current_task()
        if task is None or not task.cancelling():
            await close_grpc_output_queue(
                queue,
                consumer_closed=consumer_closed,
            )


def start_grpc_event_pump(
    events: AsyncIterable[EventT],
    queue: asyncio.Queue[MessageT | None],
    *,
    message: Callable[[EventT], MessageT | None],
    terminal_types: tuple[type, ...] = (),
    consumer_closed: asyncio.Event | None = None,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        pump_events_to_grpc_queue(
            events,
            queue,
            message=message,
            terminal_types=terminal_types,
            consumer_closed=consumer_closed,
        )
    )


async def iter_grpc_stream_lifecycle(
    queue: asyncio.Queue[MessageT | None],
    *tasks: asyncio.Task,
    cleanup: Callable[[], Awaitable[None]] | None = None,
    on_consumer_close: Callable[[], None] | None = None,
) -> AsyncIterator[MessageT]:
    reached_stream_end = False
    try:
        async for item in iter_grpc_output_queue(queue):
            yield item
        reached_stream_end = True
    finally:
        if on_consumer_close is not None:
            on_consumer_close()
        for task in tasks:
            await reap_task(task)
        task_errors: list[BaseException] = []
        if reached_stream_end:
            for task in tasks:
                if task.cancelled() or not task.done():
                    continue
                error = task.exception()
                if error is not None:
                    task_errors.append(error)
        cleanup_error: Exception | None = None
        try:
            if cleanup is not None:
                await cleanup()
        except Exception as exc:
            cleanup_error = exc
        if task_errors and cleanup_error is not None:
            raise BaseExceptionGroup(
                "gRPC stream and cleanup failed",
                [*task_errors, cleanup_error],
            )
        if task_errors:
            raise task_errors[0]
        if cleanup_error is not None:
            raise cleanup_error
