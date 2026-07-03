from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import Future
from typing import TypeVar, cast

T = TypeVar("T")


async def iterate_off_event_loop(iterator: AsyncIterator[T]) -> AsyncIterator[T]:
    """Consume a potentially blocking async iterator outside the server loop.

    Some model adapters expose an async generator but perform heavy synchronous
    work before yielding. Running that generator on the Uvicorn event loop can
    starve health checks and RTC control traffic. This bridge keeps adapter work
    on a worker thread and forwards yielded chunks back to the caller's loop.
    """

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
    worker_loop_ready: Future[asyncio.AbstractEventLoop] = Future()
    worker_task_ready: Future[asyncio.Task[None]] = Future()

    def emit(kind: str, payload: object = None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    async def consume() -> None:
        try:
            async for item in iterator:
                emit("item", item)
        except BaseException as exc:  # noqa: BLE001 - must preserve adapter failures
            emit("error", exc)
        else:
            emit("done")

    def run() -> None:
        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)
        task = worker_loop.create_task(consume())
        worker_loop_ready.set_result(worker_loop)
        worker_task_ready.set_result(task)
        try:
            worker_loop.run_until_complete(task)
        finally:
            worker_loop.close()

    producer = asyncio.create_task(asyncio.to_thread(run))
    finished = False
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "item":
                yield cast(T, payload)
            elif kind == "error":
                finished = True
                await producer
                raise cast(BaseException, payload)
            else:
                finished = True
                break
        if not producer.done():
            await producer
    finally:
        if not finished and not producer.done():
            if worker_loop_ready.done() and worker_task_ready.done():
                worker_loop = worker_loop_ready.result()
                worker_task = worker_task_ready.result()
                if not worker_loop.is_closed() and not worker_task.done():
                    worker_loop.call_soon_threadsafe(worker_task.cancel)
            producer.cancel()
