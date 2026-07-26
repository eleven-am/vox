from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from contextlib import suppress
from typing import Generic, ParamSpec, TypeVar, cast

from vox.core.errors import AdapterExecutionBusyError

T = TypeVar("T")
P = ParamSpec("P")
_SKIPPED = object()
_DEFAULT_MAX_PENDING = 2
_DEFAULT_OUTPUT_QUEUE_SIZE = 2


class _SyncJob(Generic[T]):
    def __init__(self, operation: Callable[[], T]) -> None:
        self._operation = operation
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> T | object:
        if self._cancelled.is_set():
            return _SKIPPED
        return self._operation()


class _IteratorJob(Generic[T]):
    def __init__(
        self,
        iterator: AsyncIterator[T],
        emit: Callable[[str, object], None],
    ) -> None:
        self._iterator = iterator
        self._emit = emit
        self._cancelled = threading.Event()
        self._state_lock = threading.Lock()
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._state_lock:
            worker_loop = self._worker_loop
            worker_task = self._worker_task
        if (
            worker_loop is not None
            and worker_task is not None
            and not worker_loop.is_closed()
            and not worker_task.done()
        ):
            with suppress(RuntimeError):
                worker_loop.call_soon_threadsafe(worker_task.cancel)

    async def _consume(self) -> None:
        async for item in self._iterator:
            if not self._cancelled.is_set():
                self._emit("item", item)

    def run(self) -> None:
        if self._cancelled.is_set():
            return
        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)
        worker_task = worker_loop.create_task(self._consume())
        with self._state_lock:
            self._worker_loop = worker_loop
            self._worker_task = worker_task
        if self._cancelled.is_set():
            worker_task.cancel()
        try:
            worker_loop.run_until_complete(worker_task)
        except asyncio.CancelledError:
            pass
        finally:
            cleanup_error: BaseException | None = None
            try:
                worker_loop.run_until_complete(worker_loop.shutdown_asyncgens())
            except BaseException as error:
                cleanup_error = error
            try:
                worker_loop.run_until_complete(worker_loop.shutdown_default_executor())
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            finally:
                with self._state_lock:
                    self._worker_task = None
                    self._worker_loop = None
                asyncio.set_event_loop(None)
                worker_loop.close()
            if cleanup_error is not None:
                raise cleanup_error


class AdapterExecutionLane:
    def __init__(
        self,
        *,
        max_pending: int = _DEFAULT_MAX_PENDING,
        output_queue_size: int = _DEFAULT_OUTPUT_QUEUE_SIZE,
    ) -> None:
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._pending_count = 0
        self._closed = False
        self._max_pending = max(1, int(max_pending))
        self._output_queue_size = max(1, int(output_queue_size))
        self._idle = threading.Event()
        self._idle.set()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return self._pending_count

    def _submit(
        self,
        operation: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        with self._lock:
            if self._closed:
                raise RuntimeError("adapter execution lane is closed")
            if self._pending_count >= self._max_pending:
                raise AdapterExecutionBusyError()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="vox-adapter",
                )
            self._pending_count += 1
            self._idle.clear()
            executor = self._executor
        try:
            future = executor.submit(operation, *args, **kwargs)
        except BaseException:
            with self._lock:
                self._pending_count -= 1
                if self._pending_count == 0:
                    self._idle.set()
            raise

        def complete(_future: Future[T]) -> None:
            with self._lock:
                self._pending_count -= 1
                if self._pending_count == 0:
                    self._idle.set()

        future.add_done_callback(complete)
        return future

    async def iterate(self, iterator: AsyncIterator[T]) -> AsyncIterator[T]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=self._output_queue_size)
        job: _IteratorJob[T]

        def emit(kind: str, payload: object) -> None:
            try:
                put = asyncio.run_coroutine_threadsafe(queue.put((kind, payload)), loop)
            except RuntimeError:
                return
            while True:
                try:
                    put.result(timeout=0.05)
                    return
                except FutureTimeoutError:
                    if job.cancelled:
                        put.cancel()
                        return
                except BaseException:
                    return

        job = _IteratorJob(iterator, emit)
        concurrent_future = self._submit(job.run)
        producer = asyncio.wrap_future(concurrent_future, loop=loop)
        next_item: asyncio.Task[tuple[str, object]] | None = None
        try:
            while True:
                if not queue.empty():
                    _kind, payload = queue.get_nowait()
                    yield cast(T, payload)
                    continue
                if producer.done():
                    await asyncio.shield(producer)
                    return
                next_item = asyncio.create_task(queue.get())
                done, _pending = await asyncio.wait(
                    (next_item, producer),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if next_item in done:
                    _kind, payload = next_item.result()
                    next_item = None
                    yield cast(T, payload)
                    continue
                next_item.cancel()
                with suppress(asyncio.CancelledError):
                    await next_item
                next_item = None
                await asyncio.shield(producer)
                return
        finally:
            job.cancel()
            if next_item is not None and not next_item.done():
                next_item.cancel()
                with suppress(asyncio.CancelledError):
                    await next_item

    def run_sync(
        self,
        operation: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        return self._submit(operation, *args, **kwargs).result()

    async def run(self, operation: Callable[[], T]) -> T:
        loop = asyncio.get_running_loop()
        job = _SyncJob(operation)
        concurrent_future = self._submit(job.run)
        producer = asyncio.wrap_future(concurrent_future, loop=loop)
        try:
            result = await asyncio.shield(producer)
        except asyncio.CancelledError:
            job.cancel()
            raise
        if result is _SKIPPED:
            raise asyncio.CancelledError
        return cast(T, result)

    async def wait_idle(self, *, timeout: float | None = None) -> None:
        completed = await asyncio.to_thread(self._idle.wait, timeout)
        if not completed:
            raise TimeoutError("adapter execution lane did not become idle")

    def close(self) -> None:
        with self._lock:
            if self._pending_count:
                raise RuntimeError("cannot close adapter execution lane while work is active")
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
