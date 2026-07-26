from __future__ import annotations

import asyncio

import pytest

from vox.grpc.streaming_queue import (
    close_grpc_output_queue,
    iter_grpc_output_queue,
    iter_grpc_stream_lifecycle,
    pump_events_to_grpc_queue,
    put_grpc_output_queue,
    start_grpc_event_pump,
)


class _Event:
    def __init__(self, value: str) -> None:
        self.value = value


class _SkipEvent:
    pass


class _DoneEvent:
    pass


async def _events(*events):
    for event in events:
        yield event


def _message(event) -> str | None:
    if isinstance(event, _Event):
        return event.value
    if isinstance(event, _DoneEvent):
        return "done"
    return None


@pytest.mark.asyncio
async def test_iter_grpc_output_queue_yields_until_sentinel():
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    await queue.put("a")
    await queue.put("b")
    await close_grpc_output_queue(queue)
    await queue.put("ignored")

    assert [item async for item in iter_grpc_output_queue(queue)] == ["a", "b"]


@pytest.mark.asyncio
async def test_pump_events_to_grpc_queue_filters_empty_messages_and_stops_on_terminal():
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    await pump_events_to_grpc_queue(
        _events(_Event("a"), _SkipEvent(), _DoneEvent(), _Event("ignored")),
        queue,
        message=_message,
        terminal_types=(_DoneEvent,),
    )

    assert [item async for item in iter_grpc_output_queue(queue)] == ["a", "done"]


@pytest.mark.asyncio
async def test_pump_events_to_grpc_queue_closes_queue_when_mapper_raises():
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def boom(_event) -> str:
        raise RuntimeError("mapper failed")

    with pytest.raises(RuntimeError, match="mapper failed"):
        await pump_events_to_grpc_queue(_events(_Event("a")), queue, message=boom)

    assert [item async for item in iter_grpc_output_queue(queue)] == []


@pytest.mark.asyncio
async def test_start_grpc_event_pump_starts_standard_queue_pump_task():
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    task = start_grpc_event_pump(
        _events(_Event("a"), _DoneEvent(), _Event("ignored")),
        queue,
        message=_message,
        terminal_types=(_DoneEvent,),
    )

    assert [item async for item in iter_grpc_output_queue(queue)] == ["a", "done"]
    await task


@pytest.mark.asyncio
async def test_iter_grpc_stream_lifecycle_reaps_tasks_and_runs_cleanup_after_queue_close():
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    await queue.put("a")
    await close_grpc_output_queue(queue)
    cleaned = False

    async def background() -> None:
        await asyncio.sleep(30)

    async def cleanup() -> None:
        nonlocal cleaned
        cleaned = True

    task = asyncio.create_task(background())

    assert [item async for item in iter_grpc_stream_lifecycle(queue, task, cleanup=cleanup)] == ["a"]

    assert task.done()
    assert cleaned is True


@pytest.mark.asyncio
async def test_iter_grpc_stream_lifecycle_runs_cleanup_when_consumer_stops_early():
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    await queue.put("a")
    await queue.put("b")
    cleaned = False

    async def background() -> None:
        await asyncio.sleep(30)

    async def cleanup() -> None:
        nonlocal cleaned
        cleaned = True

    task = asyncio.create_task(background())

    stream = iter_grpc_stream_lifecycle(queue, task, cleanup=cleanup)
    assert await stream.__anext__() == "a"
    await stream.aclose()

    assert task.done()
    assert cleaned is True


@pytest.mark.asyncio
async def test_grpc_producer_unblocks_when_bounded_queue_consumer_closes():
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)
    consumer_closed = asyncio.Event()
    queue.put_nowait("a")
    queue.put_nowait("b")
    producer = asyncio.create_task(
        put_grpc_output_queue(
            queue,
            "c",
            consumer_closed=consumer_closed,
        )
    )
    await asyncio.sleep(0)

    assert producer.done() is False

    consumer_closed.set()
    assert await asyncio.wait_for(producer, timeout=1.0) is False
    assert queue.get_nowait() == "a"
    assert queue.get_nowait() == "b"
