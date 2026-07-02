from __future__ import annotations

import asyncio

import pytest

from vox.server.conversation_cleanup import close_conversation_runtime_resources


class FakeOrchestrator:
    def __init__(self) -> None:
        self.ended = False
        self.closed = False

    async def end_of_stream(self) -> None:
        self.ended = True

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_close_conversation_runtime_resources_flushes_events_before_close():
    orchestrator = FakeOrchestrator()
    released = asyncio.Event()
    observed_ended = False

    async def emit_events() -> None:
        nonlocal observed_ended
        await released.wait()
        observed_ended = orchestrator.ended

    emit_task = asyncio.create_task(emit_events())

    async def release_after_end() -> None:
        while not orchestrator.ended:
            await asyncio.sleep(0)
        released.set()

    release_task = asyncio.create_task(release_after_end())

    await close_conversation_runtime_resources(
        orchestrator=orchestrator,
        emit_task=emit_task,
    )

    await release_task
    assert observed_ended is True
    assert emit_task.done()
    assert orchestrator.closed is True
