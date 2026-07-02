from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress

TimerExpiredHandler = Callable[[str], Awaitable[None]]


class ConversationTimerRegistry:
    def __init__(self, on_expired: TimerExpiredHandler) -> None:
        self._on_expired = on_expired
        self._timers: dict[str, asyncio.Task] = {}

    def has_active(self, key: str) -> bool:
        task = self._timers.get(key)
        return task is not None and not task.done()

    def has_any_active(self) -> bool:
        return any(not task.done() for task in self._timers.values())

    async def start(self, key: str, duration_ms: int) -> None:
        await self.cancel(key)
        self._timers[key] = asyncio.create_task(self._timer_task(key, duration_ms))

    async def cancel(self, key: str) -> None:
        task = self._timers.pop(key, None)
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

    def cancel_all(self) -> None:
        for task in list(self._timers.values()):
            task.cancel()
        self._timers.clear()

    async def _timer_task(self, key: str, duration_ms: int) -> None:
        try:
            await asyncio.sleep(duration_ms / 1000.0)
        except asyncio.CancelledError:
            return

        if self._timers.get(key) is not asyncio.current_task():
            return
        self._timers.pop(key, None)
        await self._on_expired(key)
