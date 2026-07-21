from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConversationTimerLease:
    key: str


TimerExpiredHandler = Callable[[ConversationTimerLease], Awaitable[None]]


class ConversationTimerRegistry:
    def __init__(self, on_expired: TimerExpiredHandler) -> None:
        self._on_expired = on_expired
        self._timers: dict[str, tuple[ConversationTimerLease, asyncio.Task]] = {}

    def has_active(self, key: str) -> bool:
        return key in self._timers

    def has_any_active(self) -> bool:
        return bool(self._timers)

    async def start(self, key: str, duration_ms: int) -> ConversationTimerLease:
        await self.cancel(key)
        lease = ConversationTimerLease(key=key)
        task = asyncio.create_task(self._timer_task(lease, duration_ms))
        self._timers[key] = (lease, task)
        return lease

    async def cancel(self, key: str) -> None:
        entry = self._timers.pop(key, None)
        if entry is None:
            return
        _, task = entry
        if task is not asyncio.current_task() and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

    def cancel_all(self) -> None:
        entries = tuple(self._timers.values())
        self._timers.clear()
        for _, task in entries:
            task.cancel()

    def consume(self, lease: ConversationTimerLease) -> bool:
        entry = self._timers.get(lease.key)
        if entry is None or entry[0] is not lease:
            return False
        self._timers.pop(lease.key, None)
        return True

    async def _timer_task(self, lease: ConversationTimerLease, duration_ms: int) -> None:
        try:
            await asyncio.sleep(duration_ms / 1000.0)
        except asyncio.CancelledError:
            return

        entry = self._timers.get(lease.key)
        if entry is None or entry[0] is not lease or entry[1] is not asyncio.current_task():
            return
        await self._on_expired(lease)
