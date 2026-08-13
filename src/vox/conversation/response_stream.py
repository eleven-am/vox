from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from vox.conversation.response_output import ResponseOutputConfig
from vox.conversation.spoken_history import ResponseSpokenHistory, SpokenHistorySnapshot

RESPONSE_STREAM_END: Final = object()
RESPONSE_STREAM_QUEUE_MAX = 1024


class AppendResult(StrEnum):
    ACCEPTED = "accepted"
    SESSION_CLOSED = "session_closed"
    NO_ACTIVE_RESPONSE = "no_active_response"
    RESPONSE_MISMATCH = "response_mismatch"
    RESPONSE_COMMITTED = "response_committed"
    STREAM_ENDED = "stream_ended"

    @property
    def is_accepted(self) -> bool:
        return self is AppendResult.ACCEPTED


@dataclass
class ResponseStream:
    queue: asyncio.Queue[str | object]
    response_id: str
    output: ResponseOutputConfig
    generation_id: str | None = None
    supersedes_generation_id: str | None = None
    committed: bool = False
    pending_done: bool = False
    allow_interruptions: bool = True
    audio_started: bool = False
    closed: bool = False
    text_parts: list[str] = field(default_factory=list)
    spoken_history: ResponseSpokenHistory = field(
        default_factory=lambda: ResponseSpokenHistory(playout_available=False)
    )
    _closed_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @classmethod
    def create(
        cls,
        *,
        response_id: str,
        output: ResponseOutputConfig,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
        supersedes_generation_id: str | None = None,
        playout_observed: bool = False,
    ) -> ResponseStream:
        return cls(
            queue=asyncio.Queue(maxsize=RESPONSE_STREAM_QUEUE_MAX),
            response_id=response_id,
            output=output,
            generation_id=generation_id,
            supersedes_generation_id=supersedes_generation_id,
            allow_interruptions=allow_interruptions,
            spoken_history=ResponseSpokenHistory(playout_available=playout_observed),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._closed_event.set()
        self._discard_queued_items()

    async def append_text(self, text: str) -> AppendResult:
        return await self._enqueue(text)

    def mark_committed(self) -> bool:
        if self.committed:
            return False
        self.committed = True
        return True

    async def enqueue_end(self) -> AppendResult:
        return await self._enqueue(RESPONSE_STREAM_END)

    async def next_text(self) -> str | None:
        if self.closed:
            return None
        get_task = asyncio.create_task(self.queue.get())
        close_task = asyncio.create_task(self._closed_event.wait())
        try:
            await asyncio.wait({get_task, close_task}, return_when=asyncio.FIRST_COMPLETED)
            if self.closed or close_task.done():
                return None
            item = get_task.result()
        finally:
            for task in (get_task, close_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(get_task, close_task, return_exceptions=True)
        if item is RESPONSE_STREAM_END:
            return None
        item_text = str(item)
        self.text_parts.append(item_text)
        return item_text

    async def _enqueue(self, item: str | object) -> AppendResult:
        if self.closed:
            return AppendResult.STREAM_ENDED
        put_task = asyncio.create_task(self.queue.put(item))
        close_task = asyncio.create_task(self._closed_event.wait())
        try:
            await asyncio.wait({put_task, close_task}, return_when=asyncio.FIRST_COMPLETED)
            if self.closed or close_task.done():
                self._discard_queued_items()
                return AppendResult.STREAM_ENDED
            await put_task
            if self.closed:
                self._discard_queued_items()
                return AppendResult.STREAM_ENDED
            return AppendResult.ACCEPTED
        finally:
            for task in (put_task, close_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put_task, close_task, return_exceptions=True)

    def _discard_queued_items(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def assistant_context_text(self, *, separator: str = "") -> str:
        if not separator:
            return "".join(self.text_parts).strip()
        return separator.join(part.strip() for part in self.text_parts if part.strip()).strip()

    def spoken_snapshot(self) -> SpokenHistorySnapshot:
        return self.spoken_history.snapshot()

    def spoken_context_text(self) -> str:
        return self.spoken_history.completed_text()
