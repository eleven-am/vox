from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Final

RESPONSE_STREAM_END: Final = object()
RESPONSE_STREAM_QUEUE_MAX = 1024


@dataclass
class ResponseStream:
    queue: asyncio.Queue[str | object]
    response_id: str
    committed: bool = False
    pending_done: bool = False
    allow_interruptions: bool = True
    audio_started: bool = False
    text_parts: list[str] = field(default_factory=list)
    heard_parts: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, *, response_id: str, allow_interruptions: bool = True) -> ResponseStream:
        return cls(
            queue=asyncio.Queue(maxsize=RESPONSE_STREAM_QUEUE_MAX),
            response_id=response_id,
            allow_interruptions=allow_interruptions,
        )

    async def append_text(self, text: str) -> None:
        await self.queue.put(text)

    def mark_committed(self) -> bool:
        if self.committed:
            return False
        self.committed = True
        return True

    async def enqueue_end(self) -> None:
        await self.queue.put(RESPONSE_STREAM_END)

    async def next_text(self) -> str | None:
        item = await self.queue.get()
        if item is RESPONSE_STREAM_END:
            return None
        item_text = str(item)
        self.text_parts.append(item_text)
        return item_text

    def add_heard_text(self, text: str) -> None:
        self.heard_parts.append(text)

    def assistant_context_text(self, *, separator: str = "") -> str:
        if not separator:
            heard_text = "".join(self.heard_parts).strip()
            if heard_text:
                return heard_text
            return "".join(self.text_parts).strip()

        heard_text = separator.join(part.strip() for part in self.heard_parts if part.strip()).strip()
        if heard_text:
            return heard_text
        return separator.join(part.strip() for part in self.text_parts if part.strip()).strip()
