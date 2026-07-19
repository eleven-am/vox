from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vox.conversation.response_stream import AppendResult, ResponseStream

TerminalReason = Literal["done", "cancelled", "failed"]


@dataclass(frozen=True)
class TerminalRecord:
    response_id: str
    generation_id: str | None
    reason: TerminalReason


@dataclass
class ConversationResponseLifecycle:
    counter: int = 0
    stream: ResponseStream | None = None
    terminal: TerminalRecord | None = None

    @property
    def response_active(self) -> bool:
        return self.stream is not None and not self.stream.closed

    def open_uncommitted_stream(self) -> ResponseStream | None:
        stream = self.stream
        if stream is not None and not stream.committed and not stream.closed:
            return stream
        return None

    def appendable_stream(self, expected_response_id: str) -> ResponseStream | AppendResult:
        stream = self.stream
        if stream is None:
            return AppendResult.NO_ACTIVE_RESPONSE
        if stream.response_id != expected_response_id:
            return AppendResult.RESPONSE_MISMATCH
        if stream.committed:
            return AppendResult.RESPONSE_COMMITTED
        if stream.closed:
            return AppendResult.STREAM_ENDED
        return stream

    def start_stream(
        self,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
    ) -> ResponseStream:
        self.counter += 1
        stream = ResponseStream.create(
            response_id=f"resp_{self.counter}",
            allow_interruptions=allow_interruptions,
            generation_id=generation_id,
        )
        self.stream = stream
        self.terminal = None
        return stream

    def commit_stream(self, stream: ResponseStream) -> bool:
        if self.stream is not stream or stream.closed:
            return False
        return stream.mark_committed()

    def terminalize(self, stream: ResponseStream, reason: TerminalReason) -> TerminalRecord | None:
        if stream is not self.stream:
            return None
        record = TerminalRecord(
            response_id=stream.response_id,
            generation_id=stream.generation_id,
            reason=reason,
        )
        self.terminal = record
        stream.close()
        self.stream = None
        return record

    def assistant_context_text(self, *, separator: str = "") -> str:
        if self.stream is None:
            return ""
        return self.stream.assistant_context_text(separator=separator)
