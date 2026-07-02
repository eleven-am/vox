from __future__ import annotations

from dataclasses import dataclass

from vox.conversation.response_stream import ResponseStream


@dataclass
class ConversationResponseLifecycle:
    """Tracks the active assistant response stream and response identifiers."""

    counter: int = 0
    stream: ResponseStream | None = None
    active_response_id: str | None = None
    last_cancelled_response_id: str | None = None

    def open_uncommitted_stream(self) -> ResponseStream | None:
        if self.stream is not None and not self.stream.committed:
            return self.stream
        return None

    def start_stream(self, *, allow_interruptions: bool = True) -> ResponseStream:
        self.counter += 1
        response_id = f"resp_{self.counter}"
        stream = ResponseStream.create(
            response_id=response_id,
            allow_interruptions=allow_interruptions,
        )
        self.stream = stream
        self.active_response_id = response_id
        self.last_cancelled_response_id = None
        return stream

    def finish_stream_if_current(self, stream: ResponseStream) -> None:
        if self.stream is stream:
            self.stream = None
        if self.active_response_id == stream.response_id:
            self.active_response_id = None

    def clear_finished_stream_if_current(self, stream: ResponseStream) -> None:
        if not stream.pending_done:
            self.finish_stream_if_current(stream)

    def remember_cancelled_response(self) -> str | None:
        response_id = self.stream.response_id if self.stream is not None else self.active_response_id
        self.last_cancelled_response_id = response_id
        return response_id

    def clear_active_response(self, stream: ResponseStream | None) -> None:
        if stream is not None and self.active_response_id == stream.response_id:
            self.active_response_id = None
        self.stream = None

    def active_or_cancelled_response_id(self) -> str | None:
        return self.active_response_id or self.last_cancelled_response_id

    def assistant_context_text(self, *, separator: str = "") -> str:
        if self.stream is None:
            return ""
        return self.stream.assistant_context_text(separator=separator)
