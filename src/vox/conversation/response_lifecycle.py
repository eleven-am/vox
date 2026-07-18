from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from vox.conversation.response_stream import ResponseStream


class ResponsePhase(StrEnum):
    NONE = "none"
    STARTED = "started"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    DONE = "done"


_LIVE_PHASES = frozenset({ResponsePhase.STARTED, ResponsePhase.COMMITTED})


@dataclass
class ConversationResponseLifecycle:
    """Tracks the active assistant response stream and response identifiers."""

    counter: int = 0
    stream: ResponseStream | None = None
    active_response_id: str | None = None
    last_cancelled_response_id: str | None = None
    phase: ResponsePhase = field(default=ResponsePhase.NONE)

    @property
    def response_active(self) -> bool:
        return self.stream is not None and self.phase in _LIVE_PHASES

    def open_uncommitted_stream(self) -> ResponseStream | None:
        if self.stream is not None and not self.stream.committed and self.phase is ResponsePhase.STARTED:
            return self.stream
        return None

    def appendable_stream(self, expected_response_id: str) -> ResponseStream | None:
        stream = self.open_uncommitted_stream()
        if stream is None or stream.response_id != expected_response_id:
            return None
        return stream

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
        self.phase = ResponsePhase.STARTED
        return stream

    def commit_stream(self, stream: ResponseStream) -> bool:
        if self.stream is not stream or self.phase is not ResponsePhase.STARTED:
            return False
        if not stream.mark_committed():
            return False
        self.phase = ResponsePhase.COMMITTED
        return True

    def finish_stream_if_current(self, stream: ResponseStream) -> None:
        stream.close()
        if self.stream is stream:
            self.stream = None
            self.phase = ResponsePhase.DONE
        if self.active_response_id == stream.response_id:
            self.active_response_id = None

    def clear_finished_stream_if_current(self, stream: ResponseStream) -> None:
        if not stream.pending_done:
            self.finish_stream_if_current(stream)

    def fail_stream_if_current(self, stream: ResponseStream) -> None:
        stream.close()
        if self.stream is stream:
            self.stream = None
            self.phase = ResponsePhase.CANCELLED
            self.last_cancelled_response_id = stream.response_id
        if self.active_response_id == stream.response_id:
            self.active_response_id = None

    def remember_cancelled_response(self) -> str | None:
        response_id = self.stream.response_id if self.stream is not None else self.active_response_id
        if response_id is None:
            return None
        self.last_cancelled_response_id = response_id
        self.phase = ResponsePhase.CANCELLED
        return response_id

    def clear_active_response(self, stream: ResponseStream | None) -> None:
        if stream is not None:
            stream.close()
        if self.stream is not None:
            self.stream.close()
        if stream is not None and self.active_response_id == stream.response_id:
            self.active_response_id = None
        self.stream = None

    def active_or_cancelled_response_id(self) -> str | None:
        return self.active_response_id or self.last_cancelled_response_id

    def assistant_context_text(self, *, separator: str = "") -> str:
        if self.stream is None:
            return ""
        return self.stream.assistant_context_text(separator=separator)
