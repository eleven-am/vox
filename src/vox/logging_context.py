"""Request-scoped correlation ID for log records.

`request_id_var` is a `ContextVar` set once per inbound connection / call and
inherited by every task spawned from it. `RequestIdFilter` pulls the current
value onto each `LogRecord` so formatters can reference `%(request_id)s` even
when no ID is active (defaults to ``"-"``).
"""

from __future__ import annotations

import contextvars
import logging
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "vox_request_id", default="-",
)


def new_request_id() -> str:
    """Short, log-friendly correlation ID (12 hex chars of a UUID4)."""
    return uuid.uuid4().hex[:12]


def current_request_id() -> str:
    return request_id_var.get()


def request_id_from_incoming(incoming: object | None) -> str:
    if incoming is None:
        return new_request_id()
    text = incoming.decode() if isinstance(incoming, bytes) else str(incoming)
    text = text.strip()
    return text or new_request_id()


def bind_request_id(incoming: object | None = None) -> contextvars.Token[str]:
    return request_id_var.set(request_id_from_incoming(incoming))


def reset_request_id(token: contextvars.Token[str]) -> None:
    request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    """Injects the active request ID onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True
