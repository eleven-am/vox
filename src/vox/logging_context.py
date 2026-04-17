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


class RequestIdFilter(logging.Filter):
    """Injects the active request ID onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True
