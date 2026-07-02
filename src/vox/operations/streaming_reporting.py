from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from vox.operations.errors import NoDefaultModelError, OperationError


class OperationErrorReporter(Protocol):
    async def report_operation_error(self, exc: OperationError) -> None: ...


class StreamingErrorReporter(Protocol):
    async def report_error(self, message: str) -> None: ...


def streaming_operation_error_message(exc: OperationError) -> str:
    if isinstance(exc, NoDefaultModelError):
        model_type = exc.model_type.upper()
        return f"No {model_type} model specified and no default {model_type} model available"
    return str(exc)


async def run_or_report_operation_error(
    action: Callable[[], Awaitable[None]],
    report_operation_error: Callable[[OperationError], Awaitable[None]],
) -> bool:
    try:
        await action()
    except OperationError as exc:
        await report_operation_error(exc)
        return False
    return True


class StreamingOperationErrorReporter:
    async def report_operation_error(self: StreamingErrorReporter, exc: OperationError) -> None:
        await self.report_error(streaming_operation_error_message(exc))

    async def run_or_report_operation_error(
        self: OperationErrorReporter,
        action: Callable[[], Awaitable[None]],
    ) -> bool:
        return await run_or_report_operation_error(action, self.report_operation_error)
