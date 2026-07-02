"""Transport-edge mapping for operation errors."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException

from vox.operations.errors import OperationError, OperationErrorKind, classify_operation_error


def operation_error_to_http(exc: OperationError) -> HTTPException:
    kind = classify_operation_error(exc)
    if kind is OperationErrorKind.INVALID_ARGUMENT:
        return HTTPException(status_code=400, detail=str(exc))
    if kind is OperationErrorKind.UNPROCESSABLE_ENTITY:
        return HTTPException(status_code=422, detail=str(exc))
    if kind is OperationErrorKind.NOT_FOUND:
        return HTTPException(status_code=404, detail=str(exc))
    if kind is OperationErrorKind.CONFLICT:
        return HTTPException(status_code=409, detail=str(exc))
    if kind is OperationErrorKind.RESOURCE_EXHAUSTED:
        return HTTPException(status_code=507, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@contextmanager
def map_operation_errors_to_http() -> Iterator[None]:
    try:
        yield
    except OperationError as exc:
        raise operation_error_to_http(exc) from exc
