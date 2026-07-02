"""Transport-edge mapping for operation errors."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException

from vox.operations.errors import OperationError, OperationErrorKind, classify_operation_error


HTTP_STATUS_BY_OPERATION_ERROR_KIND: dict[OperationErrorKind, int] = {
    OperationErrorKind.INVALID_ARGUMENT: 400,
    OperationErrorKind.UNPROCESSABLE_ENTITY: 422,
    OperationErrorKind.NOT_FOUND: 404,
    OperationErrorKind.CONFLICT: 409,
    OperationErrorKind.RESOURCE_EXHAUSTED: 507,
    OperationErrorKind.INTERNAL: 500,
}


def operation_error_to_http(exc: OperationError) -> HTTPException:
    kind = classify_operation_error(exc)
    return HTTPException(status_code=HTTP_STATUS_BY_OPERATION_ERROR_KIND[kind], detail=str(exc))


@contextmanager
def map_operation_errors_to_http() -> Iterator[None]:
    try:
        yield
    except OperationError as exc:
        raise operation_error_to_http(exc) from exc


@contextmanager
def map_route_errors_to_http(
    *,
    logger: logging.Logger,
    unexpected_detail: str,
    unexpected_log_message: str,
) -> Iterator[None]:
    try:
        with map_operation_errors_to_http():
            yield
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(unexpected_log_message)
        raise HTTPException(status_code=500, detail=unexpected_detail) from exc
