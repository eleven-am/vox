"""gRPC transport-edge mapping for operation errors."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc

from vox.operations.errors import OperationError, OperationErrorKind, classify_operation_error

GRPC_STATUS_BY_OPERATION_ERROR_KIND: dict[OperationErrorKind, grpc.StatusCode] = {
    OperationErrorKind.UNAUTHENTICATED: grpc.StatusCode.UNAUTHENTICATED,
    OperationErrorKind.INVALID_ARGUMENT: grpc.StatusCode.INVALID_ARGUMENT,
    OperationErrorKind.UNPROCESSABLE_ENTITY: grpc.StatusCode.INVALID_ARGUMENT,
    OperationErrorKind.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    OperationErrorKind.CONFLICT: grpc.StatusCode.FAILED_PRECONDITION,
    OperationErrorKind.RESOURCE_EXHAUSTED: grpc.StatusCode.RESOURCE_EXHAUSTED,
    OperationErrorKind.INTERNAL: grpc.StatusCode.INTERNAL,
}


def operation_error_status(exc: OperationError) -> tuple[grpc.StatusCode, str]:
    kind = classify_operation_error(exc)
    return GRPC_STATUS_BY_OPERATION_ERROR_KIND[kind], str(exc)


async def abort_operation_error(context, exc: OperationError) -> None:
    code, message = operation_error_status(exc)
    await context.abort(code, message)


@asynccontextmanager
async def map_operation_errors_to_grpc(context) -> AsyncIterator[None]:
    try:
        yield
    except OperationError as exc:
        await abort_operation_error(context, exc)
        raise RuntimeError("gRPC context.abort returned without raising") from exc


@asynccontextmanager
async def map_route_errors_to_grpc(
    context,
    *,
    logger: logging.Logger,
    unexpected_message: str,
    unexpected_log_message: str,
) -> AsyncIterator[None]:
    try:
        yield
    except OperationError as exc:
        await abort_operation_error(context, exc)
        raise RuntimeError("gRPC context.abort returned without raising") from exc
    except Exception as exc:
        logger.exception(unexpected_log_message)
        await context.abort(grpc.StatusCode.INTERNAL, unexpected_message)
        raise RuntimeError("gRPC context.abort returned without raising") from exc
