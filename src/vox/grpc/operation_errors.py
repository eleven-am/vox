"""gRPC transport-edge mapping for operation errors."""

from __future__ import annotations

import grpc

from vox.operations.errors import OperationError, OperationErrorKind, classify_operation_error


def operation_error_status(exc: OperationError) -> tuple[grpc.StatusCode, str]:
    kind = classify_operation_error(exc)
    if kind is OperationErrorKind.INVALID_ARGUMENT:
        return grpc.StatusCode.INVALID_ARGUMENT, str(exc)
    if kind is OperationErrorKind.UNPROCESSABLE_ENTITY:
        return grpc.StatusCode.INVALID_ARGUMENT, str(exc)
    if kind is OperationErrorKind.NOT_FOUND:
        return grpc.StatusCode.NOT_FOUND, str(exc)
    if kind is OperationErrorKind.CONFLICT:
        return grpc.StatusCode.FAILED_PRECONDITION, str(exc)
    if kind is OperationErrorKind.RESOURCE_EXHAUSTED:
        return grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc)
    return grpc.StatusCode.INTERNAL, str(exc)


async def abort_operation_error(context, exc: OperationError) -> None:
    code, message = operation_error_status(exc)
    await context.abort(code, message)
