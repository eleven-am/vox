from __future__ import annotations

import grpc
import pytest

from vox.grpc.operation_errors import (
    abort_operation_error,
    map_operation_errors_to_grpc,
    operation_error_status,
)
from vox.operations.errors import (
    EmptyInputError,
    ModelInUseError,
    NoAudioGeneratedError,
    StoredModelNotFoundError,
)
from vox.server.operation_errors import operation_error_to_http


@pytest.mark.parametrize(
    ("error", "http_status", "grpc_status"),
    [
        (EmptyInputError(), 400, grpc.StatusCode.INVALID_ARGUMENT),
        (StoredModelNotFoundError("missing:model"), 404, grpc.StatusCode.NOT_FOUND),
        (ModelInUseError("parakeet"), 409, grpc.StatusCode.FAILED_PRECONDITION),
        (NoAudioGeneratedError(), 500, grpc.StatusCode.INTERNAL),
    ],
)
def test_operation_error_transport_mappings(error, http_status, grpc_status):
    http_error = operation_error_to_http(error)
    grpc_code, grpc_message = operation_error_status(error)

    assert http_error.status_code == http_status
    assert http_error.detail == str(error)
    assert grpc_code is grpc_status
    assert grpc_message == str(error)


@pytest.mark.asyncio
async def test_abort_operation_error_uses_shared_grpc_mapping():
    class Context:
        def __init__(self) -> None:
            self.abort_args = None

        async def abort(self, code, message):
            self.abort_args = (code, message)

    context = Context()
    error = StoredModelNotFoundError("missing:model")

    await abort_operation_error(context, error)

    assert context.abort_args == (grpc.StatusCode.NOT_FOUND, str(error))


@pytest.mark.asyncio
async def test_map_operation_errors_to_grpc_aborts_operation_errors():
    class Context:
        def __init__(self) -> None:
            self.abort_args = None

        async def abort(self, code, message):
            self.abort_args = (code, message)

    context = Context()
    error = ModelInUseError("parakeet")

    with pytest.raises(RuntimeError, match="context.abort returned without raising"):
        async with map_operation_errors_to_grpc(context):
            raise error

    assert context.abort_args == (grpc.StatusCode.FAILED_PRECONDITION, str(error))


@pytest.mark.asyncio
async def test_map_operation_errors_to_grpc_does_not_hide_unexpected_errors():
    class Context:
        async def abort(self, code, message):
            raise AssertionError("abort should not be called")

    with pytest.raises(RuntimeError, match="boom"):
        async with map_operation_errors_to_grpc(Context()):
            raise RuntimeError("boom")
