from __future__ import annotations

import logging

import grpc
import pytest

from vox.grpc.operation_errors import (
    GRPC_STATUS_BY_OPERATION_ERROR_KIND,
    abort_operation_error,
    map_operation_errors_to_grpc,
    map_route_errors_to_grpc,
    operation_error_status,
)
from vox.operations.errors import (
    EmptyInputError,
    ModelInUseError,
    NoAudioGeneratedError,
    OperationErrorKind,
    StoredModelNotFoundError,
)
from vox.server.operation_errors import HTTP_STATUS_BY_OPERATION_ERROR_KIND, operation_error_to_http
from vox.server.auth import (
    API_KEY_HEADER,
    API_KEY_QUERY_PARAM,
    AUTHORIZATION_HEADER,
    MISSING_OR_INVALID_API_KEY,
    authorize_api_key_connection,
    bearer_token_from_authorization,
    extract_api_key_from_parts,
)


def test_transport_mappings_cover_every_operation_error_kind():
    assert set(HTTP_STATUS_BY_OPERATION_ERROR_KIND) == set(OperationErrorKind)
    assert set(GRPC_STATUS_BY_OPERATION_ERROR_KIND) == set(OperationErrorKind)


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


@pytest.mark.parametrize(
    ("raw", "token"),
    [
        (None, None),
        ("", None),
        ("Basic abc", None),
        ("Bearer rtc_token", "rtc_token"),
        ("  bearer   rtc_token  ", "rtc_token"),
        ("Bearer   ", None),
    ],
)
def test_bearer_token_from_authorization_header(raw, token):
    assert bearer_token_from_authorization(raw) == token


@pytest.mark.parametrize(
    ("headers", "query", "api_key"),
    [
        ({"authorization": "Bearer bearer-key", "x-api-key": "header-key"}, {"api_key": "query-key"}, "bearer-key"),
        ({"authorization": "Basic ignored", "x-api-key": " header-key "}, {"api_key": "query-key"}, "header-key"),
        ({}, {"api_key": " query-key "}, "query-key"),
        ({"authorization": "Bearer   ", "x-api-key": "   "}, {"api_key": "   "}, None),
    ],
)
def test_extract_api_key_from_parts_uses_shared_precedence(headers, query, api_key):
    assert extract_api_key_from_parts(headers, query) == api_key


def test_api_key_extraction_field_names_are_explicit_policy():
    assert AUTHORIZATION_HEADER == "authorization"
    assert API_KEY_HEADER == "x-api-key"
    assert API_KEY_QUERY_PARAM == "api_key"


def test_authorize_api_key_connection_accepts_when_auth_is_disabled(monkeypatch):
    monkeypatch.delenv("VOX_API_KEY", raising=False)

    class Context:
        headers = {}
        query = {}

        def __init__(self) -> None:
            self.accepted = False
            self.decline_args = None

        def accept(self):
            self.accepted = True

        def decline(self, *args):
            self.decline_args = args

    context = Context()

    authorize_api_key_connection(context)

    assert context.accepted is True
    assert context.decline_args is None


def test_authorize_api_key_connection_declines_invalid_key(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")

    class Context:
        headers = {API_KEY_HEADER: "wrong"}
        query = {}

        def __init__(self) -> None:
            self.accepted = False
            self.decline_args = None

        def accept(self):
            self.accepted = True

        def decline(self, *args):
            self.decline_args = args

    context = Context()

    authorize_api_key_connection(context)

    assert context.accepted is False
    assert context.decline_args == (401, MISSING_OR_INVALID_API_KEY)


def test_authorize_api_key_connection_accepts_valid_key(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")

    class Context:
        headers = {AUTHORIZATION_HEADER: "Bearer secret"}
        query = {API_KEY_QUERY_PARAM: "wrong"}

        def __init__(self) -> None:
            self.accepted = False
            self.decline_args = None

        def accept(self):
            self.accepted = True

        def decline(self, *args):
            self.decline_args = args

    context = Context()

    authorize_api_key_connection(context)

    assert context.accepted is True
    assert context.decline_args is None


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


@pytest.mark.asyncio
async def test_map_route_errors_to_grpc_preserves_operation_mapping():
    class Context:
        def __init__(self) -> None:
            self.abort_args = None

        async def abort(self, code, message):
            self.abort_args = (code, message)

    context = Context()

    with pytest.raises(RuntimeError, match="context.abort returned without raising"):
        async with map_route_errors_to_grpc(
            context,
            logger=logging.getLogger("tests.transport_operation_errors"),
            unexpected_message="internal",
            unexpected_log_message="unexpected route failure",
        ):
            raise ModelInUseError("parakeet")

    assert context.abort_args == (grpc.StatusCode.FAILED_PRECONDITION, str(ModelInUseError("parakeet")))


@pytest.mark.asyncio
async def test_map_route_errors_to_grpc_converts_unexpected_errors(caplog):
    class Context:
        def __init__(self) -> None:
            self.abort_args = None

        async def abort(self, code, message):
            self.abort_args = (code, message)

    context = Context()

    with pytest.raises(RuntimeError, match="context.abort returned without raising"):
        async with map_route_errors_to_grpc(
            context,
            logger=logging.getLogger("tests.transport_operation_errors"),
            unexpected_message="internal failure",
            unexpected_log_message="unexpected route failure",
        ):
            raise RuntimeError("boom")

    assert context.abort_args == (grpc.StatusCode.INTERNAL, "internal failure")
    assert "unexpected route failure" in caplog.text
