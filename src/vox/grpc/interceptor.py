"""gRPC server interceptor — request ID + per-call log line.

Sets the same `request_id_var` ContextVar the HTTP path uses, so the
interceptor, the servicer, and any ConversationSession spawned inside share
one correlation ID. Accepts an incoming ``x-request-id`` metadata key to
let callers thread their own ID through.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager

import grpc
from grpc.aio import ServerInterceptor

from vox.logging_context import bind_request_id, reset_request_id

logger = logging.getLogger("vox.grpc.request")

_METADATA_KEY = "x-request-id"


def _extract_rid(metadata: tuple | None) -> str | None:
    if not metadata:
        return None
    for key, value in metadata:
        if key.lower() == _METADATA_KEY:
            text = value.decode() if isinstance(value, bytes) else str(value)
            text = text.strip()
            if text:
                return text
    return None


@contextmanager
def _grpc_request_scope(method: str, incoming: str | None) -> Iterator[None]:
    token = bind_request_id(incoming)
    start = time.perf_counter()
    status = "OK"
    try:
        yield
    except Exception:
        status = "ERROR"
        raise
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info("%s %s (%d ms)", method, status, duration_ms)
        reset_request_id(token)


class RequestIdInterceptor(ServerInterceptor):
    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        handler = await continuation(handler_call_details)
        if handler is None:
            return handler

        method = handler_call_details.method
        incoming = _extract_rid(handler_call_details.invocation_metadata)

        async def _wrap_unary_unary(request, context):
            with _grpc_request_scope(method, incoming):
                return await handler.unary_unary(request, context)

        async def _wrap_unary_stream(request, context):
            with _grpc_request_scope(method, incoming):
                async for item in handler.unary_stream(request, context):
                    yield item

        async def _wrap_stream_unary(request_iterator, context):
            with _grpc_request_scope(method, incoming):
                return await handler.stream_unary(request_iterator, context)

        async def _wrap_stream_stream(request_iterator, context):
            with _grpc_request_scope(method, incoming):
                async for item in handler.stream_stream(request_iterator, context):
                    yield item

        if handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                _wrap_unary_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                _wrap_unary_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                _wrap_stream_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(
                _wrap_stream_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return handler
