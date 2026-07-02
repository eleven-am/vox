"""Shared WebSocket transport helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from contextvars import Token
from dataclasses import dataclass
from typing import Any, TypeVar

from starlette.websockets import WebSocketDisconnect

from vox.core.tasks import drain_task
from vox.logging_context import bind_request_id, reset_request_id
from vox.operations.errors import OperationError, UnknownMessageTypeError
from vox.server.auth import require_ws_api_key


@dataclass(frozen=True)
class WsTextFrame:
    message: dict[str, Any]


@dataclass(frozen=True)
class WsBytesFrame:
    data: bytes


@dataclass(frozen=True)
class WsDisconnectFrame:
    pass


WsFrame = WsTextFrame | WsBytesFrame | WsDisconnectFrame
T = TypeVar("T")


async def send_ws_error(websocket: Any, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


async def send_ws_operation_error(websocket: Any, exc: OperationError) -> None:
    await send_ws_error(websocket, str(exc))


async def send_ws_unknown_message_type(websocket: Any, msg_type: object) -> None:
    await send_ws_operation_error(websocket, UnknownMessageTypeError(str(msg_type)))


async def ws_operation_result_or_error(websocket: Any, build: Callable[[], T]) -> T | None:
    try:
        return build()
    except OperationError as exc:
        await send_ws_operation_error(websocket, exc)
        return None


async def safe_send_ws_error(websocket: Any, message: str) -> None:
    with suppress(Exception):
        await send_ws_error(websocket, message)


async def safe_close_websocket(websocket: Any) -> None:
    with suppress(Exception):
        await websocket.close()


@asynccontextmanager
async def websocket_route_error_scope(
    websocket: Any,
    *,
    logger: Any,
    disconnect_log_message: str,
    error_log_message: str,
    closed_log_message: str,
) -> AsyncIterator[None]:
    try:
        yield
    except WebSocketDisconnect:
        logger.info(disconnect_log_message)
    except Exception as exc:  # noqa: BLE001
        logger.exception(error_log_message)
        await safe_send_ws_error(websocket, str(exc))
    finally:
        logger.info(closed_log_message)


@asynccontextmanager
async def websocket_session_event_scope(
    session: Any,
    emit_events: Callable[[], Any],
    *,
    drain_timeout: float | None = 5.0,
) -> AsyncIterator[None]:
    emit_task = asyncio.create_task(emit_events())
    try:
        yield
    finally:
        await drain_task(emit_task, timeout=drain_timeout)
        await session.close()


@asynccontextmanager
async def websocket_connection_scope(websocket: Any, *, require_auth: bool = True) -> AsyncIterator[None]:
    if require_auth and not await require_ws_api_key(websocket):
        raise WebSocketDisconnect(code=1008)
    await websocket.accept()
    token = bind_websocket_request_id(websocket)
    try:
        yield
    finally:
        await safe_close_websocket(websocket)
        reset_websocket_request_id(token)


def bind_websocket_request_id(websocket: Any) -> Token:
    return bind_request_id(websocket.headers.get("x-request-id"))


def reset_websocket_request_id(token: Token) -> None:
    reset_request_id(token)


async def iter_ws_json_messages(websocket: Any) -> AsyncIterator[dict[str, Any]]:
    while True:
        message = await receive_ws_json_message(websocket)
        if message is None:
            return
        yield message


async def receive_ws_json_message(websocket: Any) -> dict[str, Any] | None:
    while True:
        frame = await receive_ws_frame(websocket)
        if frame is None:
            continue
        if isinstance(frame, WsDisconnectFrame):
            return None
        if isinstance(frame, WsBytesFrame):
            await send_ws_error(websocket, "only JSON text frames are supported")
            continue
        return frame.message


async def receive_ws_frame(websocket: Any) -> WsFrame | None:
    return await parse_ws_frame(websocket, await websocket.receive())


async def parse_ws_frame(websocket: Any, raw: dict[str, Any]) -> WsFrame | None:
    if raw.get("type") == "websocket.disconnect":
        return WsDisconnectFrame()

    if "text" in raw:
        message = await parse_ws_json_text_frame(websocket, raw)
        if message is None:
            return None
        return WsTextFrame(message)

    if "bytes" in raw:
        data = raw["bytes"]
        if data:
            return WsBytesFrame(data)
        return None

    await send_ws_error(websocket, "only JSON text frames are supported")
    return None


async def parse_ws_json_text_frame(websocket: Any, raw: dict[str, Any]) -> dict[str, Any] | None:
    if "text" not in raw or raw["text"] is None:
        await send_ws_error(websocket, "only JSON text frames are supported")
        return None

    try:
        return json.loads(raw["text"])
    except json.JSONDecodeError as exc:
        await send_ws_error(websocket, f"invalid JSON: {exc}")
        return None


async def receive_required_ws_config(websocket: Any, downstream: str) -> dict[str, Any] | None:
    while True:
        message = await receive_ws_json_message(websocket)
        if message is None:
            return None
        if message.get("type") != "config":
            await send_ws_error(websocket, f"Configuration message required before {downstream}")
            continue
        return message


async def emit_ws_session_events(
    websocket: Any,
    events: AsyncIterable[Any],
    *,
    json_payload: Callable[[Any], dict[str, Any] | None],
    terminal_types: tuple[type, ...],
    binary_payload: Callable[[Any], bytes | None] | None = None,
    suppress_send_errors: bool = False,
) -> None:
    async for event in events:
        await _send_ws_session_event(
            websocket,
            event,
            json_payload=json_payload,
            binary_payload=binary_payload,
            suppress_send_errors=suppress_send_errors,
        )
        if isinstance(event, terminal_types):
            return


async def _send_ws_session_event(
    websocket: Any,
    event: Any,
    *,
    json_payload: Callable[[Any], dict[str, Any] | None],
    binary_payload: Callable[[Any], bytes | None] | None,
    suppress_send_errors: bool,
) -> None:
    sender = _send_ws_session_event_once(
        websocket,
        event,
        json_payload=json_payload,
        binary_payload=binary_payload,
    )
    if suppress_send_errors:
        with suppress(Exception):
            await sender
        return
    await sender


async def _send_ws_session_event_once(
    websocket: Any,
    event: Any,
    *,
    json_payload: Callable[[Any], dict[str, Any] | None],
    binary_payload: Callable[[Any], bytes | None] | None,
) -> None:
    if binary_payload is not None:
        data = binary_payload(event)
        if data is not None:
            await websocket.send_bytes(data)
            return

    payload = json_payload(event)
    if payload is not None:
        await websocket.send_json(payload)
