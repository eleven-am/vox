"""Shared WebSocket transport helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from vox.operations.errors import OperationError


async def send_ws_error(websocket: Any, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


async def send_ws_operation_error(websocket: Any, exc: OperationError) -> None:
    await send_ws_error(websocket, str(exc))


async def safe_send_ws_error(websocket: Any, message: str) -> None:
    with suppress(Exception):
        await send_ws_error(websocket, message)


async def iter_ws_json_messages(websocket: Any) -> AsyncIterator[dict[str, Any]]:
    while True:
        raw = await websocket.receive()
        if raw.get("type") == "websocket.disconnect":
            return
        if "text" not in raw or raw["text"] is None:
            await send_ws_error(websocket, "only JSON text frames are supported")
            continue

        try:
            message = json.loads(raw["text"])
        except json.JSONDecodeError as exc:
            await send_ws_error(websocket, f"invalid JSON: {exc}")
            continue

        yield message
