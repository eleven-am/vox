"""Shared WebSocket transport helpers."""

from __future__ import annotations

from contextlib import suppress
from typing import Any


async def send_ws_error(websocket: Any, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


async def safe_send_ws_error(websocket: Any, message: str) -> None:
    with suppress(Exception):
        await send_ws_error(websocket, message)
