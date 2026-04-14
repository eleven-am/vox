from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from vox.server.app import lifespan


def _make_app(*, grpc_port: int | None) -> FastAPI:
    app = FastAPI()
    app.state.scheduler = MagicMock(start=AsyncMock(), stop=AsyncMock())
    app.state.store = MagicMock()
    app.state.registry = MagicMock()
    app.state.grpc_port = grpc_port
    return app


@pytest.mark.asyncio
async def test_lifespan_stops_scheduler_if_grpc_start_fails():
    app = _make_app(grpc_port=9090)

    with (
        patch(
            "vox.grpc.server.start_grpc_server",
            AsyncMock(side_effect=RuntimeError("grpc startup failed")),
        ),
        pytest.raises(RuntimeError, match="grpc startup failed"),
    ):
        async with lifespan(app):
            pass

    app.state.scheduler.start.assert_awaited_once()
    app.state.scheduler.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_stops_grpc_server_and_scheduler_on_shutdown():
    app = _make_app(grpc_port=9090)
    grpc_server = MagicMock(stop=AsyncMock())

    with patch(
        "vox.grpc.server.start_grpc_server",
        AsyncMock(return_value=grpc_server),
    ):
        async with lifespan(app):
            pass

    app.state.scheduler.start.assert_awaited_once()
    grpc_server.stop.assert_awaited_once_with(grace=5)
    app.state.scheduler.stop.assert_awaited_once()
