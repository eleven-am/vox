from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from vox.server.app import lifespan


def _make_app(*, grpc_port: int | None) -> FastAPI:
    app = FastAPI()
    app.state.scheduler = MagicMock(start=AsyncMock(), stop=AsyncMock())
    app.state.store = MagicMock(root=Path("/path/that/does/not/exist"))
    app.state.registry = MagicMock()
    app.state.grpc_port = grpc_port
    app.state.rtc_registry = MagicMock(close_all=AsyncMock())
    return app


@pytest.mark.asyncio
async def test_lifespan_stops_scheduler_if_grpc_start_fails():
    app = _make_app(grpc_port=9090)
    grpc_server_module = ModuleType("vox.grpc.server")
    grpc_server_module.start_grpc_server = AsyncMock(side_effect=RuntimeError("grpc startup failed"))

    with (
        patch.dict(sys.modules, {"vox.grpc.server": grpc_server_module}),
        pytest.raises(RuntimeError, match="grpc startup failed"),
    ):
        async with lifespan(app):
            pass

    app.state.scheduler.start.assert_awaited_once()
    app.state.scheduler.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_prunes_stale_temp_directories_before_scheduler_start(tmp_path: Path):
    app = _make_app(grpc_port=None)
    app.state.store.root = tmp_path

    with patch("vox.server.app.prune_stale_temp_dirs") as prune:
        async with lifespan(app):
            pass

    prune.assert_called_once_with(tmp_path / "tmp")
    app.state.scheduler.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_preloads_core_ner(_mock_application_ner_preload):
    app = _make_app(grpc_port=None)

    async with lifespan(app):
        pass

    _mock_application_ner_preload.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_lifespan_preloads_core_native_modules_before_ner(
    _mock_application_ner_preload,
):
    app = _make_app(grpc_port=None)
    preload_order: list[str] = []

    def preload_native() -> None:
        preload_order.append("native")

    async def preload_ner() -> None:
        preload_order.append("ner")

    _mock_application_ner_preload.side_effect = preload_ner
    with patch(
        "vox.server.app.preload_core_native_modules",
        side_effect=preload_native,
    ):
        async with lifespan(app):
            pass

    assert preload_order == ["native", "ner"]


@pytest.mark.asyncio
async def test_lifespan_stops_grpc_server_and_scheduler_on_shutdown():
    app = _make_app(grpc_port=9090)
    grpc_server = MagicMock(stop=AsyncMock())
    grpc_server_module = ModuleType("vox.grpc.server")
    grpc_server_module.start_grpc_server = AsyncMock(return_value=grpc_server)

    with patch.dict(sys.modules, {"vox.grpc.server": grpc_server_module}):
        async with lifespan(app):
            pass

    app.state.scheduler.start.assert_awaited_once()
    grpc_server.stop.assert_awaited_once_with(grace=5)
    app.state.scheduler.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_closes_pondsocket_before_stopping_scheduler():
    app = _make_app(grpc_port=None)
    shutdown_order: list[str] = []

    async def close_pondsocket() -> None:
        shutdown_order.append("pondsocket")

    async def stop_scheduler() -> None:
        shutdown_order.append("scheduler")

    app.state.pondsocket = MagicMock(close=AsyncMock(side_effect=close_pondsocket))
    app.state.scheduler.stop.side_effect = stop_scheduler

    async with lifespan(app):
        pass

    app.state.pondsocket.close.assert_awaited_once()
    app.state.scheduler.stop.assert_awaited_once()
    assert shutdown_order == ["pondsocket", "scheduler"]


@pytest.mark.asyncio
async def test_lifespan_closes_all_rtc_sessions_before_stopping_scheduler():
    app = _make_app(grpc_port=None)
    shutdown_order: list[str] = []

    async def close_rtc() -> None:
        shutdown_order.append("rtc")

    async def stop_scheduler() -> None:
        shutdown_order.append("scheduler")

    app.state.rtc_registry.close_all.side_effect = close_rtc
    app.state.scheduler.stop.side_effect = stop_scheduler

    async with lifespan(app):
        pass

    app.state.rtc_registry.close_all.assert_awaited_once()
    assert shutdown_order == ["rtc", "scheduler"]
