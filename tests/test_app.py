from __future__ import annotations

import asyncio
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

import vox.server.app as app_module
from vox.core.store import BlobStore
from vox.operations.models import PullTaskRegistry
from vox.server.app import lifespan


def test_grpc_listen_address_uses_requested_host():
    from vox.grpc.server import grpc_listen_address

    assert grpc_listen_address("127.0.0.1", 9090) == "127.0.0.1:9090"
    assert grpc_listen_address("0.0.0.0", 9090) == "0.0.0.0:9090"
    assert grpc_listen_address("::", 9090) == "[::]:9090"
    assert grpc_listen_address("2001:db8::1", 9090) == "[2001:db8::1]:9090"


def _make_app(*, grpc_port: int | None) -> FastAPI:
    app = FastAPI()
    app.state.scheduler = MagicMock(start=AsyncMock(), stop=AsyncMock())
    app.state.store = MagicMock(root=Path("/path/that/does/not/exist"))
    app.state.registry = MagicMock()
    app.state.grpc_port = grpc_port
    app.state.bind_host = "127.0.0.1"
    app.state.max_upload_bytes = 1234
    app.state.rtc_registry = MagicMock(close_all=AsyncMock())
    app.state.speech_context = MagicMock(close=AsyncMock())
    app.state.pull_tasks = MagicMock(close=AsyncMock())
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
async def test_lifespan_rejects_grpc_start_without_speech_context():
    app = _make_app(grpc_port=9090)
    del app.state.speech_context

    with pytest.raises(RuntimeError, match="speech-context"):
        async with lifespan(app):
            pass

    app.state.scheduler.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_prunes_stale_temp_directories_before_scheduler_start(tmp_path: Path):
    app = _make_app(grpc_port=None)
    app.state.store.root = tmp_path
    scratch = tmp_path / "scratch"
    ambient = tmp_path / "operating-system-temp"

    with (
        patch.dict(
            "os.environ",
            {
                "TMPDIR": str(ambient),
                "VOX_TEMP_ROOT": str(scratch),
            },
        ),
        patch("vox.server.app.prune_stale_temp_dirs") as prune,
    ):
        async with lifespan(app):
            pass

    prune.assert_called_once_with(scratch)
    app.state.scheduler.start.assert_awaited_once()


def test_create_app_prunes_stale_adapter_and_runtime_install_directories(tmp_path: Path):
    with patch("vox.server.app.prune_stale_install_directories") as prune:
        app_module.create_app(vox_home=tmp_path)

    assert prune.call_count == 2
    prune.assert_any_call(tmp_path / "adapters")
    prune.assert_any_call(tmp_path / "runtime")


def test_create_app_recovers_pull_transactions_before_registry_construction(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def recover(_store) -> int:
        calls.append("recover")
        return 0

    def registry(_store):
        calls.append("registry")
        return MagicMock()

    monkeypatch.setattr(app_module, "recover_pull_transactions", recover, raising=False)
    monkeypatch.setattr(app_module, "ModelRegistry", registry)
    monkeypatch.setattr(app_module, "Scheduler", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(app_module, "SpeechContextService", MagicMock(return_value=MagicMock()))

    app_module.create_app(vox_home=tmp_path)

    assert calls[:2] == ["recover", "registry"]


def test_create_app_collects_unreferenced_blobs_before_registry_construction(
    monkeypatch,
    tmp_path: Path,
):
    store = BlobStore(root=tmp_path)
    digest = store.write_blob(BytesIO(b"orphaned-before-journal"))

    def registry(recovered_store):
        assert recovered_store.has_blob(digest) is False
        return MagicMock()

    monkeypatch.setattr(app_module, "ModelRegistry", registry)
    monkeypatch.setattr(app_module, "Scheduler", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        app_module,
        "SpeechContextService",
        MagicMock(return_value=MagicMock()),
    )

    app_module.create_app(vox_home=tmp_path)


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
    grpc_server_module.start_grpc_server.assert_awaited_once()
    assert grpc_server_module.start_grpc_server.await_args.kwargs == {
        "pull_tasks": app.state.pull_tasks,
        "host": "127.0.0.1",
        "port": 9090,
        "max_message_bytes": 1234,
    }
    grpc_server.stop.assert_awaited_once_with(grace=5)
    app.state.scheduler.stop.assert_awaited_once()


def test_create_app_owns_model_pull_tasks(tmp_path: Path):
    app = app_module.create_app(vox_home=tmp_path)

    assert isinstance(app.state.pull_tasks, PullTaskRegistry)


@pytest.mark.asyncio
async def test_lifespan_closes_model_pull_tasks_with_the_shared_deadline():
    app = _make_app(grpc_port=None)

    async with lifespan(app):
        pass

    app.state.pull_tasks.close.assert_awaited_once()
    deadline = app.state.pull_tasks.close.await_args.kwargs["deadline"]
    assert isinstance(deadline, float)


@pytest.mark.asyncio
async def test_lifespan_closes_pondsocket_before_stopping_scheduler():
    app = _make_app(grpc_port=None)
    shutdown_order: list[str] = []

    async def close_pondsocket() -> None:
        shutdown_order.append("pondsocket")

    async def stop_scheduler(**_kwargs) -> None:
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

    async def stop_scheduler(**_kwargs) -> None:
        shutdown_order.append("scheduler")

    app.state.rtc_registry.close_all.side_effect = close_rtc
    app.state.scheduler.stop.side_effect = stop_scheduler

    async with lifespan(app):
        pass

    app.state.rtc_registry.close_all.assert_awaited_once()
    assert shutdown_order == ["rtc", "scheduler"]


@pytest.mark.asyncio
async def test_lifespan_uses_one_deadline_for_all_shutdown_owners():
    app = _make_app(grpc_port=None)
    app.state.shutdown_timeout_seconds = 0.05
    started: set[str] = set()

    async def wait_owner(owner: str, **_kwargs) -> None:
        started.add(owner)
        await asyncio.sleep(0.15)

    async def close_pondsocket() -> None:
        await wait_owner("pondsocket")

    async def close_rtc() -> None:
        await wait_owner("rtc")

    async def close_speech_context() -> None:
        await wait_owner("speech_context")

    async def stop_scheduler(**kwargs) -> None:
        await wait_owner("scheduler", **kwargs)

    app.state.pondsocket = MagicMock(close=AsyncMock(side_effect=close_pondsocket))
    app.state.rtc_registry.close_all.side_effect = close_rtc
    app.state.speech_context = MagicMock(close=AsyncMock(side_effect=close_speech_context))
    app.state.scheduler.stop.side_effect = stop_scheduler

    with pytest.raises(ExceptionGroup, match="shutdown"):
        async with asyncio.timeout(0.3):
            async with lifespan(app):
                pass

    assert started == {"pondsocket", "rtc", "speech_context", "scheduler"}


@pytest.mark.asyncio
async def test_shutdown_keeps_cancellation_resistant_owner_until_physical_completion():
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def resistant_close() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    rtc_registry = MagicMock(close_all=AsyncMock(side_effect=resistant_close))
    scheduler = MagicMock(stop=AsyncMock())

    with pytest.raises(ExceptionGroup, match="shutdown"):
        await app_module._shutdown_services(
            grpc_server=None,
            pond=None,
            rtc_registry=rtc_registry,
            speech_context=None,
            pull_tasks=MagicMock(close=AsyncMock()),
            scheduler=scheduler,
            timeout=0.01,
        )

    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
    assert len(app_module._SHUTDOWN_TASKS) == 1

    release.set()
    for _ in range(100):
        if not app_module._SHUTDOWN_TASKS:
            break
        await asyncio.sleep(0.001)

    assert not app_module._SHUTDOWN_TASKS
