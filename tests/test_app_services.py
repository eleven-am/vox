from __future__ import annotations

from types import SimpleNamespace

from vox.server.app_services import (
    app_pondsocket,
    app_rtc_registry,
    app_scheduler,
    app_services,
    app_state,
    app_store,
    set_app_pondsocket_gateway,
)
from vox.server.rtc_registry import RtcSessionRegistry


def _request_with_state(**state):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


def _app_with_state(**state):
    return SimpleNamespace(state=SimpleNamespace(**state))


def test_app_services_reads_canonical_server_state_fields():
    scheduler = object()
    registry = object()
    store = object()
    pull_tasks = object()

    services = app_services(
        _request_with_state(
            scheduler=scheduler,
            registry=registry,
            store=store,
            pull_tasks=pull_tasks,
        )
    )

    assert services.scheduler is scheduler
    assert services.registry is registry
    assert services.store is store
    assert services.pull_tasks is pull_tasks


def test_app_services_accepts_app_object_for_installer_paths():
    scheduler = object()
    registry = object()
    store = object()
    pull_tasks = object()
    app = _app_with_state(
        scheduler=scheduler,
        registry=registry,
        store=store,
        pull_tasks=pull_tasks,
    )

    services = app_services(app)

    assert services.scheduler is scheduler
    assert services.registry is registry
    assert services.store is store
    assert services.pull_tasks is pull_tasks
    assert app_state(app) is app.state


def test_single_service_helpers_read_canonical_server_state_fields():
    scheduler = object()
    store = object()
    request = _request_with_state(scheduler=scheduler, store=store)

    assert app_scheduler(request) is scheduler
    assert app_store(request) is store


def test_app_rtc_registry_returns_existing_registry():
    registry = RtcSessionRegistry()
    request = _request_with_state(rtc_registry=registry)

    assert app_rtc_registry(request) is registry


def test_app_rtc_registry_lazily_creates_registry_for_app_object():
    app = _app_with_state()

    registry = app_rtc_registry(app)

    assert isinstance(registry, RtcSessionRegistry)
    assert app.state.rtc_registry is registry


def test_app_pondsocket_returns_optional_gateway_instance():
    pond = object()

    assert app_pondsocket(_request_with_state(pondsocket=pond)) is pond
    assert app_pondsocket(_request_with_state()) is None


def test_set_app_pondsocket_gateway_registers_gateway_state():
    app = _app_with_state()
    pond = object()

    set_app_pondsocket_gateway(app, pondsocket=pond, mount_path="/v1/socket")

    assert app_pondsocket(app) is pond
    assert app.state.pondsocket_mount_path == "/v1/socket"
