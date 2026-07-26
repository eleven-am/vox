from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vox.operations.models import PullTaskRegistry
from vox.server.rtc_registry import RtcSessionRegistry
from vox.speech_context.service import SpeechContextService


@dataclass(frozen=True, slots=True)
class AppServices:
    scheduler: Any
    registry: Any
    store: Any
    speech_context: SpeechContextService | None
    pull_tasks: PullTaskRegistry


def app_state(request_or_ws_or_app: Any) -> Any:
    app = getattr(request_or_ws_or_app, "app", request_or_ws_or_app)
    return app.state


def app_services(request_or_ws_or_app: Any) -> AppServices:
    state = app_state(request_or_ws_or_app)
    return AppServices(
        scheduler=state.scheduler,
        registry=state.registry,
        store=state.store,
        speech_context=getattr(state, "speech_context", None),
        pull_tasks=state.pull_tasks,
    )


def app_scheduler(request_or_ws_or_app: Any) -> Any:
    return app_state(request_or_ws_or_app).scheduler


def app_store(request_or_ws_or_app: Any) -> Any:
    return app_state(request_or_ws_or_app).store


def app_speech_context(request_or_ws_or_app: Any) -> SpeechContextService | None:
    return getattr(app_state(request_or_ws_or_app), "speech_context", None)


def app_rtc_registry(request_or_ws_or_app: Any) -> RtcSessionRegistry:
    state = app_state(request_or_ws_or_app)
    registry = getattr(state, "rtc_registry", None)
    if registry is None:
        registry = RtcSessionRegistry()
        state.rtc_registry = registry
    return registry


def app_pondsocket(request_or_ws_or_app: Any) -> Any | None:
    return getattr(app_state(request_or_ws_or_app), "pondsocket", None)


def set_app_pondsocket_gateway(
    request_or_ws_or_app: Any,
    *,
    pondsocket: Any,
    mount_path: str,
) -> None:
    state = app_state(request_or_ws_or_app)
    state.pondsocket = pondsocket
    state.pondsocket_mount_path = mount_path
