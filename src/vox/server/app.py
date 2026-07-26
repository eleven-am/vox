from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from vox.core.atomic_install import prune_stale_install_directories
from vox.core.hf_runtime import configure_hf_runtime
from vox.core.pull_transaction import recover_pull_transactions
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.core.temp_storage import prune_stale_temp_dirs, vox_temp_root
from vox.logging_config import configure_logging
from vox.operations.models import PullTaskRegistry
from vox.server.app_services import app_pondsocket, app_rtc_registry, app_services
from vox.server.middleware import ApiKeyAuthMiddleware, RequestIdMiddleware
from vox.server.preload import (
    merged_preload_models,
    preload_core_native_modules,
    preload_models,
    preload_ner,
    preload_turn_detector,
    preload_vad,
    should_preload_vad,
)
from vox.server.rtc_registry import RtcSessionRegistry
from vox.server.uploads import UploadSizeLimitMiddleware, configured_max_upload_bytes
from vox.speech_context.service import SpeechContextService

logger = logging.getLogger(__name__)

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_SHUTDOWN_TASKS: set[asyncio.Task[Any]] = set()


def _parse_cors_origins(value: str | None) -> list[str]:
    if not value:
        return []
    return [origin.strip() for origin in value.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    grpc_server = None
    services = app_services(app)
    rtc_registry = app_rtc_registry(app)
    prune_stale_temp_dirs(vox_temp_root())
    await services.scheduler.start()
    try:
        preload_core_native_modules()
        await preload_ner()

        merged = merged_preload_models(
            list(getattr(app.state, "preload_models", [])),
            os.environ.get("VOX_PRELOAD"),
        )
        if merged:
            await preload_models(services.scheduler, merged)

        if should_preload_vad(getattr(app.state, "preload_vad", False)):
            await preload_vad()

        turn_detector = getattr(app.state, "preload_turn_detector", None)
        if turn_detector:
            await preload_turn_detector(services.scheduler, turn_detector)

        grpc_port = getattr(app.state, "grpc_port", None)
        if grpc_port:
            from vox.grpc.server import start_grpc_server

            speech_context = services.speech_context
            if speech_context is None:
                raise RuntimeError("gRPC startup requires the speech-context service")
            grpc_server = await start_grpc_server(
                services.store,
                services.registry,
                services.scheduler,
                rtc_registry,
                speech_context,
                pull_tasks=services.pull_tasks,
                host=getattr(app.state, "bind_host", "0.0.0.0"),
                port=grpc_port,
                max_message_bytes=getattr(app.state, "max_upload_bytes", None),
            )

        logger.info("Vox server started")
        yield
    finally:
        timeout = float(
            getattr(
                app.state,
                "shutdown_timeout_seconds",
                DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
            )
        )
        await _shutdown_services(
            grpc_server=grpc_server,
            pond=app_pondsocket(app),
            rtc_registry=rtc_registry,
            speech_context=services.speech_context,
            pull_tasks=services.pull_tasks,
            scheduler=services.scheduler,
            timeout=max(0.1, timeout),
        )
        logger.info("Vox server stopped")


async def _run_shutdown_owner(name: str, operation: Awaitable[Any]) -> None:
    await operation
    logger.info("%s stopped", name)


def _consume_shutdown_task(task: asyncio.Task[Any]) -> None:
    _SHUTDOWN_TASKS.discard(task)
    if task.done() and not task.cancelled():
        with suppress(Exception):
            task.exception()


async def _shutdown_services(
    *,
    grpc_server: Any,
    pond: Any,
    rtc_registry: Any,
    speech_context: Any,
    pull_tasks: PullTaskRegistry,
    scheduler: Any,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    operations = []
    if grpc_server is not None:
        operations.append(("gRPC server", grpc_server.stop(grace=5)))
    if pond is not None:
        operations.append(("PondSocket server", pond.close()))
    operations.append(("RTC sessions", rtc_registry.close_all()))
    if speech_context is not None:
        operations.append(("Speech context workers", speech_context.close()))
    operations.append(("Model pulls", pull_tasks.close(deadline=deadline)))
    operations.append(("Scheduler", scheduler.stop(deadline=deadline)))
    tasks = {}
    for name, operation in operations:
        task = asyncio.create_task(_run_shutdown_owner(name, operation))
        _SHUTDOWN_TASKS.add(task)
        task.add_done_callback(_consume_shutdown_task)
        tasks[task] = name
    done, pending = await asyncio.wait(
        tasks,
        timeout=max(0.0, deadline - time.monotonic()),
    )
    errors: list[BaseException] = []
    if pending:
        names = ", ".join(sorted(tasks[task] for task in pending))
        errors.append(TimeoutError(f"shutdown timed out waiting for: {names}"))
        for task in pending:
            task.cancel()
        await asyncio.sleep(0)
    for task in done:
        if task.cancelled():
            continue
        error = task.exception()
        if error is not None:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup("Vox shutdown failed", errors)


def create_app(
    *,
    vox_home: Path | None = None,
    default_device: str = "auto",
    max_loaded: int = 3,
    ttl_seconds: int = 300,
    idle_trim_seconds: int = 0,
    grpc_port: int | None = None,
    bind_host: str = "0.0.0.0",
    preload_models: list[str] | None = None,
    preload_vad: bool = False,
    preload_turn_detector: str | None = None,
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
) -> FastAPI:
    configure_logging()
    configure_hf_runtime()
    app = FastAPI(title="Vox", version="0.1.0", lifespan=lifespan)
    app.add_middleware(UploadSizeLimitMiddleware)
    app.add_middleware(ApiKeyAuthMiddleware)
    cors_origins = _parse_cors_origins(os.environ.get("VOX_CORS_ORIGINS"))
    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["authorization", "content-type", "x-api-key", "x-request-id"],
            expose_headers=["x-request-id"],
        )
    app.add_middleware(RequestIdMiddleware)

    if vox_home is None:
        env_home = os.environ.get("VOX_HOME")
        if env_home:
            vox_home = Path(env_home)
    store = BlobStore(root=vox_home)
    with store.writer_lease():
        store.prune_manifest_staging()
        recover_pull_transactions(store)
        store.gc_blobs(grace_seconds=0)
        prune_stale_install_directories(store.root / "adapters")
        prune_stale_install_directories(store.root / "runtime")
    registry = ModelRegistry(store)
    scheduler = Scheduler(
        registry,
        default_device=default_device,
        max_loaded=max_loaded,
        ttl_seconds=ttl_seconds,
        idle_trim_seconds=idle_trim_seconds,
    )

    app.state.store = store
    app.state.registry = registry
    app.state.scheduler = scheduler
    app.state.speech_context = SpeechContextService(home=store.root)
    app.state.pull_tasks = PullTaskRegistry()
    app.state.rtc_registry = RtcSessionRegistry()
    app.state.grpc_port = grpc_port
    app.state.bind_host = bind_host
    app.state.max_upload_bytes = configured_max_upload_bytes()
    app.state.preload_models = list(preload_models or [])
    app.state.preload_vad = preload_vad
    app.state.preload_turn_detector = preload_turn_detector
    app.state.shutdown_timeout_seconds = max(0.1, float(shutdown_timeout_seconds))

    from vox.server.pondsocket_gateway import install_pondsocket_gateway
    from vox.server.routes import bidi, health, models, rtc, stream, synthesize, system, transcribe, voices

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(system.router)
    app.include_router(transcribe.router)
    app.include_router(synthesize.router)
    app.include_router(voices.router)
    app.include_router(stream.router)
    app.include_router(bidi.router)
    app.include_router(rtc.router)

    if not install_pondsocket_gateway(app):
        raise RuntimeError("PondSocket gateway is required but pondsocket and pondsocket-asgi are unavailable")

    return app
