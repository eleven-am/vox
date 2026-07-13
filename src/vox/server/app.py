from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from vox.core.hf_runtime import configure_hf_runtime
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.core.temp_storage import prune_stale_temp_dirs
from vox.logging_config import configure_logging
from vox.server.app_services import app_pondsocket, app_rtc_registry, app_services
from vox.server.middleware import ApiKeyAuthMiddleware, RequestIdMiddleware
from vox.server.preload import (
    merged_preload_models,
    preload_models,
    preload_turn_detector,
    preload_vad,
    should_preload_vad,
)
from vox.server.rtc_registry import RtcSessionRegistry

logger = logging.getLogger(__name__)


def _parse_cors_origins(value: str | None) -> list[str]:
    if not value:
        return []
    return [origin.strip() for origin in value.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    grpc_server = None
    services = app_services(app)
    rtc_registry = app_rtc_registry(app)
    prune_stale_temp_dirs(services.store.root / "tmp")
    await services.scheduler.start()
    try:
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

            grpc_server = await start_grpc_server(
                services.store,
                services.registry,
                services.scheduler,
                rtc_registry,
                port=grpc_port,
            )

        logger.info("Vox server started")
        yield
    finally:
        try:
            if grpc_server is not None:
                await grpc_server.stop(grace=5)
                logger.info("gRPC server stopped")
        finally:
            try:
                pond = app_pondsocket(app)
                if pond is not None:
                    await pond.close()
                    logger.info("PondSocket server stopped")
            finally:
                await services.scheduler.stop()
                logger.info("Vox server stopped")


def create_app(
    *,
    vox_home: Path | None = None,
    default_device: str = "auto",
    max_loaded: int = 3,
    ttl_seconds: int = 300,
    max_vram_bytes: int | None = None,
    vram_headroom_bytes: int = 512 * 1024 * 1024,
    idle_trim_seconds: int = 0,
    memory_over_budget: str = "reject",
    grpc_port: int | None = None,
    preload_models: list[str] | None = None,
    preload_vad: bool = False,
    preload_turn_detector: str | None = None,
) -> FastAPI:
    configure_logging()
    configure_hf_runtime()
    app = FastAPI(title="Vox", version="0.1.0", lifespan=lifespan)
    app.add_middleware(ApiKeyAuthMiddleware)
    cors_origins = _parse_cors_origins(os.environ.get("VOX_CORS_ORIGINS"))
    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["authorization", "content-type", "x-api-key"],
        )
    app.add_middleware(RequestIdMiddleware)

    if vox_home is None:
        env_home = os.environ.get("VOX_HOME")
        if env_home:
            vox_home = Path(env_home)
    store = BlobStore(root=vox_home)
    registry = ModelRegistry(store)
    scheduler = Scheduler(
        registry,
        default_device=default_device,
        max_loaded=max_loaded,
        ttl_seconds=ttl_seconds,
        max_vram_bytes=max_vram_bytes,
        vram_headroom_bytes=vram_headroom_bytes,
        idle_trim_seconds=idle_trim_seconds,
        memory_over_budget=memory_over_budget,
    )

    app.state.store = store
    app.state.registry = registry
    app.state.scheduler = scheduler
    app.state.rtc_registry = RtcSessionRegistry()
    app.state.grpc_port = grpc_port
    app.state.preload_models = list(preload_models or [])
    app.state.preload_vad = preload_vad
    app.state.preload_turn_detector = preload_turn_detector

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
