from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app.state.scheduler.start()

    grpc_server = None
    grpc_port = getattr(app.state, "grpc_port", None)
    if grpc_port:
        from vox.grpc.server import start_grpc_server
        grpc_server = await start_grpc_server(
            app.state.store,
            app.state.registry,
            app.state.scheduler,
            port=grpc_port,
        )

    logger.info("Vox server started")
    yield

    if grpc_server is not None:
        await grpc_server.stop(grace=5)
        logger.info("gRPC server stopped")

    await app.state.scheduler.stop()
    logger.info("Vox server stopped")


def create_app(
    *,
    vox_home: Path | None = None,
    default_device: str = "auto",
    max_loaded: int = 3,
    ttl_seconds: int = 300,
    grpc_port: int | None = None,
) -> FastAPI:
    app = FastAPI(title="Vox", version="0.1.0", lifespan=lifespan)

    if vox_home is None:
        env_home = os.environ.get("VOX_HOME")
        if env_home:
            vox_home = Path(env_home)
    store = BlobStore(root=vox_home)
    registry = ModelRegistry(store)
    scheduler = Scheduler(registry, default_device=default_device, max_loaded=max_loaded, ttl_seconds=ttl_seconds)

    app.state.store = store
    app.state.registry = registry
    app.state.scheduler = scheduler
    app.state.grpc_port = grpc_port

    from vox.server.routes import health, models, synthesize, transcribe, voices, stream
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(transcribe.router)
    app.include_router(synthesize.router)
    app.include_router(voices.router)
    app.include_router(stream.router)

    return app
