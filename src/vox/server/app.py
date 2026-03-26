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
    """Start scheduler on startup, stop on shutdown."""
    await app.state.scheduler.start()
    logger.info("Vox server started")
    yield
    await app.state.scheduler.stop()
    logger.info("Vox server stopped")


def create_app(
    *,
    vox_home: Path | None = None,
    default_device: str = "auto",
    max_loaded: int = 3,
    ttl_seconds: int = 300,
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

    from vox.server.routes import health, models, synthesize, transcribe, voices
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(transcribe.router)
    app.include_router(synthesize.router)
    app.include_router(voices.router)

    return app
