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

logger = logging.getLogger(__name__)


def _parse_preload_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [m.strip() for m in value.split(",") if m.strip()]


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


async def _preload_models(app: FastAPI, model_refs: list[str]) -> None:
    for ref in model_refs:
        try:
            async with app.state.scheduler.acquire(ref):
                pass
            logger.info("Preloaded model: %s", ref)
        except Exception as exc:
            logger.warning("Failed to preload %s: %s", ref, exc)


async def _preload_vad() -> None:
    try:
        from vox.streaming.vad import SileroVAD

        SileroVAD()._ensure_model()
        logger.info("Preloaded Silero VAD")
    except Exception as exc:
        logger.warning("Failed to preload VAD: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    grpc_server = None
    await app.state.scheduler.start()
    try:
        preload_refs = list(getattr(app.state, "preload_models", []))
        env_preload = _parse_preload_list(os.environ.get("VOX_PRELOAD"))

        seen: set[str] = set()
        merged: list[str] = []
        for ref in preload_refs + env_preload:
            if ref not in seen:
                seen.add(ref)
                merged.append(ref)
        if merged:
            await _preload_models(app, merged)

        if getattr(app.state, "preload_vad", False) or _env_bool("VOX_PRELOAD_VAD"):
            await _preload_vad()

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
    finally:
        try:
            if grpc_server is not None:
                await grpc_server.stop(grace=5)
                logger.info("gRPC server stopped")
        finally:
            await app.state.scheduler.stop()
            logger.info("Vox server stopped")


def create_app(
    *,
    vox_home: Path | None = None,
    default_device: str = "auto",
    max_loaded: int = 3,
    ttl_seconds: int = 300,
    grpc_port: int | None = None,
    preload_models: list[str] | None = None,
    preload_vad: bool = False,
) -> FastAPI:
    configure_hf_runtime()
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
    app.state.preload_models = list(preload_models or [])
    app.state.preload_vad = preload_vad

    from vox.server.routes import bidi, conversation, health, models, stream, synthesize, transcribe, voices
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(transcribe.router)
    app.include_router(synthesize.router)
    app.include_router(voices.router)
    app.include_router(stream.router)
    app.include_router(bidi.router)
    app.include_router(conversation.router)

    return app
