from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def parse_preload_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [model_ref.strip() for model_ref in value.split(",") if model_ref.strip()]


def env_bool(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def merged_preload_models(explicit_refs: list[str], env_value: str | None) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for ref in explicit_refs + parse_preload_list(env_value):
        if ref in seen:
            continue
        seen.add(ref)
        merged.append(ref)
    return merged


def should_preload_vad(explicit: bool) -> bool:
    return explicit or env_bool("VOX_PRELOAD_VAD")


async def preload_models(scheduler: Any, model_refs: list[str]) -> None:
    for ref in model_refs:
        try:
            async with scheduler.acquire(ref):
                pass
            logger.info("Preloaded model: %s", ref)
        except Exception as exc:
            logger.warning("Failed to preload %s: %s", ref, exc)


async def preload_vad() -> None:
    try:
        from vox.streaming.vad import SileroVAD

        SileroVAD()._ensure_model()
        logger.info("Preloaded Silero VAD")
    except Exception as exc:
        logger.warning("Failed to preload VAD: %s", exc)
