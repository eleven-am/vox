from __future__ import annotations

import logging
import os
import platform
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from vox.core.runtime import RuntimeCapabilities, detect_runtime_capabilities, infer_runtime_profile
from vox.core.types import ModelFormat, ModelInfo

logger = logging.getLogger(__name__)


DEVICE_MEMORY_HEADROOM_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class PlacementTier:
    name: str
    total_memory_max_bytes: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedModelView:
    full_name: str
    device: str
    vram_bytes: int
    ref_count: int
    last_used: float


@dataclass
class Placement:
    device: str
    precision: str | None = None
    evict: list[str] = field(default_factory=list)
    tier: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def runtime_profile_for_alias(*, device_hint: str | None = None) -> str:
    device = (device_hint or os.environ.get("VOX_DEVICE", "auto")).strip().lower()
    machine = platform.machine().strip().lower()
    if device == "cuda" and machine in {"arm64", "aarch64"}:
        return "spark"
    return infer_runtime_profile(device_hint=device)


def auto_device_for_model(info: ModelInfo, capabilities: RuntimeCapabilities) -> str:
    if info.format == ModelFormat.ONNX:
        if capabilities.onnx_cuda:
            return "cuda"
        return "auto" if capabilities.onnx_coreml else "cpu"

    if info.format in {ModelFormat.PYTORCH, ModelFormat.CT2}:
        if capabilities.torch_cuda:
            return "cuda"
        if capabilities.mps:
            return "mps"
        return "cpu"

    return "cpu"


def select_tier(
    tiers: Iterable[PlacementTier],
    *,
    total_memory_bytes: int | None,
) -> PlacementTier | None:
    candidates = [t for t in tiers]
    if not candidates:
        return None
    if total_memory_bytes is None:
        for tier in candidates:
            if tier.total_memory_max_bytes is None:
                return tier
        return candidates[-1]

    bounded = [t for t in candidates if t.total_memory_max_bytes is not None]
    bounded.sort(key=lambda t: t.total_memory_max_bytes or 0)
    for tier in bounded:
        if total_memory_bytes <= (tier.total_memory_max_bytes or 0):
            return tier
    for tier in candidates:
        if tier.total_memory_max_bytes is None:
            return tier
    return bounded[-1] if bounded else None


def _eviction_plan(
    *,
    device: str,
    required_bytes: int,
    free_bytes: int,
    loaded_models: Iterable[LoadedModelView],
) -> tuple[list[str], int]:
    if required_bytes <= free_bytes:
        return [], free_bytes

    candidates = sorted(
        (m for m in loaded_models if m.ref_count == 0 and m.device == device),
        key=lambda m: m.last_used,
    )
    evict: list[str] = []
    projected_free = free_bytes
    for candidate in candidates:
        evict.append(candidate.full_name)
        projected_free += max(int(candidate.vram_bytes), 0)
        if required_bytes <= projected_free:
            break
    return evict, projected_free


def decide_placement(
    info: ModelInfo,
    *,
    requested_device: str,
    capabilities: RuntimeCapabilities,
    loaded_models: Iterable[LoadedModelView] = (),
    estimated_vram_bytes: int = 0,
    free_memory_query: Callable[[str], int | None] | None = None,
    total_memory_query: Callable[[str], int | None] | None = None,
    tiers: Iterable[PlacementTier] = (),
) -> Placement:
    requested = (requested_device or "auto").strip().lower()
    loaded_list = list(loaded_models)
    tier_list = list(tiers)

    if requested != "auto":
        chosen_tier = _resolve_tier(tier_list, total_memory_query) if requested == "cuda" else None
        notes = dict(chosen_tier.extras) if chosen_tier is not None else {}
        return Placement(
            device=requested,
            precision=None,
            evict=[],
            tier=chosen_tier.name if chosen_tier else None,
            notes=notes,
        )

    candidate_device = auto_device_for_model(info, capabilities)
    chosen_tier = (
        _resolve_tier(tier_list, total_memory_query)
        if candidate_device == "cuda"
        else None
    )
    notes = dict(chosen_tier.extras) if chosen_tier is not None else {}

    if candidate_device != "cuda" or estimated_vram_bytes <= 0 or free_memory_query is None:
        return Placement(
            device=candidate_device,
            precision=None,
            evict=[],
            tier=chosen_tier.name if chosen_tier else None,
            notes=notes,
        )

    free_bytes = free_memory_query(candidate_device)
    if free_bytes is None:
        return Placement(
            device=candidate_device,
            precision=None,
            evict=[],
            tier=chosen_tier.name if chosen_tier else None,
            notes=notes,
        )

    required_bytes = estimated_vram_bytes + DEVICE_MEMORY_HEADROOM_BYTES
    if required_bytes <= free_bytes:
        return Placement(
            device=candidate_device,
            precision=None,
            evict=[],
            tier=chosen_tier.name if chosen_tier else None,
            notes=notes,
        )

    evict, projected_free = _eviction_plan(
        device=candidate_device,
        required_bytes=required_bytes,
        free_bytes=free_bytes,
        loaded_models=loaded_list,
    )

    if required_bytes <= projected_free:
        return Placement(
            device=candidate_device,
            precision=None,
            evict=evict,
            tier=chosen_tier.name if chosen_tier else None,
            notes=notes,
        )

    logger.info(
        "Routing %s to CPU: estimated %d bytes plus headroom exceeds free %s memory (%d bytes)",
        info.full_name,
        estimated_vram_bytes,
        candidate_device,
        free_bytes,
    )
    return Placement(
        device="cpu",
        precision=None,
        evict=[],
        tier=None,
        notes={},
    )


def _resolve_tier(
    tiers: list[PlacementTier],
    total_memory_query: Callable[[str], int | None] | None,
) -> PlacementTier | None:
    if not tiers:
        return None
    total = None
    if total_memory_query is not None:
        total = total_memory_query("cuda")
    return select_tier(tiers, total_memory_bytes=total)


def detect_capabilities() -> RuntimeCapabilities:
    return detect_runtime_capabilities()


__all__ = [
    "DEVICE_MEMORY_HEADROOM_BYTES",
    "LoadedModelView",
    "Placement",
    "PlacementTier",
    "RuntimeCapabilities",
    "auto_device_for_model",
    "decide_placement",
    "detect_capabilities",
    "infer_runtime_profile",
    "runtime_profile_for_alias",
    "select_tier",
]
