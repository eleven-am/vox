from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from vox.core.capabilities import incompatible_pull_allowed
from vox.core.hf_runtime import configure_hf_runtime
from vox.core.model_resolution import parse_model_variant_ref, resolve_catalog_entry
from vox.core.store import Manifest, ManifestLayer
from vox.core.types import ModelInfo
from vox.operations.errors import (
    CatalogEntryNotFoundError,
    ModelIncompatibleError,
    ModelInUseError,
    StoredModelNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelLayer:
    media_type: str
    digest: str
    size: int
    filename: str


@dataclass(frozen=True)
class ShowResult:
    name: str
    config: dict[str, Any]
    layers: tuple[ModelLayer, ...]


@dataclass(frozen=True)
class PullEvent:
    status: str
    completed: int = 0
    total: int = 0
    error: str = ""


@dataclass(frozen=True)
class ModelReferenceRequest:
    name: str
    variant: str | None = None


@dataclass(frozen=True)
class ResolvedModelReference:
    requested_name: str
    parsed_name: str
    parsed_tag: str
    requested_variant: str | None
    resolved_name: str
    resolved_tag: str
    explicit_tag: bool


def model_reference_request_from_fields(
    *,
    name: str,
    variant: str | None = None,
) -> ModelReferenceRequest:
    return ModelReferenceRequest(name=name, variant=variant or None)


def model_info_payload(model: ModelInfo) -> dict[str, Any]:
    return {
        "name": model.full_name,
        "type": model.type.value,
        "format": model.format.value,
        "architecture": model.architecture,
        "size_bytes": model.size_bytes,
        "description": model.description,
    }


def list_models_payload(models: list[ModelInfo]) -> dict[str, Any]:
    return {"models": [model_info_payload(model) for model in models]}


def model_layer_payload(layer: ModelLayer) -> dict[str, Any]:
    return {
        "media_type": layer.media_type,
        "digest": layer.digest,
        "size": layer.size,
        "filename": layer.filename,
    }


def show_model_payload(result: ShowResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "config": result.config,
        "layers": [model_layer_payload(layer) for layer in result.layers],
    }


def pull_event_payload(event: PullEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": event.status}
    if event.total > 0:
        payload["completed"] = event.completed
        payload["total"] = event.total
    if event.error:
        payload["error"] = event.error
    return payload


def delete_model_payload() -> dict[str, str]:
    return {"status": "success"}


def list_models(*, store: Any) -> list[ModelInfo]:
    return list(store.list_models())


def show_model(*, store: Any, registry: Any, request: ModelReferenceRequest) -> ShowResult:
    resolved = resolve_model_reference(registry=registry, request=request)
    manifest = store.resolve_model(resolved.resolved_name, resolved.resolved_tag)
    if not manifest:
        raise StoredModelNotFoundError(request.name)
    return ShowResult(
        name=request.name,
        config=dict(manifest.config),
        layers=tuple(
            ModelLayer(
                media_type=layer.media_type,
                digest=layer.digest,
                size=layer.size,
                filename=layer.filename,
            )
            for layer in manifest.layers
        ),
    )


async def delete_model(
    *,
    store: Any,
    scheduler: Any,
    registry: Any,
    request: ModelReferenceRequest,
) -> None:
    resolved = resolve_model_reference(registry=registry, request=request)

    unloaded = await scheduler.unload(f"{resolved.resolved_name}:{resolved.resolved_tag}")
    if not unloaded:
        raise ModelInUseError(request.name)

    manifest = store.resolve_model(resolved.resolved_name, resolved.resolved_tag)
    if not manifest:
        raise StoredModelNotFoundError(request.name)

    store.delete_model(resolved.resolved_name, resolved.resolved_tag)
    store.gc_blobs()
    logger.info("model deleted: %s:%s", resolved.resolved_name, resolved.resolved_tag)


def pull_model(
    *,
    store: Any,
    scheduler: Any,
    registry: Any,
    request: ModelReferenceRequest,
) -> AsyncIterator[PullEvent]:
    resolved = resolve_model_reference(registry=registry, request=request)
    catalog_entry = registry.lookup(
        resolved.parsed_name,
        resolved.parsed_tag,
        explicit_tag=resolved.explicit_tag,
    )

    if not catalog_entry:
        raise CatalogEntryNotFoundError(request.name)

    variant_resolution = resolve_catalog_entry(
        catalog_entry,
        forced_variant=resolved.requested_variant,
    )
    catalog_entry = variant_resolution.entry
    missing = list(variant_resolution.missing)
    if not incompatible_pull_allowed() and missing:
        raise ModelIncompatibleError(request.name, missing)
    if not catalog_entry:
        raise ModelIncompatibleError(request.name, missing or ["no compatible catalog entry resolved"])

    logger.info(
        "pull requested: %s -> %s:%s (variant=%s, adapter=%s, source=%s)",
        request.name,
        resolved.resolved_name,
        resolved.resolved_tag,
        variant_resolution.variant_id or "-",
        catalog_entry.get("adapter", "?"), catalog_entry.get("source", "?"),
    )

    async def _gen() -> AsyncIterator[PullEvent]:
        yield PullEvent(status=f"pulling {request.name}")
        if missing:
            yield PullEvent(
                status="warning",
                error=(
                    "pull compatibility bypassed by VOX_ALLOW_INCOMPATIBLE=1: "
                    + "; ".join(missing)
                ),
            )
        for warning in variant_resolution.warnings:
            yield PullEvent(status="warning", error=warning)

        adapter_name = catalog_entry.get("adapter", "")
        adapter_package = catalog_entry.get("adapter_package", "")
        if adapter_package:
            yield PullEvent(status=f"checking adapter {adapter_name}")
            if not registry.ensure_adapter(adapter_name, adapter_package):
                yield PullEvent(status="error", error=f"Failed to install adapter package: {adapter_package}")
                return
            yield PullEvent(status=f"adapter {adapter_name} ready")

        source = catalog_entry["source"]
        specific_files = catalog_entry.get("files")

        try:
            configure_hf_runtime()
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()

            if specific_files:
                files_to_download = list(specific_files)
            else:
                repo_info = await asyncio.to_thread(api.repo_info, source)
                files_to_download = [
                    s.rfilename for s in repo_info.siblings
                    if not s.rfilename.startswith(".")
                ]

            layers: list[ManifestLayer] = []
            total_files = len(files_to_download)

            for i, filename in enumerate(files_to_download):
                yield PullEvent(status=f"downloading {filename}", completed=i, total=total_files)

                local_path = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=source,
                    filename=filename,
                    cache_dir=None,
                )

                file_size = os.path.getsize(local_path)
                with open(local_path, "rb") as f:
                    digest = store.write_blob(f)

                ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
                media_type = f"application/vox.model.{ext}"

                layers.append(ManifestLayer(
                    media_type=media_type,
                    digest=digest,
                    size=file_size,
                    filename=filename,
                ))

            manifest = Manifest(
                layers=layers,
                config={
                    "architecture": catalog_entry["architecture"],
                    "type": catalog_entry["type"],
                    "adapter": catalog_entry["adapter"],
                    "format": catalog_entry["format"],
                    "source": source,
                    "runtime_source": catalog_entry.get("runtime_source", ""),
                    "parameters": catalog_entry.get("parameters", {}),
                    "description": catalog_entry.get("description", ""),
                    "license": catalog_entry.get("license", ""),
                    "adapter_package": catalog_entry.get("adapter_package", ""),
                    "runtime": _runtime_diagnostic_payload(
                        variant_id=variant_resolution.variant_id,
                        preferred_backend=variant_resolution.preferred_backend,
                        warnings=variant_resolution.warnings,
                        snapshot=variant_resolution.snapshot,
                    ),
                },
            )
            store.save_manifest(resolved.resolved_name, resolved.resolved_tag, manifest)

            if adapter_name:
                yield PullEvent(status=f"preparing adapter runtime {adapter_name}")
                await asyncio.to_thread(_prepare_adapter_runtime, registry, adapter_name)
                yield PullEvent(status=f"adapter runtime {adapter_name} ready")

            if adapter_name == "voxtral-tts-vllm":
                model_ref = f"{resolved.resolved_name}:{resolved.resolved_tag}"
                yield PullEvent(status=f"preloading {model_ref}")
                await scheduler.preload(model_ref)
                yield PullEvent(status=f"{model_ref} ready")

            total_bytes = sum(layer.size for layer in layers)
            logger.info(
                "pull complete: %s:%s (%d layers, %.1f MiB)",
                resolved.resolved_name,
                resolved.resolved_tag,
                len(layers),
                total_bytes / (1024 * 1024),
            )
            yield PullEvent(status="success")

        except Exception as e:
            logger.exception("pull failed: %s", request.name)
            yield PullEvent(status="error", error=str(e))

    return _gen()


def resolve_model_reference(
    *,
    registry: Any,
    request: ModelReferenceRequest,
) -> ResolvedModelReference:
    parsed = parse_model_variant_ref(request.name)
    resolved_name, resolved_tag = registry.resolve_model_ref(
        parsed.name,
        parsed.tag,
        explicit_tag=parsed.explicit_tag,
    )
    return ResolvedModelReference(
        requested_name=request.name,
        parsed_name=parsed.name,
        parsed_tag=parsed.tag,
        requested_variant=request.variant,
        resolved_name=resolved_name,
        resolved_tag=resolved_tag,
        explicit_tag=parsed.explicit_tag,
    )


def _runtime_diagnostic_payload(
    *,
    variant_id: str | None,
    preferred_backend: str | None,
    warnings: tuple[str, ...],
    snapshot: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checked_at_pull": True,
        "resolved_variant": variant_id or "",
        "preferred_backend": preferred_backend or "",
        "warnings": list(warnings),
    }
    if snapshot is not None:
        payload["detected"] = asdict(snapshot)
    return payload


def _prepare_adapter_runtime(registry: Any, adapter_name: str) -> None:
    adapter_cls = registry.get_adapter_class(adapter_name)
    adapter = adapter_cls()
    adapter.prepare_runtime()
