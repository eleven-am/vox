from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from vox.core.hf_runtime import configure_hf_runtime
from vox.core.store import Manifest, ManifestLayer
from vox.core.types import ModelInfo, parse_model_name
from vox.operations.errors import (
    CatalogEntryNotFoundError,
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


def list_models(*, store: Any) -> list[ModelInfo]:
    return list(store.list_models())


def show_model(*, store: Any, registry: Any, name: str) -> ShowResult:
    explicit_tag = ":" in name
    parsed_name, parsed_tag = parse_model_name(name)
    resolved_name, resolved_tag = registry.resolve_model_ref(
        parsed_name, parsed_tag, explicit_tag=explicit_tag,
    )
    manifest = store.resolve_model(resolved_name, resolved_tag)
    if not manifest:
        raise StoredModelNotFoundError(name)
    return ShowResult(
        name=name,
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


async def delete_model(*, store: Any, scheduler: Any, registry: Any, name: str) -> None:
    explicit_tag = ":" in name
    parsed_name, parsed_tag = parse_model_name(name)
    resolved_name, resolved_tag = registry.resolve_model_ref(
        parsed_name, parsed_tag, explicit_tag=explicit_tag,
    )

    unloaded = await scheduler.unload(f"{resolved_name}:{resolved_tag}")
    if not unloaded:
        raise ModelInUseError(name)

    manifest = store.resolve_model(resolved_name, resolved_tag)
    if not manifest:
        raise StoredModelNotFoundError(name)

    store.delete_model(resolved_name, resolved_tag)
    store.gc_blobs()
    logger.info("model deleted: %s:%s", resolved_name, resolved_tag)


def pull_model(
    *,
    store: Any,
    scheduler: Any,
    registry: Any,
    name: str,
) -> AsyncIterator[PullEvent]:
    explicit_tag = ":" in name
    parsed_name, parsed_tag = parse_model_name(name)
    resolved_name, resolved_tag = registry.resolve_model_ref(parsed_name, parsed_tag, explicit_tag=explicit_tag)
    catalog_entry = registry.lookup(parsed_name, parsed_tag, explicit_tag=explicit_tag)

    if not catalog_entry:
        raise CatalogEntryNotFoundError(name)

    logger.info(
        "pull requested: %s -> %s:%s (adapter=%s, source=%s)",
        name, resolved_name, resolved_tag,
        catalog_entry.get("adapter", "?"), catalog_entry.get("source", "?"),
    )

    async def _gen() -> AsyncIterator[PullEvent]:
        yield PullEvent(status=f"pulling {name}")

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
                },
            )
            store.save_manifest(resolved_name, resolved_tag, manifest)

            if adapter_name == "voxtral-tts-vllm":
                model_ref = f"{resolved_name}:{resolved_tag}"
                yield PullEvent(status=f"preloading {model_ref}")
                await scheduler.preload(model_ref)
                yield PullEvent(status=f"{model_ref} ready")

            total_bytes = sum(layer.size for layer in layers)
            logger.info(
                "pull complete: %s:%s (%d layers, %.1f MiB)",
                resolved_name, resolved_tag, len(layers), total_bytes / (1024 * 1024),
            )
            yield PullEvent(status="success")

        except Exception as e:
            logger.exception("pull failed: %s", name)
            yield PullEvent(status="error", error=str(e))

    return _gen()
