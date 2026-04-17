from __future__ import annotations

import asyncio
import logging
import os

import grpc

from vox.core.hf_runtime import configure_hf_runtime
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore, Manifest, ManifestLayer
from vox.core.types import parse_model_name
from vox.grpc import vox_pb2, vox_pb2_grpc

logger = logging.getLogger(__name__)


class ModelServicer(vox_pb2_grpc.ModelServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Pull(self, request, context):
        explicit_tag = ":" in request.name
        name, tag = parse_model_name(request.name)
        resolved_name, resolved_tag = self._registry.resolve_model_ref(name, tag, explicit_tag=explicit_tag)
        catalog_entry = self._registry.lookup(name, tag, explicit_tag=explicit_tag)
        if not catalog_entry:
            yield vox_pb2.PullProgress(status="error", error=f"Model '{request.name}' not found in catalog")
            return

        yield vox_pb2.PullProgress(status=f"pulling {request.name}")

        adapter_name = catalog_entry.get("adapter", "")
        adapter_package = catalog_entry.get("adapter_package", "")
        if adapter_package:
            yield vox_pb2.PullProgress(status=f"checking adapter {adapter_name}")
            if not self._registry.ensure_adapter(adapter_name, adapter_package):
                yield vox_pb2.PullProgress(
                    status="error",
                    error=f"Failed to install adapter package: {adapter_package}",
                )
                return
            yield vox_pb2.PullProgress(status=f"adapter {adapter_name} ready")

        source = catalog_entry["source"]
        specific_files = catalog_entry.get("files")

        try:
            configure_hf_runtime()
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()

            if specific_files:
                files_to_download = specific_files
            else:
                repo_info = await asyncio.to_thread(api.repo_info, source)
                files_to_download = [
                    s.rfilename for s in repo_info.siblings
                    if not s.rfilename.startswith(".")
                ]

            layers = []
            total_files = len(files_to_download)

            for i, filename in enumerate(files_to_download):
                yield vox_pb2.PullProgress(
                    status=f"downloading {filename}",
                    completed=i,
                    total=total_files,
                )

                local_path = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=source,
                    filename=filename,
                    cache_dir=None,
                )

                file_size = os.path.getsize(local_path)
                with open(local_path, "rb") as f:
                    digest = self._store.write_blob(f)

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
                    "parameters": catalog_entry.get("parameters", {}),
                    "description": catalog_entry.get("description", ""),
                    "license": catalog_entry.get("license", ""),
                    "adapter_package": catalog_entry.get("adapter_package", ""),
                },
            )
            self._store.save_manifest(resolved_name, resolved_tag, manifest)

            yield vox_pb2.PullProgress(status="success")

        except Exception as e:
            logger.exception(f"Failed to pull {request.name}")
            yield vox_pb2.PullProgress(status="error", error=str(e))

    async def List(self, request, context):
        models = self._store.list_models()
        return vox_pb2.ListModelsResponse(
            models=[
                vox_pb2.ModelInfo(
                    name=m.full_name,
                    type=m.type.value,
                    format=m.format.value,
                    architecture=m.architecture,
                    size_bytes=m.size_bytes,
                    description=m.description,
                )
                for m in models
            ]
        )

    async def Show(self, request, context):
        explicit_tag = ":" in request.name
        name, tag = parse_model_name(request.name)
        resolved_name, resolved_tag = self._registry.resolve_model_ref(name, tag, explicit_tag=explicit_tag)
        manifest = self._store.resolve_model(resolved_name, resolved_tag)
        if not manifest:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Model '{request.name}' not found")

        config_map = {}
        for k, v in manifest.config.items():
            config_map[k] = str(v) if not isinstance(v, str) else v

        return vox_pb2.ShowResponse(
            name=request.name,
            config=config_map,
            layers=[
                vox_pb2.LayerInfo(
                    media_type=layer.media_type,
                    digest=layer.digest,
                    size=layer.size,
                    filename=layer.filename,
                )
                for layer in manifest.layers
            ],
        )

    async def Delete(self, request, context):
        explicit_tag = ":" in request.name
        name, tag = parse_model_name(request.name)
        resolved_name, resolved_tag = self._registry.resolve_model_ref(name, tag, explicit_tag=explicit_tag)

        unloaded = await self._scheduler.unload(f"{resolved_name}:{resolved_tag}")
        if not unloaded:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"Model '{request.name}' is currently in use")

        manifest = self._store.resolve_model(resolved_name, resolved_tag)
        if not manifest:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Model '{request.name}' not found")

        self._store.delete_model(resolved_name, resolved_tag)
        self._store.gc_blobs()
        return vox_pb2.DeleteResponse(status="success")
