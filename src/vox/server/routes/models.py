from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vox.core.hf_runtime import configure_hf_runtime
from vox.core.store import Manifest, ManifestLayer
from vox.core.types import parse_model_name

logger = logging.getLogger(__name__)
router = APIRouter()


class PullRequest(BaseModel):
    name: str


@router.post("/v1/models/pull")
async def pull_model(req: PullRequest, request: Request):
    store = request.app.state.store
    registry = request.app.state.registry

    explicit_tag = ":" in req.name
    name, tag = parse_model_name(req.name)
    resolved_name, resolved_tag = registry.resolve_model_ref(name, tag, explicit_tag=explicit_tag)
    catalog_entry = registry.lookup(name, tag, explicit_tag=explicit_tag)
    if not catalog_entry:
        raise HTTPException(status_code=404, detail=f"Model '{req.name}' not found in catalog")

    logger.info(
        "pull requested: %s -> %s:%s (adapter=%s, source=%s)",
        req.name, resolved_name, resolved_tag,
        catalog_entry.get("adapter", "?"), catalog_entry.get("source", "?"),
    )

    async def stream_progress():
        yield json.dumps({"status": f"pulling {req.name}"}) + "\n"


        adapter_name = catalog_entry.get("adapter", "")
        adapter_package = catalog_entry.get("adapter_package", "")
        if adapter_package:
            yield json.dumps({"status": f"checking adapter {adapter_name}"}) + "\n"
            if not registry.ensure_adapter(adapter_name, adapter_package):
                yield json.dumps({
                    "status": "error",
                    "error": f"Failed to install adapter package: {adapter_package}"
                }) + "\n"
                return
            yield json.dumps({"status": f"adapter {adapter_name} ready"}) + "\n"

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

            layers: list[dict[str, Any]] = []
            total_files = len(files_to_download)

            for i, filename in enumerate(files_to_download):
                yield json.dumps({
                    "status": f"downloading {filename}",
                    "completed": i,
                    "total": total_files,
                }) + "\n"


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

                layers.append({
                    "media_type": media_type,
                    "digest": digest,
                    "size": file_size,
                    "filename": filename,
                })


            manifest = Manifest(
                layers=[ManifestLayer(**layer_dict) for layer_dict in layers],
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

            total_bytes = sum(layer["size"] for layer in layers)
            logger.info(
                "pull complete: %s:%s (%d layers, %.1f MiB)",
                resolved_name, resolved_tag, len(layers), total_bytes / (1024 * 1024),
            )
            yield json.dumps({"status": "success"}) + "\n"

        except Exception as e:
            logger.exception("pull failed: %s", req.name)
            yield json.dumps({"status": "error", "error": str(e)}) + "\n"

    return StreamingResponse(stream_progress(), media_type="application/x-ndjson")


@router.get("/v1/models")
async def list_models(request: Request):
    store = request.app.state.store
    models = store.list_models()
    return {
        "models": [
            {
                "name": m.full_name,
                "type": m.type.value,
                "format": m.format.value,
                "architecture": m.architecture,
                "size_bytes": m.size_bytes,
                "description": m.description,
            }
            for m in models
        ]
    }


async def _show_by_name(name: str, request: Request) -> dict:
    store = request.app.state.store
    registry = request.app.state.registry
    explicit_tag = ":" in name
    parsed_name, parsed_tag = parse_model_name(name)
    resolved_name, resolved_tag = registry.resolve_model_ref(
        parsed_name, parsed_tag, explicit_tag=explicit_tag,
    )
    manifest = store.resolve_model(resolved_name, resolved_tag)
    if not manifest:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {
        "name": name,
        "config": manifest.config,
        "layers": [
            {
                "media_type": layer_dict.media_type,
                "digest": layer_dict.digest,
                "size": layer_dict.size,
                "filename": layer_dict.filename,
            }
            for layer_dict in manifest.layers
        ],
    }


async def _delete_by_name(name: str, request: Request) -> dict:
    store = request.app.state.store
    scheduler = request.app.state.scheduler
    registry = request.app.state.registry
    explicit_tag = ":" in name
    parsed_name, parsed_tag = parse_model_name(name)
    resolved_name, resolved_tag = registry.resolve_model_ref(
        parsed_name, parsed_tag, explicit_tag=explicit_tag,
    )

    unloaded = await scheduler.unload(f"{resolved_name}:{resolved_tag}")
    if not unloaded:
        raise HTTPException(status_code=409, detail=f"Model '{name}' is currently in use")

    manifest = store.resolve_model(resolved_name, resolved_tag)
    if not manifest:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    store.delete_model(resolved_name, resolved_tag)
    store.gc_blobs()
    logger.info("model deleted: %s:%s", resolved_name, resolved_tag)
    return {"status": "success"}


@router.get("/v1/models/{name:path}")
async def show_model_v1(name: str, request: Request):
    return await _show_by_name(name, request)


@router.delete("/v1/models/{name:path}")
async def delete_model_v1(name: str, request: Request):
    return await _delete_by_name(name, request)
