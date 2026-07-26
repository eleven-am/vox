"""Model registry for Vox. The remote registry is the single source of truth."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from vox.core.adapter_resolution import AdapterMutation, AdapterResolver
from vox.core.alias_resolution import resolve_family_alias
from vox.core.errors import ModelLoadError, ModelNotFoundError
from vox.core.store import BlobStore
from vox.core.types import ModelInfo

logger = logging.getLogger(__name__)


# Remote registry is the single source of truth; no bundled offline catalog.
# Fetched entries are cached in-memory for the process.
_entry_cache: dict[tuple[str, str], dict[str, Any]] = {}


REGISTRY_BASE_URL = "https://raw.githubusercontent.com/eleven-am/vox-registry/main"


def fetch_from_registry(name: str, tag: str) -> dict[str, Any] | None:
    """Fetch model metadata from the remote GitHub registry."""
    import httpx

    url = f"{REGISTRY_BASE_URL}/library/{name}/{tag}.json"
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            return resp.json()
        return None
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"Failed to fetch from registry: {url}: {e}")
        return None


_INDEX_CACHE_TTL_S = 300.0
_index_cache: tuple[float, list[dict[str, Any]] | None] | None = None


def fetch_registry_index(*, force_refresh: bool = False) -> list[dict[str, Any]] | None:
    """Fetch the full model index from the remote registry."""
    global _index_cache
    import time

    import httpx

    now = time.monotonic()
    if not force_refresh and _index_cache is not None and now - _index_cache[0] < _INDEX_CACHE_TTL_S:
        return _index_cache[1]

    url = f"{REGISTRY_BASE_URL}/index.json"
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        result = resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        result = None

    _index_cache = (now, result)
    return result


def _is_safe_layer_filename(filename: str) -> bool:
    if not filename or filename in (".", ".."):
        return False
    normalized = filename.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or PureWindowsPath(filename).is_absolute():
        return False
    return not any(part == ".." for part in posix.parts)


class ModelRegistry:
    """Ties the remote registry, blob store, and adapter discovery together."""

    def __init__(self, store: BlobStore, resolver: AdapterResolver | None = None) -> None:
        self._store = store
        self._resolver = resolver or AdapterResolver(self._store.root)

    @property
    def adapter_resolver(self) -> AdapterResolver:
        return self._resolver

    def resolve_model_ref(self, name: str, tag: str = "latest", *, explicit_tag: bool = False) -> tuple[str, str]:
        """Resolve a possibly-bare model reference to a concrete catalog tag."""
        return resolve_family_alias(name, tag, explicit_tag=explicit_tag)

    def lookup(self, name: str, tag: str = "latest", *, explicit_tag: bool = False) -> dict | None:
        """Look up a model from the remote registry (cached in-memory)."""
        name, tag = self.resolve_model_ref(name, tag, explicit_tag=explicit_tag)
        cached = _entry_cache.get((name, tag))
        if cached is not None:
            return cached

        entry = fetch_from_registry(name, tag)
        if entry is not None:
            _entry_cache[(name, tag)] = entry
            logger.info(f"Fetched {name}:{tag} from registry")
        return entry

    def available_models(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the remote registry index as {name: {tag: summary}}."""
        remote = fetch_registry_index()
        result: dict[str, dict[str, dict[str, Any]]] = {}
        if remote:
            for entry in remote:
                name, tag = entry.get("name"), entry.get("tag")
                if name and tag:
                    result.setdefault(name, {})[tag] = entry
        return result

    def ensure_adapter(self, adapter_name: str, package_name: str) -> bool:
        return self._resolver.ensure(adapter_name, package_name)

    def stage_adapter(
        self,
        adapter_name: str,
        package_name: str,
    ) -> AdapterMutation:
        return self._resolver.stage(adapter_name, package_name)

    def get_adapter_class(self, adapter_name: str) -> type:
        return self._resolver.resolve(adapter_name)

    def resolve(self, name: str, tag: str = "latest", *, explicit_tag: bool = False) -> tuple[ModelInfo, Path]:
        """Resolve a model to its :class:`ModelInfo` and a model directory path.

        The model must already be pulled (i.e. a manifest must exist in the
        store).  Raises :class:`ModelNotFoundError` otherwise.

        The returned path is a directory where blobs are symlinked to their
        original filenames, so adapters can load files by name (e.g.
        ``model.onnx``, ``voices.bin``).
        """
        name, tag = self.resolve_model_ref(name, tag, explicit_tag=explicit_tag)
        manifest = self._store.resolve_model(name, tag)
        if manifest is None:
            raise ModelNotFoundError(f"{name}:{tag}")

        cfg = manifest.config
        size = sum(layer.size for layer in manifest.layers)

        info = ModelInfo.from_manifest_config(name, tag, cfg, size_bytes=size)
        adapter_package = cfg.get("adapter_package", "")
        if adapter_package and not self.ensure_adapter(info.adapter, adapter_package):
            raise ModelLoadError(f"Failed to install adapter package: {adapter_package}")

        source = cfg.get("runtime_source") or cfg.get("source")
        if source:
            updated_params = {**info.parameters, "_source": source}
            info = replace(info, parameters=updated_params)

        if not manifest.layers:
            raise ModelNotFoundError(f"{name}:{tag} (manifest has no layers)")

        model_dir = self._store.root / "models" / "links" / name / tag
        model_dir.mkdir(parents=True, exist_ok=True)

        for layer in manifest.layers:
            if not _is_safe_layer_filename(layer.filename):
                raise ModelLoadError(f"Manifest layer filename escapes model directory: {layer.filename!r}")
            link_path = model_dir / layer.filename
            blob_path = self._store.get_blob_path(layer.digest)
            link_path.parent.mkdir(parents=True, exist_ok=True)

            if link_path.is_symlink() and (not link_path.exists() or link_path.resolve() != blob_path.resolve()):
                link_path.unlink()
            if not link_path.exists():
                try:
                    link_path.symlink_to(blob_path)
                except OSError as e:
                    raise ModelLoadError(f"Failed to create symlink {link_path} -> {blob_path}: {e}") from e

        return info, model_dir
