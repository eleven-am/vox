"""Content-addressable blob store for Vox model files.

Storage layout:

    ~/.vox/
      models/
        manifests/
          library/
            whisper/
              large-v3     # JSON manifest
        blobs/
          sha256-<hex>     # actual model files
      voices/              # cloned voice data
      config.json
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import logging

from vox.core.types import ModelInfo

logger = logging.getLogger(__name__)






@dataclass
class ManifestLayer:
    media_type: str
    digest: str
    size: int
    filename: str

    def __post_init__(self):
        if not self.digest.startswith("sha256-"):
            raise ValueError(f"Invalid digest format: {self.digest!r}")
        if self.size < 0:
            raise ValueError(f"Invalid layer size: {self.size}")


@dataclass
class Manifest:
    schema_version: int = 1
    layers: list[ManifestLayer] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)






def _manifest_to_dict(m: Manifest) -> dict[str, Any]:
    return {
        "schema_version": m.schema_version,
        "layers": [asdict(layer) for layer in m.layers],
        "config": m.config,
    }


def _manifest_from_dict(d: dict[str, Any]) -> Manifest:
    return Manifest(
        schema_version=d.get("schema_version", 1),
        layers=[ManifestLayer(**layer) for layer in d.get("layers", [])],
        config=d.get("config", {}),
    )






_READ_CHUNK = 1 << 20


class BlobStore:
    """Content-addressable blob store with JSON manifests."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path.home() / ".vox"



    @property
    def root(self) -> Path:
        return self._root

    @property
    def blobs_dir(self) -> Path:
        return self._root / "models" / "blobs"

    @property
    def manifests_dir(self) -> Path:
        return self._root / "models" / "manifests" / "library"

    @property
    def voices_dir(self) -> Path:
        return self._root / "voices"



    def get_blob_path(self, digest: str) -> Path:
        """Return the filesystem path for a given digest string (``sha256-<hex>``)."""
        return self.blobs_dir / digest

    def has_blob(self, digest: str) -> bool:
        return self.get_blob_path(digest).exists()

    def write_blob(self, data: BinaryIO) -> str:
        """Write *data* to a blob, computing SHA-256 on the fly.

        Uses a temporary file + atomic rename so readers never see partial
        writes.  Returns the digest string ``sha256-<hex>``.
        """
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

        h = hashlib.sha256()

        fd = tempfile.NamedTemporaryFile(
            dir=self.blobs_dir, delete=False, suffix=".tmp",
        )
        try:
            while True:
                chunk = data.read(_READ_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
                fd.write(chunk)
            fd.flush()
            fd.close()

            digest = f"sha256-{h.hexdigest()}"
            final_path = self.get_blob_path(digest)

            if final_path.exists():

                Path(fd.name).unlink(missing_ok=True)
            else:
                Path(fd.name).rename(final_path)

            return digest
        except BaseException:
            Path(fd.name).unlink(missing_ok=True)
            raise



    def _manifest_path(self, name: str, tag: str) -> Path:
        return self.manifests_dir / name / tag

    def resolve_model(self, name: str, tag: str = "latest") -> Manifest | None:
        """Read and parse a manifest file, returning ``None`` if it does not exist."""
        path = self._manifest_path(name, tag)
        if not path.is_file():
            return None
        with open(path, "r") as f:
            return _manifest_from_dict(json.load(f))

    def save_manifest(self, name: str, tag: str, manifest: Manifest) -> None:
        """Atomically write a manifest JSON file."""
        path = self._manifest_path(name, tag)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(_manifest_to_dict(manifest), f, indent=2)
        tmp.rename(path)

    def list_models(self) -> list[ModelInfo]:
        """Scan the manifests directory and return info for every stored model."""
        models: list[ModelInfo] = []
        if not self.manifests_dir.is_dir():
            return models

        for model_dir in sorted(self.manifests_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            name = model_dir.name
            for tag_file in sorted(model_dir.iterdir()):
                if not tag_file.is_file():
                    continue
                tag = tag_file.name
                try:
                    with open(tag_file) as f:
                        data = json.load(f)
                    manifest = _manifest_from_dict(data)
                    cfg = manifest.config
                    size = sum(layer.size for layer in manifest.layers)
                    models.append(
                        ModelInfo.from_manifest_config(name, tag, cfg, size_bytes=size)
                    )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Skipping corrupted manifest {tag_file}: {e}")
                    continue

        return models

    def delete_model(self, name: str, tag: str) -> None:
        """Remove a manifest file.  Orphaned blobs are cleaned up by :meth:`gc_blobs`."""
        path = self._manifest_path(name, tag)
        path.unlink(missing_ok=True)


        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    def gc_blobs(self) -> int:
        """Delete blobs not referenced by any manifest.  Returns the number removed."""
        if not self.blobs_dir.is_dir():
            return 0


        referenced: set[str] = set()
        for model_info in self.list_models():

            manifest = self.resolve_model(model_info.name, model_info.tag)
            if manifest is not None:
                for layer in manifest.layers:
                    referenced.add(layer.digest)

        removed = 0
        now = time.time()
        for blob in self.blobs_dir.iterdir():
            if blob.name.startswith("sha256-") and blob.name not in referenced:
                blob.unlink(missing_ok=True)
                removed += 1
            elif blob.suffix == ".tmp":

                try:
                    age = now - blob.stat().st_mtime
                    if age > 3600:
                        blob.unlink()
                        removed += 1
                except OSError:
                    pass
        return removed
