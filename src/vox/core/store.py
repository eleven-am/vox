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

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from vox.core.types import ModelInfo

logger = logging.getLogger(__name__)
_DIGEST_PATTERN = re.compile(r"^sha256-[0-9a-f]{64}$")
_STORE_WRITER: ContextVar[StoreWriterLease | None] = ContextVar(
    "vox_store_writer",
    default=None,
)
_PROCESS_WRITER_LOCKS: dict[Path, threading.Lock] = {}
_PROCESS_WRITER_LOCKS_LOCK = threading.Lock()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    transaction_id: str | None = None


class StoreWriterBusyError(RuntimeError):
    pass


class StoreWriterLease:
    def __init__(
        self,
        *,
        root: Path,
        descriptor: int,
        epoch_path: Path,
        epoch: int,
        process_lock: threading.Lock,
    ) -> None:
        self.root = root
        self.epoch = epoch
        self._descriptor = descriptor
        self._epoch_path = epoch_path
        self._process_lock = process_lock
        self._closed = False

    def assert_current(self) -> None:
        if self._closed:
            raise RuntimeError("store writer lease is closed")
        try:
            current = int(self._epoch_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("store writer fencing epoch is unreadable") from exc
        if current != self.epoch:
            raise RuntimeError(f"store writer lease was fenced: expected epoch {self.epoch}, found {current}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._process_lock.release()


class BlobLease:
    def __init__(self, store: BlobStore) -> None:
        self._store = store
        self._digests: set[str] = set()
        self._closed = False

    def write_blob(self, data: BinaryIO) -> str:
        if self._closed:
            raise RuntimeError("blob lease is closed")
        return self._store._write_blob(data, lease=self)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._store._release_blob_lease(self)

    def abort(self) -> int:
        if self._closed:
            return 0
        digests = tuple(self._digests)
        self.close()
        return self._store.gc_blobs(candidates=digests, grace_seconds=0)


def _manifest_to_dict(m: Manifest) -> dict[str, Any]:
    return {
        "schema_version": m.schema_version,
        "layers": [asdict(layer) for layer in m.layers],
        "config": m.config,
        "transaction_id": m.transaction_id,
    }


def _manifest_from_dict(d: dict[str, Any]) -> Manifest:
    return Manifest(
        schema_version=d.get("schema_version", 1),
        layers=[ManifestLayer(**layer) for layer in d.get("layers", [])],
        config=d.get("config", {}),
        transaction_id=d.get("transaction_id"),
    )


_READ_CHUNK = 1 << 20
_BLOB_GC_GRACE_SECONDS = 3600


class BlobStore:
    """Content-addressable blob store with JSON manifests."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path.home() / ".vox"
        self._blob_lock = threading.RLock()
        self._leased_blobs: dict[str, int] = {}

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
    def manifest_staging_dir(self) -> Path:
        return self._root / "models" / "manifest-staging"

    @property
    def voices_dir(self) -> Path:
        return self._root / "voices"

    @property
    def transaction_root(self) -> Path:
        return self._root / ".transactions"

    def current_writer_lease(self) -> StoreWriterLease | None:
        lease = _STORE_WRITER.get()
        if lease is None:
            return None
        if lease.root != self._root.absolute():
            raise RuntimeError("a store writer lease cannot mutate a different store")
        lease.assert_current()
        return lease

    def acquire_writer_lease(self, *, timeout: float = 30.0) -> StoreWriterLease:
        root = self._root.absolute()
        root.mkdir(parents=True, exist_ok=True)
        transaction_root = root / ".transactions"
        transaction_root.mkdir(parents=True, exist_ok=True)
        with _PROCESS_WRITER_LOCKS_LOCK:
            process_lock = _PROCESS_WRITER_LOCKS.setdefault(root, threading.Lock())
        if not process_lock.acquire(timeout=max(0.0, timeout)):
            raise StoreWriterBusyError(f"store writer is busy: {root}")

        descriptor = -1
        try:
            lock_path = transaction_root / "store-writer.lock"
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(lock_path, flags, 0o600)
            deadline = time.monotonic() + max(0.0, timeout)
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise StoreWriterBusyError(f"store writer is busy: {root}") from exc
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

            epoch_path = transaction_root / "store-writer.epoch"
            try:
                epoch = int(epoch_path.read_text(encoding="utf-8")) + 1
            except FileNotFoundError:
                epoch = 1
            except (OSError, ValueError) as exc:
                raise RuntimeError("store writer fencing epoch is invalid") from exc
            _write_text(epoch_path, str(epoch))
            return StoreWriterLease(
                root=root,
                descriptor=descriptor,
                epoch_path=epoch_path,
                epoch=epoch,
                process_lock=process_lock,
            )
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            process_lock.release()
            raise

    @contextmanager
    def bind_writer_lease(self, lease: StoreWriterLease) -> Iterator[StoreWriterLease]:
        if lease.root != self._root.absolute():
            raise RuntimeError("a store writer lease cannot bind to a different store")
        lease.assert_current()
        token = _STORE_WRITER.set(lease)
        try:
            yield lease
        finally:
            _STORE_WRITER.reset(token)

    @contextmanager
    def writer_lease(self, *, timeout: float = 30.0) -> Iterator[StoreWriterLease]:
        current = self.current_writer_lease()
        if current is not None:
            yield current
            return
        lease = self.acquire_writer_lease(timeout=timeout)
        with self.bind_writer_lease(lease):
            try:
                yield lease
            finally:
                lease.close()

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
        return self._write_blob(data, lease=None)

    def _write_blob(self, data: BinaryIO, *, lease: BlobLease | None) -> str:
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()

        with tempfile.NamedTemporaryFile(
            dir=self.blobs_dir,
            delete=False,
            suffix=".tmp",
        ) as fd:
            tmp_path = Path(fd.name)
            try:
                while True:
                    chunk = data.read(_READ_CHUNK)
                    if not chunk:
                        break
                    h.update(chunk)
                    fd.write(chunk)
                fd.flush()
                os.fsync(fd.fileno())
            except BaseException:
                fd.close()
                tmp_path.unlink(missing_ok=True)
                raise

        try:
            digest = f"sha256-{h.hexdigest()}"
            final_path = self.get_blob_path(digest)

            with self.writer_lease(), self._blob_lock:
                if final_path.exists():
                    tmp_path.unlink(missing_ok=True)
                else:
                    tmp_path.rename(final_path)
                    _sync_directory(self.blobs_dir)
                if lease is not None and digest not in lease._digests:
                    lease._digests.add(digest)
                    self._leased_blobs[digest] = self._leased_blobs.get(digest, 0) + 1

            return digest
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @contextmanager
    def blob_lease(self) -> Iterator[BlobLease]:
        lease = self.acquire_blob_lease()
        try:
            yield lease
        finally:
            lease.close()

    def acquire_blob_lease(self) -> BlobLease:
        return BlobLease(self)

    def _release_blob_lease(self, lease: BlobLease) -> None:
        with self._blob_lock:
            for digest in lease._digests:
                remaining = self._leased_blobs.get(digest, 0) - 1
                if remaining > 0:
                    self._leased_blobs[digest] = remaining
                else:
                    self._leased_blobs.pop(digest, None)
            lease._digests.clear()

    def _manifest_path(self, name: str, tag: str) -> Path:
        return self.manifests_dir / name / tag

    def resolve_model(self, name: str, tag: str = "latest") -> Manifest | None:
        """Read and parse a manifest file, returning ``None`` if it does not exist."""
        path = self._manifest_path(name, tag)
        if not path.is_file():
            return None
        with open(path) as f:
            return _manifest_from_dict(json.load(f))

    def save_manifest(self, name: str, tag: str, manifest: Manifest) -> None:
        """Atomically write a manifest JSON file."""
        with self.writer_lease():
            path = self._manifest_path(name, tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_staging_dir.mkdir(parents=True, exist_ok=True)

            fd, tmp_name = tempfile.mkstemp(
                dir=self.manifest_staging_dir,
                prefix=f"{path.name}.",
                suffix=".tmp",
            )
            tmp = Path(tmp_name)
            try:
                with open(fd, "w") as f:
                    json.dump(_manifest_to_dict(manifest), f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
                _sync_directory(path.parent)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

    def prune_manifest_staging(self) -> int:
        with self.writer_lease():
            root = self.manifest_staging_dir
            if not root.is_dir():
                return 0
            removed = 0
            for path in tuple(root.iterdir()):
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                    removed += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    removed += 1
            if removed:
                _sync_directory(root)
            return removed

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
                    models.append(ModelInfo.from_manifest_config(name, tag, cfg, size_bytes=size))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping corrupted manifest {tag_file}: {e}")
                    continue

        return models

    def delete_model(self, name: str, tag: str) -> None:
        """Remove a manifest file.  Orphaned blobs are cleaned up by :meth:`gc_blobs`."""
        with self.writer_lease():
            path = self._manifest_path(name, tag)
            path.unlink(missing_ok=True)

            parent = path.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                _sync_directory(parent.parent)
            elif parent.is_dir():
                _sync_directory(parent)

    def gc_blobs(
        self,
        *,
        candidates: Iterable[str] | None = None,
        grace_seconds: float = _BLOB_GC_GRACE_SECONDS,
    ) -> int:
        """Delete blobs not referenced by any manifest.  Returns the number removed."""
        with self.writer_lease(), self._blob_lock:
            if not self.blobs_dir.is_dir():
                return 0

            referenced: set[str] = set(self._leased_blobs)
            referenced.update(self._journal_blob_roots())
            manifest_roots = self._manifest_blob_roots()
            if manifest_roots is None:
                return 0
            referenced.update(manifest_roots)

            removed = 0
            now = time.time()
            blobs = (
                tuple(self.get_blob_path(digest) for digest in candidates)
                if candidates is not None
                else tuple(self.blobs_dir.iterdir())
            )
            for blob in blobs:
                if not blob.exists():
                    continue
                if blob.name.startswith("sha256-") and blob.name not in referenced:
                    try:
                        age = now - blob.stat().st_mtime
                    except OSError:
                        continue
                    if age < grace_seconds:
                        continue
                    blob.unlink(missing_ok=True)
                    removed += 1
                elif candidates is None and blob.suffix == ".tmp":
                    try:
                        age = now - blob.stat().st_mtime
                        if age > 3600:
                            blob.unlink()
                            removed += 1
                    except OSError:
                        pass
            return removed

    def _manifest_blob_roots(self) -> set[str] | None:
        try:
            manifest_root = self.manifests_dir.lstat()
        except FileNotFoundError:
            return set()
        except OSError as exc:
            logger.warning(
                "Blob collection skipped because manifest root is unreadable: %s: %s",
                self.manifests_dir,
                exc,
            )
            return None
        if not stat.S_ISDIR(manifest_root.st_mode):
            logger.warning(
                "Blob collection skipped because manifest root is not a directory: %s",
                self.manifests_dir,
            )
            return None
        digests: set[str] = set()
        for model_dir in sorted(self.manifests_dir.iterdir()):
            if model_dir.is_symlink():
                logger.warning("Blob collection skipped because manifest directory is a symlink: %s", model_dir)
                return None
            if not model_dir.is_dir():
                continue
            for tag_file in sorted(model_dir.iterdir()):
                if tag_file.is_symlink():
                    logger.warning("Blob collection skipped because manifest is a symlink: %s", tag_file)
                    return None
                if not tag_file.is_file():
                    continue
                try:
                    with tag_file.open(encoding="utf-8") as stream:
                        manifest = _manifest_from_dict(json.load(stream))
                except (
                    OSError,
                    json.JSONDecodeError,
                    AttributeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    logger.warning(
                        "Blob collection skipped because manifest is unreadable: %s: %s",
                        tag_file,
                        exc,
                    )
                    return None
                digests.update(layer.digest for layer in manifest.layers)
        return digests

    def _journal_blob_roots(self) -> set[str]:
        root = self.transaction_root / "pulls"
        if not root.is_dir():
            return set()
        digests: set[str] = set()
        for path in root.glob("*.json"):
            try:
                with path.open(encoding="utf-8") as stream:
                    payload = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot collect blobs while pull journal is unreadable: {path}") from exc
            values = list(payload.get("candidate_digests", ()))
            previous = payload.get("previous_manifest")
            if isinstance(previous, dict):
                for layer in previous.get("layers", ()):
                    if isinstance(layer, dict):
                        values.append(layer.get("digest"))
            for value in values:
                if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
                    raise RuntimeError(f"cannot collect blobs with invalid pull journal digest: {path}")
                digests.add(value)
        return digests


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
