from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vox.core.store import BlobStore, Manifest, ManifestLayer

logger = logging.getLogger(__name__)
_JOURNAL_VERSION = 3
_DIGEST_PATTERN = re.compile(r"^sha256-[0-9a-f]{64}$")
_TRANSACTION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_MODEL_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SWAP_ROOTS = frozenset({"adapters", "runtime"})


class PullTransaction:
    def __init__(self, *, store: BlobStore, path: Path, payload: dict[str, Any]) -> None:
        self._store = store
        self._path = path
        self._payload = payload
        self._lock = threading.Lock()

    @classmethod
    def begin(
        cls,
        *,
        store: BlobStore,
        name: str,
        tag: str,
        previous_manifest: Manifest | None,
        candidate_digests: tuple[str, ...],
    ) -> PullTransaction:
        writer = store.current_writer_lease()
        if writer is None:
            raise RuntimeError("pull transaction requires the store writer lease")
        _validate_component(name, "model name")
        _validate_component(tag, "model tag")
        _validate_digests(candidate_digests)
        root = _transaction_root(store)
        root.mkdir(parents=True, exist_ok=True)
        for existing in root.glob("*.json"):
            payload = _read_payload(store, existing)
            if payload["state"] == "preparing":
                raise RuntimeError(f"active pull transaction already exists: {payload['id']}")

        transaction_id = uuid.uuid4().hex
        path = root / f"{transaction_id}.json"
        payload = {
            "version": _JOURNAL_VERSION,
            "id": transaction_id,
            "writer_epoch": writer.epoch,
            "state": "preparing",
            "name": name,
            "tag": tag,
            "previous_manifest": asdict(previous_manifest) if previous_manifest is not None else None,
            "candidate_manifest": None,
            "candidate_digests": list(candidate_digests),
            "swaps": [],
        }
        _validate_payload(store, path, payload)
        transaction = cls(store=store, path=path, payload=payload)
        transaction._persist()
        return transaction

    @property
    def id(self) -> str:
        return str(self._payload["id"])

    @property
    def path(self) -> Path:
        return self._path

    @property
    def committed(self) -> bool:
        return self._payload["state"] == "committed"

    def record_swap(
        self,
        *,
        stage: Path,
        target: Path,
        backup: Path | None,
    ) -> None:
        with self._lock:
            self._assert_writer()
            if self._payload["state"] != "preparing":
                raise RuntimeError("cannot add a directory swap after pull commit")
            swap = {
                "stage": _relative_path(self._store.root, stage),
                "target": _relative_path(self._store.root, target),
                "backup": _relative_path(self._store.root, backup) if backup is not None else None,
            }
            _validate_swap(self._store.root, swap)
            payload = {**self._payload, "swaps": [*self._payload["swaps"], swap]}
            self._persist(payload)

    def record_candidate_manifest(self, manifest: Manifest) -> None:
        with self._lock:
            self._assert_writer()
            if self._payload["state"] != "preparing":
                raise RuntimeError("cannot record a candidate manifest after pull commit")
            if manifest.transaction_id != self.id:
                raise RuntimeError("candidate manifest does not belong to this pull transaction")
            self._persist(
                {
                    **self._payload,
                    "candidate_manifest": asdict(manifest),
                }
            )

    def mark_committed(self) -> None:
        with self._lock:
            self._assert_writer()
            if self._payload["state"] == "committed":
                return
            candidate = self._payload["candidate_manifest"]
            if candidate is None:
                manifest = self._store.resolve_model(
                    str(self._payload["name"]),
                    str(self._payload["tag"]),
                )
                if manifest is None or manifest.transaction_id != self.id:
                    raise RuntimeError("pull transaction has no published candidate manifest")
                candidate = asdict(manifest)
            self._persist(
                {
                    **self._payload,
                    "candidate_manifest": candidate,
                    "state": "committed",
                }
            )

    def owns_canonical_manifest(self) -> bool:
        self._assert_writer()
        manifest = self._store.resolve_model(
            str(self._payload["name"]),
            str(self._payload["tag"]),
        )
        return manifest is not None and manifest.transaction_id == self.id

    def rollback(self) -> None:
        self._assert_writer()
        if self.owns_canonical_manifest():
            self.mark_committed()
            _recover_transaction(self._store, self._path, self._payload, committed=True)
            return
        _recover_transaction(self._store, self._path, self._payload, committed=False)

    def finish(self) -> None:
        self._assert_writer()
        if self._payload["state"] != "committed":
            raise RuntimeError("cannot finish an uncommitted pull transaction")
        _recover_transaction(self._store, self._path, self._payload, committed=True)

    def _assert_writer(self) -> None:
        writer = self._store.current_writer_lease()
        if writer is None:
            raise RuntimeError("pull transaction lost the store writer lease")
        if writer.epoch != self._payload["writer_epoch"]:
            raise RuntimeError("pull transaction was fenced by a newer store writer")

    def _persist(self, payload: dict[str, Any] | None = None) -> None:
        next_payload = payload or self._payload
        _validate_payload(self._store, self._path, next_payload)
        _write_json(self._path, next_payload)
        self._payload = next_payload


def recover_pull_transactions(store: BlobStore, *, timeout: float = 30.0) -> int:
    with store.writer_lease(timeout=timeout):
        return _recover_pull_transactions_locked(store)


def _recover_pull_transactions_locked(store: BlobStore) -> int:
    root = _transaction_root(store)
    if not root.is_dir():
        return 0
    recovered = 0
    removed_temporary = False
    for path in root.glob(".*.tmp"):
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
            removed_temporary = True
    if removed_temporary:
        _sync_directory(root)
    for path in sorted(root.glob("*.json")):
        payload = _read_payload(store, path)
        manifest = store.resolve_model(str(payload["name"]), str(payload["tag"]))
        owns_manifest = manifest is not None and manifest.transaction_id == payload["id"]
        if payload["state"] == "preparing" and not owns_manifest:
            previous = payload.get("previous_manifest")
            expected = _stored_manifest(previous) if previous is not None else None
            if manifest != expected:
                raise RuntimeError(f"preparing pull transaction {payload['id']} does not own the canonical manifest")
        committed = payload["state"] == "committed" or owns_manifest
        _recover_transaction(
            store,
            path,
            payload,
            committed=committed,
            promote_stage=committed,
        )
        recovered += 1
    _remove_empty_parents(root, store.root)
    return recovered


def _recover_transaction(
    store: BlobStore,
    path: Path,
    payload: dict[str, Any],
    *,
    committed: bool,
    promote_stage: bool = True,
) -> None:
    _validate_payload(store, path, payload)
    swaps = tuple(payload["swaps"])
    if committed:
        for swap in swaps:
            _roll_forward_swap(store.root, swap, promote_stage=promote_stage)
        candidate = payload["candidate_manifest"]
        if candidate is None:
            raise RuntimeError(f"committed pull transaction {payload['id']} has no candidate manifest")
        store.save_manifest(
            str(payload["name"]),
            str(payload["tag"]),
            _stored_manifest(candidate),
        )
    else:
        for swap in reversed(swaps):
            _roll_back_swap(store.root, swap)
        _restore_manifest(store, payload)

    candidates = tuple(str(digest) for digest in payload["candidate_digests"])
    path.unlink(missing_ok=True)
    _sync_directory(path.parent)
    store.gc_blobs(candidates=candidates, grace_seconds=0)
    _remove_empty_parents(path.parent, store.root)


def _roll_forward_swap(
    root: Path,
    swap: dict[str, Any],
    *,
    promote_stage: bool,
) -> None:
    stage, target, backup = _swap_paths(root, swap)
    if not target.exists() and not target.is_symlink():
        if promote_stage and (stage.exists() or stage.is_symlink()):
            stage.rename(target)
            _sync_directory(target.parent)
        elif promote_stage:
            raise RuntimeError(f"committed pull target is missing: {target}")
    if backup is not None:
        committed = backup.with_name(backup.name.replace(".previous-", ".committed-", 1))
        _remove_path(backup)
        _remove_path(committed)
    _remove_path(stage)


def _roll_back_swap(root: Path, swap: dict[str, Any]) -> None:
    stage, target, backup = _swap_paths(root, swap)
    if backup is None:
        if not stage.exists() and not stage.is_symlink():
            _remove_path(target)
        _remove_path(stage)
        return
    if backup.exists() or backup.is_symlink():
        displaced = _empty_path(target.parent, f".{target.name}.failed-")
        if target.exists() or target.is_symlink():
            target.rename(displaced)
            _sync_directory(target.parent)
        backup.rename(target)
        _sync_directory(target.parent)
        _remove_path(displaced)
    _remove_path(stage)


def _restore_manifest(store: BlobStore, payload: dict[str, Any]) -> None:
    previous = payload["previous_manifest"]
    name = str(payload["name"])
    tag = str(payload["tag"])
    if previous is None:
        store.delete_model(name, tag)
        return
    store.save_manifest(name, tag, _stored_manifest(previous))


def _stored_manifest(payload: dict[str, Any]) -> Manifest:
    return Manifest(
        schema_version=int(payload["schema_version"]),
        layers=[ManifestLayer(**layer) for layer in payload["layers"]],
        config=dict(payload["config"]),
        transaction_id=payload.get("transaction_id"),
    )


def _transaction_root(store: BlobStore) -> Path:
    return store.transaction_root / "pulls"


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.absolute().relative_to(root.absolute()))
    except ValueError as exc:
        raise ValueError(f"invalid pull transaction path: {path}") from exc


def _swap_paths(
    root: Path,
    swap: dict[str, Any],
) -> tuple[Path, Path, Path | None]:
    _validate_swap(root, swap)
    stage = _stored_path(root, swap["stage"])
    target = _stored_path(root, swap["target"])
    backup_value = swap["backup"]
    backup = _stored_path(root, backup_value) if backup_value is not None else None
    return stage, target, backup


def _stored_path(root: Path, value: str) -> Path:
    relative = Path(value)
    path = root / relative
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid pull transaction path: {value}") from exc
    return path


def _validate_payload(
    store: BlobStore,
    path: Path,
    payload: dict[str, Any],
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"invalid pull transaction journal: {path}")
    required = {
        "version",
        "id",
        "writer_epoch",
        "state",
        "name",
        "tag",
        "previous_manifest",
        "candidate_manifest",
        "candidate_digests",
        "swaps",
    }
    if set(payload) != required:
        raise ValueError(f"invalid pull transaction schema: {path}")
    if payload["version"] != _JOURNAL_VERSION:
        raise ValueError(f"unsupported pull transaction version: {payload['version']}")
    transaction_id = payload["id"]
    if not isinstance(transaction_id, str) or _TRANSACTION_PATTERN.fullmatch(transaction_id) is None:
        raise ValueError(f"invalid pull transaction id: {transaction_id}")
    if path.name != f"{transaction_id}.json":
        raise ValueError(f"pull transaction id does not match journal path: {path}")
    epoch = payload["writer_epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError(f"invalid pull transaction writer epoch: {epoch}")
    if payload["state"] not in {"preparing", "committed"}:
        raise ValueError(f"invalid pull transaction state: {payload['state']}")
    _validate_component(payload["name"], "model name")
    _validate_component(payload["tag"], "model tag")
    candidate_digests = payload["candidate_digests"]
    if not isinstance(candidate_digests, list):
        raise ValueError("invalid pull transaction candidate digests")
    _validate_digests(tuple(candidate_digests))
    previous = payload["previous_manifest"]
    if previous is not None:
        _validate_manifest(previous)
    candidate = payload["candidate_manifest"]
    if candidate is not None:
        _validate_manifest(candidate)
        if candidate.get("transaction_id") != transaction_id:
            raise ValueError("candidate manifest does not belong to the pull transaction")
        manifest_digests = tuple(layer["digest"] for layer in candidate["layers"])
        if manifest_digests != tuple(candidate_digests):
            raise ValueError("candidate manifest digests do not match the pull transaction")
    if payload["state"] == "committed" and candidate is None:
        raise ValueError("committed pull transaction has no candidate manifest")
    swaps = payload["swaps"]
    if not isinstance(swaps, list):
        raise ValueError("invalid pull transaction swaps")
    for swap in swaps:
        _validate_swap(store.root, swap)


def _validate_manifest(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("invalid pull transaction manifest")
    if not isinstance(payload.get("schema_version"), int):
        raise ValueError("invalid pull transaction manifest schema")
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise ValueError("invalid pull transaction manifest layers")
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("invalid pull transaction manifest layer")
        digest = layer.get("digest")
        if not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"invalid pull transaction digest: {digest}")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("invalid pull transaction manifest config")
    transaction_id = payload.get("transaction_id")
    if transaction_id is not None and (
        not isinstance(transaction_id, str) or _TRANSACTION_PATTERN.fullmatch(transaction_id) is None
    ):
        raise ValueError(f"invalid manifest transaction id: {transaction_id}")


def _validate_digests(values: tuple[Any, ...]) -> None:
    for digest in values:
        if not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"invalid pull transaction digest: {digest}")


def _validate_component(value: Any, label: str) -> None:
    if not isinstance(value, str) or _MODEL_COMPONENT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"invalid pull transaction {label}: {value}")


def _validate_swap(root: Path, swap: Any) -> None:
    if not isinstance(swap, dict) or set(swap) != {"stage", "target", "backup"}:
        raise ValueError("invalid pull transaction swap")
    stage_value = swap["stage"]
    target_value = swap["target"]
    backup_value = swap["backup"]
    if not isinstance(stage_value, str) or not isinstance(target_value, str):
        raise ValueError("invalid pull transaction swap path")
    if backup_value is not None and not isinstance(backup_value, str):
        raise ValueError("invalid pull transaction backup path")

    stage_relative = Path(stage_value)
    target_relative = Path(target_value)
    if (
        stage_relative.is_absolute()
        or target_relative.is_absolute()
        or ".." in stage_relative.parts
        or ".." in target_relative.parts
        or len(target_relative.parts) != 2
        or target_relative.parts[0] not in _SWAP_ROOTS
        or stage_relative.parent != target_relative.parent
        or not stage_relative.name.startswith(f".{target_relative.name}.installing-")
    ):
        raise ValueError("invalid pull transaction swap path")
    if backup_value is not None:
        backup_relative = Path(backup_value)
        if (
            backup_relative.is_absolute()
            or ".." in backup_relative.parts
            or backup_relative.parent != target_relative.parent
            or not backup_relative.name.startswith(f".{target_relative.name}.previous-")
        ):
            raise ValueError("invalid pull transaction backup path")
    _stored_path(root, stage_value)
    _stored_path(root, target_value)
    if backup_value is not None:
        _stored_path(root, backup_value)


def _read_payload(store: BlobStore, path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"pull transaction journal is unreadable: {path}") from exc
    _validate_payload(store, path, payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        _sync_directory(path.parent)
    elif path.is_dir():
        shutil.rmtree(path)
        _sync_directory(path.parent)


def _empty_path(parent: Path, prefix: str) -> Path:
    path = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
    path.rmdir()
    return path


def _remove_empty_parents(path: Path, stop: Path) -> None:
    current = path
    while current != stop and current.is_dir() and not any(current.iterdir()):
        parent = current.parent
        current.rmdir()
        _sync_directory(parent)
        current = parent
