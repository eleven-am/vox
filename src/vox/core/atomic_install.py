from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_INSTALL_DIRECTORY = re.compile(r"^\.(?P<target>.+)\.(?P<state>installing|previous|failed|committed)-.+$")
_INSTALL_TRANSACTION: ContextVar[Any | None] = ContextVar(
    "vox_install_transaction",
    default=None,
)


@dataclass
class DirectorySwap:
    target: Path
    backup: Path | None

    def commit(self) -> None:
        backup = self.backup
        if backup is not None:
            committed = backup.with_name(backup.name.replace(".previous-", ".committed-", 1))
            backup.rename(committed)
            _sync_directory(committed.parent)
            self.backup = None
            try:
                shutil.rmtree(committed)
            except OSError as exc:
                logger.warning(
                    "Committed directory backup cleanup deferred path=%s error=%s",
                    committed,
                    exc,
                )
            else:
                _sync_directory(committed.parent)

    def rollback(self) -> None:
        displaced: Path | None = None
        if self.target.exists() or self.target.is_symlink():
            displaced = _empty_temp_path(
                self.target.parent,
                f".{self.target.name}.failed-",
            )
            self.target.rename(displaced)
            _sync_directory(self.target.parent)
        backup = self.backup
        if backup is not None:
            backup.rename(self.target)
            _sync_directory(self.target.parent)
            self.backup = None
        if displaced is not None:
            try:
                _remove_path(displaced)
            except OSError as exc:
                logger.warning("Rolled-back directory cleanup deferred path=%s error=%s", displaced, exc)


@contextmanager
def bind_install_transaction(transaction: Any) -> Iterator[None]:
    token = _INSTALL_TRANSACTION.set(transaction)
    try:
        yield
    finally:
        _INSTALL_TRANSACTION.reset(token)


@contextmanager
def staged_directory(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.installing-",
        )
    )
    try:
        yield stage
    finally:
        if stage.exists() or stage.is_symlink():
            shutil.rmtree(stage, ignore_errors=True)


def publish_staged_directory(
    stage: Path,
    target: Path,
    *,
    preserve_existing: bool,
    retain_backup: bool = False,
) -> DirectorySwap:
    if preserve_existing and target.is_dir():
        merge_missing_tree(target, stage)

    transaction = _INSTALL_TRANSACTION.get()
    if transaction is not None:
        retain_backup = True
    backup: Path | None = None
    had_target = target.exists() or target.is_symlink()
    if had_target:
        backup = _empty_temp_path(
            target.parent,
            f".{target.name}.previous-",
        )
    if transaction is not None:
        transaction.record_swap(
            stage=stage,
            target=target,
            backup=backup,
        )
    if backup is not None:
        target.rename(backup)
        _sync_directory(target.parent)
    try:
        stage.rename(target)
        _sync_directory(target.parent)
    except BaseException:
        if backup is not None:
            backup.rename(target)
            _sync_directory(target.parent)
        raise
    swap = DirectorySwap(target=target, backup=backup)
    if not retain_backup:
        swap.commit()
    return swap


def prune_stale_install_directories(root: Path) -> int:
    if not root.is_dir():
        return 0
    recovered = 0
    previous: dict[str, list[Path]] = {}
    for path in tuple(root.iterdir()):
        match = _INSTALL_DIRECTORY.fullmatch(path.name)
        if path.is_symlink() or not path.is_dir() or match is None:
            continue
        if match.group("state") == "previous":
            previous.setdefault(match.group("target"), []).append(path)
            continue
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.warning("Stale installation directory cleanup deferred path=%s error=%s", path, exc)
        else:
            recovered += 1
    for target_name, backups in previous.items():
        authoritative = max(
            backups,
            key=lambda path: path.stat().st_mtime_ns,
        )
        target = root / target_name
        displaced: Path | None = None
        try:
            if target.exists() or target.is_symlink():
                displaced = _empty_temp_path(
                    root,
                    f".{target_name}.failed-",
                )
                target.rename(displaced)
            authoritative.rename(target)
            recovered += 1
            if displaced is not None:
                _remove_path(displaced)
            for backup in backups:
                if backup == authoritative:
                    continue
                _remove_path(backup)
                recovered += 1
        except OSError as exc:
            if displaced is not None and not target.exists() and displaced.exists():
                displaced.rename(target)
            logger.warning(
                "Interrupted installation recovery deferred path=%s error=%s",
                authoritative,
                exc,
            )
    return recovered


def merge_missing_tree(
    source: Path,
    destination: Path,
    *,
    excluded: Iterable[Path] = (),
) -> None:
    excluded_paths = tuple(excluded)
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if any(relative == path or path in relative.parents for path in excluded_paths):
            continue
        target = destination / relative
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.is_symlink():
            target.symlink_to(os.readlink(item), target_is_directory=item.is_dir())
        elif item.is_dir():
            target.mkdir()
        else:
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _empty_temp_path(parent: Path, prefix: str) -> Path:
    path = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
    path.rmdir()
    return path


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
