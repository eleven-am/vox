from __future__ import annotations

import errno
import os
from pathlib import Path

from vox.core.temp_storage import (
    DEFAULT_STALE_TEMP_AGE_SECONDS,
    prune_stale_temp_dirs,
    stale_temp_age_seconds,
)


def test_prune_stale_temp_dirs_only_removes_old_directories(tmp_path: Path) -> None:
    now = 10_000.0
    old_dir = tmp_path / "old-request"
    old_dir.mkdir()
    (old_dir / ".nfs-old").write_text("stale")
    os.utime(old_dir, (now - 120, now - 120))

    fresh_dir = tmp_path / "active-request"
    fresh_dir.mkdir()
    os.utime(fresh_dir, (now - 10, now - 10))

    regular_file = tmp_path / "keep.txt"
    regular_file.write_text("keep")

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked"
    symlink.symlink_to(target, target_is_directory=True)

    removed = prune_stale_temp_dirs(tmp_path, max_age_seconds=60, now=now)

    assert removed == [old_dir]
    assert old_dir.exists() is False
    assert fresh_dir.is_dir()
    assert regular_file.is_file()
    assert symlink.is_symlink()
    assert target.is_dir()


def test_prune_stale_temp_dirs_does_not_raise_when_nfs_entry_is_busy(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()

    def busy(_path: Path) -> None:
        raise OSError(errno.EBUSY, "Resource busy")

    monkeypatch.setattr("vox.core.temp_storage.shutil.rmtree", busy)

    assert prune_stale_temp_dirs(tmp_path, max_age_seconds=0) == []
    assert stale_dir.is_dir()
    assert "Unable to remove stale Vox temporary directory" in caplog.text


def test_stale_temp_age_uses_safe_default_for_invalid_environment(monkeypatch, caplog) -> None:
    monkeypatch.setenv("VOX_STALE_TEMP_AGE_SECONDS", "not-a-number")

    assert stale_temp_age_seconds() == DEFAULT_STALE_TEMP_AGE_SECONDS
    assert "Invalid VOX_STALE_TEMP_AGE_SECONDS" in caplog.text
