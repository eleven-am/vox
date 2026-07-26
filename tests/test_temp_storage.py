from __future__ import annotations

import errno
import os
from pathlib import Path

from vox.core.temp_storage import (
    DEFAULT_STALE_TEMP_AGE_SECONDS,
    DEFAULT_VOX_TEMP_ROOT,
    prune_stale_temp_dirs,
    stale_temp_age_seconds,
    vox_temp_root,
)


def test_prune_stale_temp_dirs_removes_old_owned_entries(tmp_path: Path) -> None:
    now = 10_000.0
    old_dir = tmp_path / "old-request"
    old_dir.mkdir()
    (old_dir / ".nfs-old").write_text("stale")
    os.utime(old_dir, (now - 120, now - 120))

    fresh_dir = tmp_path / "active-request"
    fresh_dir.mkdir()
    os.utime(fresh_dir, (now - 10, now - 10))

    stale_file = tmp_path / "orphan.wav"
    stale_file.write_bytes(b"stale")
    os.utime(stale_file, (now - 120, now - 120))

    fresh_file = tmp_path / "active.wav"
    fresh_file.write_bytes(b"active")
    os.utime(fresh_file, (now - 10, now - 10))

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked"
    symlink.symlink_to(target, target_is_directory=True)

    removed = prune_stale_temp_dirs(tmp_path, max_age_seconds=60, now=now)

    assert set(removed) == {old_dir, stale_file}
    assert old_dir.exists() is False
    assert fresh_dir.is_dir()
    assert stale_file.exists() is False
    assert fresh_file.is_file()
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
    assert "Unable to remove stale Vox temporary entry" in caplog.text


def test_stale_temp_age_uses_safe_default_for_invalid_environment(monkeypatch, caplog) -> None:
    monkeypatch.setenv("VOX_STALE_TEMP_AGE_SECONDS", "not-a-number")

    assert stale_temp_age_seconds() == DEFAULT_STALE_TEMP_AGE_SECONDS
    assert "Invalid VOX_STALE_TEMP_AGE_SECONDS" in caplog.text


def test_vox_temp_root_does_not_inherit_ambient_tmpdir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "operating-system-temp"))
    monkeypatch.delenv("VOX_TEMP_ROOT", raising=False)

    assert vox_temp_root() == DEFAULT_VOX_TEMP_ROOT


def test_vox_temp_root_accepts_explicit_owned_directory(tmp_path: Path, monkeypatch) -> None:
    owned = tmp_path / "vox-owned"
    monkeypatch.setenv("VOX_TEMP_ROOT", str(owned))

    assert vox_temp_root() == owned
