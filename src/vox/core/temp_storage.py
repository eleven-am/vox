from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STALE_TEMP_AGE_SECONDS = 24 * 60 * 60
DEFAULT_VOX_TEMP_ROOT = Path("/tmp/vox")


def vox_temp_root() -> Path:
    configured = os.environ.get("VOX_TEMP_ROOT", "").strip()
    return Path(configured) if configured else DEFAULT_VOX_TEMP_ROOT


def stale_temp_age_seconds() -> int:
    raw = os.environ.get("VOX_STALE_TEMP_AGE_SECONDS", "").strip()
    if not raw:
        return DEFAULT_STALE_TEMP_AGE_SECONDS
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid VOX_STALE_TEMP_AGE_SECONDS=%r; using %d seconds",
            raw,
            DEFAULT_STALE_TEMP_AGE_SECONDS,
        )
        return DEFAULT_STALE_TEMP_AGE_SECONDS


def prune_stale_temp_dirs(
    temp_root: Path,
    *,
    max_age_seconds: int | None = None,
    now: float | None = None,
) -> list[Path]:
    if not temp_root.is_dir():
        return []

    cutoff = (time.time() if now is None else now) - (
        stale_temp_age_seconds() if max_age_seconds is None else max(0, max_age_seconds)
    )
    removed: list[Path] = []
    for candidate in temp_root.iterdir():
        try:
            if candidate.lstat().st_mtime > cutoff:
                continue
            if candidate.is_symlink() or candidate.is_file():
                candidate.unlink()
            elif candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                continue
            removed.append(candidate)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Unable to remove stale Vox temporary entry %s: %s", candidate, exc)

    if removed:
        logger.info("Removed %d stale Vox temporary entries from %s", len(removed), temp_root)
    return removed
