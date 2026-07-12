from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STALE_TEMP_AGE_SECONDS = 24 * 60 * 60


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
            if candidate.is_symlink() or not candidate.is_dir() or candidate.stat().st_mtime > cutoff:
                continue
            shutil.rmtree(candidate)
            removed.append(candidate)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Unable to remove stale Vox temporary directory %s: %s", candidate, exc)

    if removed:
        logger.info("Removed %d stale Vox temporary directories from %s", len(removed), temp_root)
    return removed
