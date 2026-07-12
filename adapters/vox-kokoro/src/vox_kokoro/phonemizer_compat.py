from __future__ import annotations

import errno
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
PHONEMIZER_LOGGER = logging.getLogger("phonemizer")
PHONEMIZER_LOGGER.setLevel(logging.ERROR)

_RETRYABLE_CLEANUP_ERRNOS = {errno.EBUSY, errno.ENOTEMPTY}
_TEMP_CLEANUP_RETRY_DELAYS = (0.1, 0.5, 2.0, 5.0)


def _remove_tempdir_with_retries(
    tempdir: str | Path,
    *,
    retry_delays: tuple[float, ...] = _TEMP_CLEANUP_RETRY_DELAYS,
) -> None:
    path = Path(tempdir)
    for delay in retry_delays:
        time.sleep(delay)
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in _RETRYABLE_CLEANUP_ERRNOS:
                logger.warning("Unable to clean phonemizer temporary directory %s: %s", path, exc)
                return

    logger.debug("Deferred phonemizer temporary-directory cleanup for %s", path)


def _schedule_tempdir_cleanup(tempdir: str | Path) -> None:
    threading.Thread(
        target=_remove_tempdir_with_retries,
        args=(tempdir,),
        name="vox-kokoro-temp-cleanup",
        daemon=True,
    ).start()


def patch_espeak_compat() -> bool:
    try:
        from phonemizer.backend.espeak import api as espeak_api
        from phonemizer.backend.espeak.api import EspeakAPI
        from phonemizer.backend.espeak.words_mismatch import BaseWordsMismatch
    except ImportError:
        return False

    original_delete = getattr(EspeakAPI, "_delete", None)
    if original_delete is not None and not getattr(original_delete, "_vox_patched", False):

        def _delete_nfs_safe(library: Any, tempdir: str) -> None:
            try:
                original_delete(library, tempdir)
            except FileNotFoundError:
                return
            except OSError as exc:
                if exc.errno not in _RETRYABLE_CLEANUP_ERRNOS:
                    raise
                _schedule_tempdir_cleanup(tempdir)

        _delete_nfs_safe._vox_patched = True  # type: ignore[attr-defined]
        EspeakAPI._delete = staticmethod(_delete_nfs_safe)
        if hasattr(espeak_api, "_delete"):
            espeak_api._delete = _delete_nfs_safe

        original_init = getattr(EspeakAPI, "__init__", None)
        if original_init is not None and not getattr(original_init, "_vox_patched", False):

            def _init_nfs_safe(self: Any, *args: Any, **kwargs: Any) -> None:
                finalize = espeak_api.weakref.finalize

                def _finalize_override(obj: Any, func: Any, *f_args: Any, **f_kwargs: Any) -> Any:
                    if getattr(func, "__name__", None) == "_delete":
                        func = _delete_nfs_safe
                    return finalize(obj, func, *f_args, **f_kwargs)

                espeak_api.weakref.finalize = _finalize_override
                try:
                    original_init(self, *args, **kwargs)
                finally:
                    espeak_api.weakref.finalize = finalize

            _init_nfs_safe._vox_patched = True  # type: ignore[attr-defined]
            EspeakAPI.__init__ = _init_nfs_safe

        original_finalize = getattr(espeak_api.weakref, "finalize", None)
        if original_finalize is not None and not getattr(original_finalize, "_vox_patched", False):

            def _finalize_nfs_safe(obj: Any, func: Any, *args: Any, **kwargs: Any) -> Any:
                if getattr(func, "__name__", None) == "_delete":
                    func = _delete_nfs_safe
                return original_finalize(obj, func, *args, **kwargs)

            _finalize_nfs_safe._vox_patched = True  # type: ignore[attr-defined]
            espeak_api.weakref.finalize = _finalize_nfs_safe

    original_resume = getattr(BaseWordsMismatch, "_resume", None)
    if original_resume is not None and not getattr(original_resume, "_vox_patched", False):

        def _resume_quietly(self: Any, lines: list[str], num_mismatches: int) -> None:
            return None

        _resume_quietly._vox_patched = True  # type: ignore[attr-defined]
        BaseWordsMismatch._resume = _resume_quietly

    return True
