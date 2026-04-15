from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from importlib.util import find_spec
from pathlib import Path

logger = logging.getLogger(__name__)


def _runtime_root() -> Path:
    vox_home = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))
    return vox_home / "runtime"


def _clear_modules(prefixes: Iterable[str]) -> None:
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        ):
            sys.modules.pop(module_name, None)


def _module_available(import_name: str) -> bool:
    module = sys.modules.get(import_name)
    if module is not None:
        return True
    try:
        return find_spec(import_name) is not None
    except ValueError:
        return True


def ensure_runtime(
    package_name: str,
    package_spec: str,
    import_name: str,
    *,
    purge_modules: Iterable[str] = (),
) -> None:
    runtime_dir = _runtime_root() / package_name
    runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime_path = str(runtime_dir)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)

    if _module_available(import_name):
        return

    if purge_modules:
        _clear_modules(purge_modules)

    installers = [
        ["uv", "pip", "install", "--python", sys.executable, "--target", runtime_path, package_spec],
        [sys.executable, "-m", "pip", "install", "--target", runtime_path, package_spec],
    ]

    for installer in installers:
        try:
            result = subprocess.run(
                installer,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            logger.info("Bootstrapped %s runtime into %s", package_spec, runtime_dir)
            if runtime_path not in sys.path:
                sys.path.insert(0, runtime_path)
            if _module_available(import_name):
                return
        else:
            logger.warning("%s failed: %s", " ".join(installer), result.stderr)

    raise RuntimeError(
        f"{package_name} runtime package is missing and could not be bootstrapped: {package_spec}"
    )
