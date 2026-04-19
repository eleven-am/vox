from __future__ import annotations

import logging
import os
import subprocess
import sys
import sysconfig
from collections.abc import Iterable
from importlib.util import find_spec
from pathlib import Path

logger = logging.getLogger(__name__)


def _runtime_root() -> Path:
    vox_home = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))
    return vox_home / "runtime"


def _prune_other_runtime_paths(current_runtime_path: str) -> None:
    runtime_root = str(_runtime_root())
    pruned_paths: list[str] = []
    for path in sys.path:
        if path == current_runtime_path:
            continue
        if path.startswith(runtime_root):
            continue
        pruned_paths.append(path)
    sys.path[:] = pruned_paths


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


def _ensure_pip_available() -> None:
    if find_spec("pip") is not None:
        return

    result = subprocess.run(
        [sys.executable, "-m", "ensurepip", "--default-pip"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to bootstrap pip for the Qwen runtime install. "
            f"stderr: {result.stderr.strip()}"
        )


def ensure_runtime(
    package_name: str,
    package_spec: str,
    import_name: str,
    *,
    purge_modules: Iterable[str] = (),
    no_deps: bool = False,
    extra_packages: Iterable[str] = (),
) -> None:
    runtime_dir = _runtime_root() / package_name
    runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime_path = str(runtime_dir)
    _prune_other_runtime_paths(runtime_path)
    if runtime_path in sys.path:
        sys.path.remove(runtime_path)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)

    if _module_available(import_name):
        return

    if purge_modules:
        _clear_modules(purge_modules)

    app_purelib = sysconfig.get_paths()["purelib"]
    fallback_file = runtime_dir / "_vox_runtime_fallback_paths.pth"
    fallback_file.write_text(f"{app_purelib}\n", encoding="utf-8")

    base_uv_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        runtime_path,
        "--upgrade",
    ]
    base_pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        runtime_path,
        "--upgrade",
    ]
    if no_deps:
        base_uv_cmd.append("--no-deps")
        base_pip_cmd.append("--no-deps")

    packages = [package_spec, *extra_packages]
    installers = [
        [*base_uv_cmd, *packages],
        [*base_pip_cmd, *packages],
    ]

    for installer in installers:
        try:
            result = subprocess.run(
                installer,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0:
            logger.info("Bootstrapped %s runtime into %s", package_spec, runtime_dir)
            if runtime_path not in sys.path:
                sys.path.insert(0, runtime_path)
            if _module_available(import_name):
                return
        elif installer[0] == sys.executable and installer[1:3] == ["-m", "pip"]:
            _ensure_pip_available()
            retry = subprocess.run(
                installer,
                capture_output=True,
                text=True,
                timeout=900,
            )
            if retry.returncode == 0:
                logger.info("Bootstrapped %s runtime into %s", package_spec, runtime_dir)
                if runtime_path not in sys.path:
                    sys.path.insert(0, runtime_path)
                if _module_available(import_name):
                    return
            logger.warning("%s failed: %s", " ".join(installer), retry.stderr)
        else:
            logger.warning("%s failed: %s", " ".join(installer), result.stderr)

    raise RuntimeError(
        f"{package_name} runtime package is missing and could not be bootstrapped: {package_spec}"
    )
