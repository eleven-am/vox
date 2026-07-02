"""Shared helpers for adapter-managed runtime environments.

Adapters are installed as lightweight plugin packages under
``$VOX_HOME/adapters``. Some model families also need heavier or conflicting
runtime dependencies. Those belong under ``$VOX_HOME/runtime/<name>`` and should
not leak into the base Vox image or the global Python environment.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

logger = logging.getLogger(__name__)

InstallRunner = Callable[[list[str], int], "subprocess.CompletedProcess[str]"]
ModuleProbe = Callable[[str], bool]
InstallerName = str


@dataclass(frozen=True)
class TargetRuntime:
    name: str
    path: Path


def vox_home() -> Path:
    return Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox")))


def runtime_root(*, home: Path | None = None) -> Path:
    return (home or vox_home()) / "runtime"


def target_runtime(
    name: str,
    *,
    home: Path | None = None,
    root: Path | None = None,
) -> TargetRuntime:
    return TargetRuntime(name=name, path=(root or runtime_root(home=home)) / name)


def activate_runtime_path(
    runtime_path: Path | str,
    *,
    root: Path | None = None,
    prune_sibling_runtimes: bool = True,
) -> str:
    """Put one adapter runtime path first on ``sys.path``.

    By default, other paths directly under ``$VOX_HOME/runtime`` are removed so
    dependency sets from different adapters do not accidentally shadow each
    other.
    """

    path = str(Path(runtime_path))
    if prune_sibling_runtimes:
        root_path = str(root or runtime_root())
        retained: list[str] = []
        for entry in sys.path:
            if entry == path:
                continue
            if entry.startswith(root_path):
                continue
            retained.append(entry)
        sys.path[:] = retained
    elif path in sys.path:
        sys.path.remove(path)

    if path not in sys.path:
        sys.path.insert(0, path)
    importlib.invalidate_caches()
    return path


def purge_runtime_modules(prefixes: Iterable[str]) -> None:
    prefix_tuple = tuple(prefixes)
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefix_tuple):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def module_available(import_name: str) -> bool:
    if import_name in sys.modules:
        return True
    try:
        return find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def ensure_pip_available(*, context: str = "adapter runtime") -> None:
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
            f"Failed to bootstrap pip for {context}. "
            f"stderr: {result.stderr.strip()}"
        )


def write_app_fallback_path(runtime_dir: Path) -> None:
    app_purelib = sysconfig.get_paths()["purelib"]
    fallback_file = runtime_dir / "_vox_runtime_fallback_paths.pth"
    fallback_file.write_text(f"{app_purelib}\n", encoding="utf-8")


def ensure_target_runtime(
    runtime_name: str,
    package_spec: str,
    import_name: str,
    *,
    purge_modules: Iterable[str] = (),
    no_deps: bool = False,
    extra_packages: Iterable[str] = (),
    timeout: int = 900,
    home: Path | None = None,
    root: Path | None = None,
    install_runner: InstallRunner | None = None,
    module_probe: ModuleProbe | None = None,
    context: str | None = None,
) -> Path:
    """Install and activate a ``--target`` adapter runtime if needed.

    This covers the common adapter pattern. Very heavy runtimes can still use a
    full isolated venv, as Voxtral TTS does today.
    """

    runtime_root_path = root or runtime_root(home=home)
    runtime = target_runtime(runtime_name, home=home, root=runtime_root_path)
    runtime.path.mkdir(parents=True, exist_ok=True)
    activate_runtime_path(runtime.path, root=runtime_root_path)

    probe = module_probe or module_available
    if probe(import_name):
        return runtime.path

    if purge_modules:
        purge_runtime_modules(purge_modules)

    write_app_fallback_path(runtime.path)

    packages = [package_spec, *extra_packages]
    install_context = context or f"{runtime_name} runtime"

    if install_target_runtime_requirements(
        runtime.path,
        packages,
        no_deps=no_deps,
        timeout=timeout,
        install_runner=install_runner,
        context=install_context,
    ):
        activate_runtime_path(runtime.path, root=runtime_root_path)
        if probe(import_name):
            return runtime.path

    raise RuntimeError(
        f"{runtime_name} runtime package is missing and could not be bootstrapped: {package_spec}"
    )


def install_target_runtime_requirements(
    runtime_path: Path | str,
    requirements: Iterable[str],
    *,
    no_deps: bool = False,
    upgrade: bool = True,
    timeout: int = 900,
    expected_paths: Iterable[Path] = (),
    installer_order: Iterable[InstallerName] = ("uv", "pip"),
    extra_install_args: Iterable[str] = (),
    install_runner: InstallRunner | None = None,
    context: str = "adapter runtime",
) -> bool:
    """Install requirement specs into an adapter ``--target`` runtime path."""

    path = str(Path(runtime_path))
    packages = list(requirements)
    runner = install_runner or _default_install_runner

    for installer in _install_commands(
        runtime_path=path,
        packages=packages,
        no_deps=no_deps,
        upgrade=upgrade,
        installer_order=installer_order,
        extra_install_args=list(extra_install_args),
    ):
        try:
            if _is_python_pip_command(installer):
                ensure_pip_available(context=context)
            result = runner(installer, timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as exc:
            logger.warning("Installer %s unavailable for %s: %s", installer[0], context, exc)
            continue

        if result.returncode == 0:
            if expected_paths and not all(path.exists() for path in expected_paths):
                logger.warning(
                    "Installer reported success but runtime packages were not placed into %s: %s",
                    runtime_path,
                    packages,
                )
                continue
            return True

        if _is_python_pip_command(installer):
            retry = runner(installer, timeout)
            if retry.returncode == 0:
                if expected_paths and not all(path.exists() for path in expected_paths):
                    logger.warning(
                        "Installer reported success but runtime packages were not placed into %s: %s",
                        runtime_path,
                        packages,
                    )
                    continue
                return True
            logger.warning("%s failed: %s", " ".join(installer), retry.stderr)
            continue

        logger.warning("%s failed: %s", " ".join(installer), result.stderr)

    return False


def _install_commands(
    *,
    runtime_path: str,
    packages: list[str],
    no_deps: bool,
    upgrade: bool = True,
    installer_order: Iterable[InstallerName] = ("uv", "pip"),
    extra_install_args: list[str] | None = None,
) -> list[list[str]]:
    extra_args = extra_install_args or []
    commands: dict[str, list[str]] = {
        "uv": [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            runtime_path,
        ],
        "pip": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            runtime_path,
        ],
    }
    commands["uv"].extend(extra_args)
    commands["pip"].extend(extra_args)
    if upgrade:
        commands["uv"].append("--upgrade")
        commands["pip"].append("--upgrade")
    if no_deps:
        commands["uv"].append("--no-deps")
        commands["pip"].append("--no-deps")

    installers: list[list[str]] = []
    for name in installer_order:
        try:
            installers.append([*commands[name], *packages])
        except KeyError as exc:
            raise ValueError(f"unknown adapter runtime installer: {name}") from exc
    return installers


def _default_install_runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _is_python_pip_command(cmd: list[str]) -> bool:
    return len(cmd) >= 3 and cmd[0] == sys.executable and cmd[1:3] == ["-m", "pip"]
