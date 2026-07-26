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
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import find_spec
from pathlib import Path

from vox.core.atomic_install import (
    DirectorySwap,
    merge_missing_tree,
    publish_staged_directory,
    staged_directory,
)

logger = logging.getLogger(__name__)

InstallRunner = Callable[[list[str], int], "subprocess.CompletedProcess[str]"]
ModuleProbe = Callable[[str], bool]
InstallerName = str

_ADAPTER_RUNTIME_LOCK = threading.RLock()
_ADAPTER_RUNTIME_STAGED_LOCK = threading.Lock()


class RuntimeMutation:
    def __init__(self, *, release: Callable[[], None] | None = None) -> None:
        self._swaps: list[DirectorySwap] = []
        self._finished = False
        self._release = release

    def add(self, swap: DirectorySwap) -> None:
        if self._finished:
            raise RuntimeError("runtime mutation is already finished")
        self._swaps.append(swap)

    def commit(self) -> None:
        with _ADAPTER_RUNTIME_LOCK:
            if self._finished:
                return
            try:
                for swap in self._swaps:
                    swap.commit()
            finally:
                self._finish()

    def rollback(self) -> None:
        with _ADAPTER_RUNTIME_LOCK:
            if self._finished:
                return
            errors: list[Exception] = []
            for swap in reversed(self._swaps):
                try:
                    swap.rollback()
                except Exception as exc:
                    errors.append(exc)
            self._finish()
            if errors:
                raise ExceptionGroup("runtime mutation rollback failed", errors)

    def _finish(self) -> None:
        self._swaps.clear()
        self._finished = True
        release = self._release
        self._release = None
        if release is not None:
            release()


_ADAPTER_RUNTIME_TRANSACTION: ContextVar[RuntimeMutation | None] = ContextVar(
    "vox_adapter_runtime_transaction",
    default=None,
)


@contextmanager
def bind_runtime_mutation(
    mutation: RuntimeMutation | None,
) -> Iterator[None]:
    token = _ADAPTER_RUNTIME_TRANSACTION.set(mutation)
    try:
        yield
    finally:
        _ADAPTER_RUNTIME_TRANSACTION.reset(token)


@dataclass(frozen=True)
class TargetRuntime:
    name: str
    path: Path


def run_with_adapter_runtime_lock(operation: Callable[..., object], /, *args: object, **kwargs: object) -> object:
    """Run one dependency-graph mutation without racing another adapter lifecycle."""

    current = _ADAPTER_RUNTIME_TRANSACTION.get()
    if current is not None:
        return operation(*args, **kwargs)

    with _ADAPTER_RUNTIME_STAGED_LOCK, _ADAPTER_RUNTIME_LOCK:
        transaction = RuntimeMutation()
        token = _ADAPTER_RUNTIME_TRANSACTION.set(transaction)
        try:
            result = operation(*args, **kwargs)
        except BaseException:
            transaction.rollback()
            raise
        else:
            transaction.commit()
            return result
        finally:
            _ADAPTER_RUNTIME_TRANSACTION.reset(token)


def stage_adapter_runtime_mutation(
    operation: Callable[..., object],
    /,
    *args: object,
    **kwargs: object,
) -> RuntimeMutation:
    if _ADAPTER_RUNTIME_TRANSACTION.get() is not None:
        raise RuntimeError("cannot stage a nested runtime mutation")
    _ADAPTER_RUNTIME_STAGED_LOCK.acquire()
    transaction = RuntimeMutation(release=_ADAPTER_RUNTIME_STAGED_LOCK.release)
    try:
        with _ADAPTER_RUNTIME_LOCK:
            token = _ADAPTER_RUNTIME_TRANSACTION.set(transaction)
            try:
                operation(*args, **kwargs)
            finally:
                _ADAPTER_RUNTIME_TRANSACTION.reset(token)
    except BaseException:
        transaction.rollback()
        raise
    return transaction


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
        root_path = Path(root or runtime_root()).resolve()
        protected_runtime_paths = _loaded_native_runtime_paths(root_path)
        retained: list[str] = []
        for entry in sys.path:
            if entry == path:
                continue
            entry_path = Path(entry).resolve()
            if entry_path.is_relative_to(root_path):
                if entry_path in protected_runtime_paths:
                    retained.append(entry)
                continue
            retained.append(entry)
        sys.path[:] = retained
    elif path in sys.path:
        sys.path.remove(path)

    if path not in sys.path:
        sys.path.insert(0, path)
    importlib.invalidate_caches()
    return path


def _loaded_native_runtime_paths(root_path: Path) -> set[Path]:
    protected: set[Path] = set()
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file or not any(str(module_file).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
            continue
        try:
            relative_path = Path(module_file).resolve().relative_to(root_path)
        except (OSError, ValueError):
            continue
        if relative_path.parts:
            protected.add(root_path / relative_path.parts[0])
    return protected


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


def module_available_in_runtime(
    import_name: str,
    runtime_path: Path | str,
    *,
    include_app_fallback: bool,
) -> bool:
    """Probe runtime availability without silently accepting app-env packages."""

    try:
        spec = find_spec(import_name)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    if include_app_fallback:
        return True

    runtime = Path(runtime_path).resolve()
    candidate_paths: list[Path] = []
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        candidate_paths.append(Path(spec.origin))
    if spec.submodule_search_locations:
        candidate_paths.extend(Path(path) for path in spec.submodule_search_locations)
    return any(_is_relative_to(path, runtime) for path in candidate_paths)


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
        raise RuntimeError(f"Failed to bootstrap pip for {context}. stderr: {result.stderr.strip()}")


def write_app_fallback_path(runtime_dir: Path) -> None:
    app_purelib = sysconfig.get_paths()["purelib"]
    fallback_file = runtime_dir / "_vox_runtime_fallback_paths.pth"
    fallback_file.write_text(f"{app_purelib}\n", encoding="utf-8")


def ensure_target_runtime(
    runtime_name: str,
    package_spec: str,
    import_name: str,
    *,
    include_app_fallback: bool,
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
    full isolated venv, as Voxtral TTS does today. Callers must set
    ``include_app_fallback`` explicitly: ``True`` is a deliberate compatibility
    bridge that lets a target runtime import the Vox app environment after its
    own isolated packages, while ``False`` keeps the runtime strict.
    """

    runtime_root_path = root or runtime_root(home=home)
    runtime = target_runtime(runtime_name, home=home, root=runtime_root_path)

    probe = module_probe or (
        lambda name: module_available_in_runtime(
            name,
            runtime.path,
            include_app_fallback=include_app_fallback,
        )
    )
    if runtime.path.is_dir():
        activate_runtime_path(runtime.path, root=runtime_root_path)
    if probe(import_name):
        if include_app_fallback:
            write_app_fallback_path(runtime.path)
        return runtime.path

    if purge_modules:
        purge_runtime_modules(purge_modules)

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
        if include_app_fallback:
            write_app_fallback_path(runtime.path)
        if probe(import_name):
            return runtime.path

    raise RuntimeError(f"{runtime_name} runtime package is missing and could not be bootstrapped: {package_spec}")


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

    target = Path(runtime_path)
    packages = list(requirements)
    runner = install_runner or _default_install_runner

    for installer_name in installer_order:
        with staged_directory(target) as stage:
            installer = _install_commands(
                runtime_path=str(stage),
                packages=packages,
                no_deps=no_deps,
                upgrade=upgrade,
                installer_order=(installer_name,),
                extra_install_args=list(extra_install_args),
            )[0]
            try:
                if _is_python_pip_command(installer):
                    ensure_pip_available(context=context)
                result = runner(installer, timeout)
            except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as exc:
                logger.warning("Installer %s unavailable for %s: %s", installer[0], context, exc)
                continue

            if result.returncode != 0 and _is_python_pip_command(installer):
                result = runner(installer, timeout)

            if result.returncode == 0:
                staged_expected = _map_expected_paths_to_stage(
                    target,
                    stage,
                    expected_paths,
                )
                if _install_result_has_expected_paths(
                    runtime_path=stage,
                    packages=packages,
                    expected_paths=staged_expected,
                ):
                    _publish_runtime_stage(
                        stage,
                        target,
                        preserve_existing=True,
                    )
                    return True
                continue

            logger.warning("%s failed: %s", " ".join(installer), result.stderr)

    return False


def remove_target_runtime_paths(
    runtime_path: Path | str,
    paths: Iterable[Path | str],
) -> bool:
    target = Path(runtime_path)
    if not target.is_dir():
        return False

    excluded: list[Path] = []
    for item in paths:
        path = Path(item)
        try:
            relative = path.relative_to(target)
        except ValueError as exc:
            raise ValueError(f"Runtime removal path is outside {target}: {path}") from exc
        if relative == Path("."):
            raise ValueError(f"Runtime removal path cannot be the runtime root: {target}")
        if path.exists() or path.is_symlink():
            excluded.append(relative)
    if not excluded:
        return False

    with staged_directory(target) as stage:
        merge_missing_tree(target, stage, excluded=excluded)
        _publish_runtime_stage(stage, target, preserve_existing=False)
    return True


@contextmanager
def staged_target_runtime(
    runtime_path: Path | str,
    *,
    preserve_existing: bool = False,
) -> Iterator[Path]:
    target = Path(runtime_path)
    with staged_directory(target) as stage:
        if preserve_existing and target.is_dir():
            merge_missing_tree(target, stage)
        yield stage
        _publish_runtime_stage(stage, target, preserve_existing=False)


def _publish_runtime_stage(
    stage: Path,
    target: Path,
    *,
    preserve_existing: bool,
) -> None:
    transaction = _ADAPTER_RUNTIME_TRANSACTION.get()
    swap = publish_staged_directory(
        stage,
        target,
        preserve_existing=preserve_existing,
        retain_backup=transaction is not None,
    )
    if transaction is not None:
        transaction.add(swap)


def _map_expected_paths_to_stage(
    target: Path,
    stage: Path,
    expected_paths: Iterable[Path],
) -> tuple[Path, ...]:
    mapped: list[Path] = []
    for expected in expected_paths:
        path = Path(expected)
        try:
            relative = path.relative_to(target)
        except ValueError:
            mapped.append(path)
        else:
            mapped.append(stage / relative)
    return tuple(mapped)


def _install_result_has_expected_paths(
    *,
    runtime_path: Path | str,
    packages: list[str],
    expected_paths: Iterable[Path],
) -> bool:
    expected = tuple(expected_paths)
    if not expected:
        return True
    if all(path.exists() for path in expected):
        return True
    logger.warning(
        "Installer reported success but runtime packages were not placed into %s: %s",
        runtime_path,
        packages,
    )
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
    except ValueError:
        return False
    return True
