"""Adapter resolution module: discover, install, activate, and load Vox adapters."""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import EntryPoint, distributions, entry_points
from pathlib import Path

from vox.core.errors import AdapterNotFoundError

logger = logging.getLogger(__name__)


ADAPTERS_DIR = "adapters"
ADAPTERS_NO_DEPS_ENV = "VOX_ADAPTERS_NO_DEPS"
ADAPTER_INSTALL_TIMEOUT_ENV = "VOX_ADAPTER_INSTALL_TIMEOUT_SECONDS"
ADAPTERS_ALLOW_UNVERIFIED_ENV = "VOX_ALLOW_UNVERIFIED_ADAPTERS"
DEFAULT_NO_DEPS_ADAPTER_PACKAGES = {
    "vox-dia",
    "vox-kokoro",
    "vox-microsoft",
    "vox-openvoice",
    "vox-qwen",
    "vox-sesame",
    "vox-voxtral",
    "vox-whisper",
}
KNOWN_ADAPTER_PACKAGES = frozenset({
    "vox-chatterbox",
    "vox-cosyvoice",
    "vox-dia",
    "vox-indextts",
    "vox-kokoro",
    "vox-microsoft",
    "vox-neutts",
    "vox-openvoice",
    "vox-orpheus",
    "vox-parakeet",
    "vox-piper",
    "vox-qwen",
    "vox-sesame",
    "vox-spark",
    "vox-voxtral",
    "vox-whisper",
    "vox-xtts",
})


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _adapter_package_allowed(package_name: str) -> bool:
    if package_name in KNOWN_ADAPTER_PACKAGES:
        return True
    return os.environ.get(ADAPTERS_ALLOW_UNVERIFIED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


InstallRunner = Callable[[list[str], int], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class AdapterInstallSpec:
    entry_point: EntryPoint
    path: Path


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    source: str
    path: Path | None
    version: str | None


def _default_install_runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class AdapterResolver:
    """Single-responsibility module owning adapter discovery, install, activation, and caching."""

    def __init__(
        self,
        vox_home: Path,
        *,
        install_runner: InstallRunner | None = None,
    ) -> None:
        self._vox_home = vox_home
        self._install_runner = install_runner or _default_install_runner
        self._adapters: dict[str, type] = {}
        self._installed_specs: dict[str, AdapterInstallSpec] = {}
        self._sanitize_sys_path()
        self._refresh_global_adapters()
        self._refresh_installed_specs()

    @property
    def vox_home(self) -> Path:
        return self._vox_home

    def resolve(self, adapter_name: str) -> type:
        cls = self._adapters.get(adapter_name)
        if cls is not None:
            return cls

        spec = self._valid_installed_spec(adapter_name)
        if spec is None:
            self._refresh_installed_specs()
            spec = self._valid_installed_spec(adapter_name)
        if spec is None:
            raise AdapterNotFoundError(adapter_name)

        return self._load_installed_spec(adapter_name, spec)

    def _load_installed_spec(self, adapter_name: str, spec: AdapterInstallSpec) -> type:
        with self._activated_path(spec.path):
            try:
                cls = spec.entry_point.load()
            except Exception as exc:
                logger.warning(
                    "Dropping broken adapter install spec for '%s' at %s: %s",
                    adapter_name,
                    spec.path,
                    exc,
                )
                self._installed_specs.pop(adapter_name, None)
                self._adapters.pop(adapter_name, None)
                raise AdapterNotFoundError(adapter_name) from exc

        self._adapters[adapter_name] = cls
        return cls

    def discover(self) -> list[AdapterInfo]:
        infos: list[AdapterInfo] = []
        seen: set[str] = set()
        for name, cls in self._adapters.items():
            seen.add(name)
            infos.append(
                AdapterInfo(
                    name=name,
                    source="entry_point",
                    path=Path(sys.modules[cls.__module__].__file__).parent
                    if getattr(cls, "__module__", None) in sys.modules
                    else None,
                    version=None,
                )
            )
        for name, spec in self._installed_specs.items():
            if name in seen:
                continue
            seen.add(name)
            infos.append(
                AdapterInfo(
                    name=name,
                    source="isolated_install",
                    path=spec.path,
                    version=self._installed_version_at(spec.path),
                )
            )
        return infos

    def installed_version(self, package_name: str) -> str | None:
        package_dir = self._adapter_install_dir(package_name)
        if not package_dir.is_dir():
            return None
        return self._installed_version_at(package_dir, package_name=package_name)

    def ensure(self, adapter_name: str, package_name: str) -> bool:
        if self._cached_adapter_loads(adapter_name, package_name) or self._installed_spec_loads(adapter_name):
            return True

        self._sanitize_sys_path()
        self._refresh_installed_specs()
        if self._cached_adapter_loads(adapter_name, package_name) or self._installed_spec_loads(adapter_name):
            return True

        if not _adapter_package_allowed(package_name):
            logger.error(
                "Refusing to install unverified adapter package %r for adapter '%s'. "
                "Set %s=1 to allow installing packages outside the built-in catalog.",
                package_name, adapter_name, ADAPTERS_ALLOW_UNVERIFIED_ENV,
            )
            return False

        logger.info("Adapter '%s' installing %s from package registry...", adapter_name, package_name)
        if not self._install_package(package_name):
            return False

        self._sanitize_sys_path()
        self._refresh_installed_specs()
        return self._cached_adapter_loads(adapter_name, package_name) or self._installed_spec_loads(adapter_name)

    def _refresh_global_adapters(self) -> None:
        adapters: dict[str, type] = {}
        for ep in entry_points(group="vox.adapters"):
            try:
                adapters[ep.name] = ep.load()
            except Exception as e:
                logger.warning(f"Skipping broken adapter plugin '{ep.name}': {e}")
        self._adapters = adapters

    def _refresh_installed_specs(self) -> None:
        self._installed_specs = self._scan_install_specs()

    def _valid_installed_spec(self, adapter_name: str) -> AdapterInstallSpec | None:
        spec = self._installed_specs.get(adapter_name)
        if spec is None:
            return None
        if spec.path.is_dir():
            return spec
        logger.warning(
            "Dropping stale adapter install spec for '%s': %s no longer exists",
            adapter_name,
            spec.path,
        )
        self._installed_specs.pop(adapter_name, None)
        self._adapters.pop(adapter_name, None)
        return None

    def _installed_spec_loads(self, adapter_name: str) -> bool:
        spec = self._valid_installed_spec(adapter_name)
        if spec is None:
            return False
        try:
            self._load_installed_spec(adapter_name, spec)
        except AdapterNotFoundError:
            return False
        return True

    def _cached_adapter_loads(self, adapter_name: str, package_name: str) -> bool:
        cls = self._adapters.get(adapter_name)
        if cls is None:
            return False

        module = sys.modules.get(getattr(cls, "__module__", ""))
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return True

        try:
            module_path = Path(module_file).resolve()
            package_dir = self._adapter_install_dir(package_name).resolve()
            adapters_root = self._ensure_adapters_root().resolve()
        except OSError:
            self._adapters.pop(adapter_name, None)
            return False

        if _path_is_relative_to(module_path, package_dir):
            if package_dir.is_dir():
                return True
            logger.warning(
                "Dropping stale cached adapter '%s': install dir %s no longer exists",
                adapter_name,
                package_dir,
            )
            self._adapters.pop(adapter_name, None)
            return False

        if _path_is_relative_to(module_path, adapters_root):
            logger.warning(
                "Dropping cached adapter '%s' loaded from unexpected install dir %s",
                adapter_name,
                module_path,
            )
            self._adapters.pop(adapter_name, None)
            return False

        return True

    def _scan_install_specs(self) -> dict[str, AdapterInstallSpec]:
        adapters_root = self._ensure_adapters_root()
        specs: dict[str, AdapterInstallSpec] = {}
        for package_dir in sorted(path for path in adapters_root.iterdir() if path.is_dir()):
            try:
                package_dists = list(distributions(path=[str(package_dir)]))
            except Exception as exc:
                logger.warning("Skipping adapter install dir '%s': %s", package_dir, exc)
                continue
            for dist in package_dists:
                for ep in dist.entry_points:
                    if ep.group != "vox.adapters":
                        continue
                    specs[ep.name] = AdapterInstallSpec(entry_point=ep, path=package_dir)
        return specs

    def _ensure_adapters_root(self) -> Path:
        adapters_root = self._vox_home / ADAPTERS_DIR
        adapters_root.mkdir(parents=True, exist_ok=True)
        return adapters_root

    def _adapter_install_dir(self, package_name: str) -> Path:
        return self._vox_home / ADAPTERS_DIR / package_name

    def _installed_version_at(
        self,
        package_dir: Path,
        *,
        package_name: str | None = None,
    ) -> str | None:
        normalized = package_name.replace("-", "_") if package_name else None
        try:
            for dist in distributions(path=[str(package_dir)]):
                name = (dist.metadata.get("Name") or "").replace("-", "_")
                if normalized is None or name == normalized:
                    return dist.version
        except Exception as exc:
            logger.warning("Failed to inspect adapter version at '%s': %s", package_dir, exc)
        return None

    def _install_package(self, package_name: str) -> bool:
        target_dir = self._adapter_install_dir(package_name)
        target_dir.mkdir(parents=True, exist_ok=True)

        install_timeout = int(os.environ.get(ADAPTER_INSTALL_TIMEOUT_ENV, "900"))
        install_no_deps = package_name in DEFAULT_NO_DEPS_ADAPTER_PACKAGES
        install_no_deps = install_no_deps or os.environ.get(ADAPTERS_NO_DEPS_ENV, "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        installers = [
            ["uv", "pip", "install", "--python", sys.executable],
            [sys.executable, "-m", "pip", "install"],
        ]
        for installer in installers:
            try:
                cmd = [*installer, "--target", str(target_dir), "--upgrade"]
                if installer[:2] == ["uv", "pip"]:
                    cmd.extend(["--refresh-package", package_name])
                if install_no_deps:
                    cmd.append("--no-deps")
                cmd.append(package_name)
                result = self._install_runner(cmd, install_timeout)
                if result.returncode == 0:
                    logger.info("Installed adapter package: %s", package_name)
                    return True
                logger.warning("%s failed: %s", " ".join(installer), result.stderr)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        logger.error("Failed to install adapter package: %s", package_name)
        return False

    def _sanitize_sys_path(self) -> None:
        adapters_root = self._ensure_adapters_root()
        adapters_root_str = str(adapters_root.resolve())
        sys.path[:] = [
            entry for entry in sys.path
            if str(Path(entry).resolve()) != adapters_root_str
        ]
        self._deactivate_install_dirs()
        importlib.invalidate_caches()

    def _deactivate_install_dirs(self, *, keep: Path | None = None) -> None:
        adapters_root = self._ensure_adapters_root().resolve()
        keep_resolved = keep.resolve() if keep is not None else None
        retained: list[str] = []
        for entry in sys.path:
            try:
                resolved = Path(entry).resolve()
            except OSError:
                retained.append(entry)
                continue
            if resolved.parent == adapters_root and resolved != keep_resolved:
                continue
            retained.append(entry)
        sys.path[:] = retained

    @contextmanager
    def _activated_path(self, adapter_path: Path) -> Iterator[None]:
        original_sys_path = list(sys.path)
        self._deactivate_install_dirs(keep=adapter_path)
        path_str = str(adapter_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        importlib.invalidate_caches()
        try:
            yield
        finally:
            sys.path[:] = original_sys_path
            importlib.invalidate_caches()
