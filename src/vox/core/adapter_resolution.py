"""Adapter resolution module: discover, install, activate, and load Vox adapters."""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import EntryPoint, distributions, entry_points
from pathlib import Path

from vox.core.errors import AdapterNotFoundError

logger = logging.getLogger(__name__)


ADAPTERS_DIR = "adapters"
BUNDLED_ADAPTERS_ENV = "VOX_BUNDLED_ADAPTERS"
BUNDLED_ADAPTERS_NO_DEPS_ENV = "VOX_BUNDLED_ADAPTERS_NO_DEPS"
DISABLE_BUNDLED_ADAPTERS_ENV = "VOX_DISABLE_BUNDLED_ADAPTERS"
ADAPTERS_NO_DEPS_ENV = "VOX_ADAPTERS_NO_DEPS"
ADAPTER_INSTALL_TIMEOUT_ENV = "VOX_ADAPTER_INSTALL_TIMEOUT_SECONDS"
DEFAULT_NO_DEPS_ADAPTER_PACKAGES = {
    "vox-dia",
    "vox-kokoro",
    "vox-microsoft",
    "vox-openvoice",
    "vox-parakeet",
    "vox-qwen",
    "vox-voxtral",
}


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


def _bundled_adapters_root_default() -> Path:
    return Path(__file__).resolve().parents[3] / "adapters"


class AdapterResolver:
    """Single-responsibility module owning adapter discovery, install, activation, and caching."""

    def __init__(
        self,
        vox_home: Path,
        bundled_adapters_root: Path | None = None,
        *,
        install_runner: InstallRunner | None = None,
    ) -> None:
        self._vox_home = vox_home
        self._bundled_root = bundled_adapters_root or _bundled_adapters_root_default()
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

        spec = self._installed_specs.get(adapter_name)
        if spec is None:
            self._refresh_installed_specs()
            spec = self._installed_specs.get(adapter_name)
        if spec is None:
            raise AdapterNotFoundError(adapter_name)

        with self._activated_path(spec.path):
            cls = spec.entry_point.load()

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

    def bundled_version(self, package_name: str) -> str | None:
        bundled_source = self._find_bundled_source(package_name)
        if bundled_source is None:
            return None
        pyproject = bundled_source / "pyproject.toml"
        if not pyproject.is_file():
            return None
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning("Failed to parse bundled adapter metadata for '%s': %s", package_name, exc)
            return None
        project = data.get("project", {})
        version = project.get("version")
        return version if isinstance(version, str) else None

    def ensure(self, adapter_name: str, package_name: str) -> bool:
        bundled_version = self.bundled_version(package_name)
        installed_version = self.installed_version(package_name)
        needs_refresh = bundled_version is not None and installed_version != bundled_version
        if adapter_name in self._adapters or adapter_name in self._installed_specs:
            if not needs_refresh:
                return True
            logger.info(
                "Refreshing adapter '%s' from bundled source (%s -> %s)",
                package_name,
                installed_version or "missing",
                bundled_version,
            )
            self._adapters.pop(adapter_name, None)

        self._sanitize_sys_path()
        self._refresh_installed_specs()
        installed_version = self.installed_version(package_name)
        needs_refresh = bundled_version is not None and installed_version != bundled_version
        if adapter_name in self._adapters or adapter_name in self._installed_specs:
            if not needs_refresh:
                return True
            self._adapters.pop(adapter_name, None)

        action = "refreshing" if needs_refresh else "installing"
        logger.info("Adapter '%s' %s %s...", adapter_name, action, package_name)
        if not self._install_package(package_name):
            return False

        self._sanitize_sys_path()
        self._refresh_installed_specs()
        return adapter_name in self._adapters or adapter_name in self._installed_specs

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

    def _find_bundled_source(self, package_name: str) -> Path | None:
        if os.environ.get(DISABLE_BUNDLED_ADAPTERS_ENV, "").lower() in {"1", "true", "yes", "on"}:
            return None

        candidates: list[Path] = []
        bundled_root = os.environ.get(BUNDLED_ADAPTERS_ENV)
        if bundled_root:
            candidates.append(Path(bundled_root))
        candidates.append(self._bundled_root)

        for base_dir in candidates:
            candidate = base_dir / package_name
            if (candidate / "pyproject.toml").is_file():
                return candidate
        return None

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
        package_spec = package_name

        bundled_source = self._find_bundled_source(package_name)
        install_timeout = int(os.environ.get(ADAPTER_INSTALL_TIMEOUT_ENV, "900"))
        install_no_deps = package_name in DEFAULT_NO_DEPS_ADAPTER_PACKAGES
        install_no_deps = install_no_deps or os.environ.get(ADAPTERS_NO_DEPS_ENV, "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if bundled_source is not None:
            package_spec = str(bundled_source)
            install_no_deps = install_no_deps or os.environ.get(BUNDLED_ADAPTERS_NO_DEPS_ENV, "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            logger.info("Installing bundled adapter package from %s", bundled_source)

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
                cmd.append(package_spec)
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
