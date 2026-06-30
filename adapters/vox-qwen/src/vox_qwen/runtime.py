from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from vox.core.adapter_runtime import (
    ensure_target_runtime,
    module_available,
    runtime_root,
)


def _runtime_root() -> Path:
    return runtime_root()


def _module_available(import_name: str) -> bool:
    return module_available(import_name)


def ensure_runtime(
    package_name: str,
    package_spec: str,
    import_name: str,
    *,
    purge_modules: Iterable[str] = (),
    no_deps: bool = False,
    extra_packages: Iterable[str] = (),
) -> None:
    ensure_target_runtime(
        package_name,
        package_spec,
        import_name,
        purge_modules=purge_modules,
        no_deps=no_deps,
        extra_packages=extra_packages,
        root=_runtime_root(),
        module_probe=_module_available,
        context="Qwen runtime install",
    )
