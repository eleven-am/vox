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
    required_imports: Iterable[str] = (),
) -> None:
    required_names = tuple(required_imports)

    def probe_runtime(name: str) -> bool:
        return _module_available(name) and all(_module_available(required) for required in required_names)

    ensure_target_runtime(
        package_name,
        package_spec,
        import_name,
        purge_modules=purge_modules,
        no_deps=no_deps,
        extra_packages=extra_packages,
        root=_runtime_root(),
        module_probe=probe_runtime,
        context="Qwen runtime install",
        include_app_fallback=True,
    )
