from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [ROOT / "src", *sorted((ROOT / "adapters").glob("*/src"))]
for source_dir in reversed(SOURCE_DIRS):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _mock_application_ner_preload(monkeypatch):
    import vox.server.app as app_module

    preload = AsyncMock()
    monkeypatch.setattr(app_module, "preload_ner", preload)
    return preload


def _build_index(catalog: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for name, tags in catalog.items():
        for tag, entry in tags.items():
            adapter_package = entry.get("adapter_package")
            if adapter_package is None:
                variants = entry.get("variants")
                if variants:
                    adapter_package = variants[0].get("adapter_package")
            summaries.append(
                {
                    "name": name,
                    "tag": tag,
                    "type": entry.get("type"),
                    "description": entry.get("description"),
                    "adapter_package": adapter_package,
                }
            )
    return summaries


@pytest.fixture(autouse=True)
def _mock_remote_registry(monkeypatch):
    import vox.core.registry as registry_module
    from tests._catalog_fixture import FIXTURE_CATALOG

    def _fake_fetch(name: str, tag: str) -> dict[str, Any] | None:
        entry = FIXTURE_CATALOG.get(name, {}).get(tag)
        return deepcopy(entry) if entry is not None else None

    def _fake_index(*, force_refresh: bool = False) -> list[dict[str, Any]]:
        return _build_index(FIXTURE_CATALOG)

    monkeypatch.setattr(registry_module, "fetch_from_registry", _fake_fetch)
    monkeypatch.setattr(registry_module, "fetch_registry_index", _fake_index)
    registry_module._entry_cache.clear()
    registry_module._index_cache = None
    yield
    registry_module._entry_cache.clear()
    registry_module._index_cache = None
