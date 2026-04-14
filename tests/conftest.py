from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [ROOT / "src", *sorted((ROOT / "adapters").glob("*/src"))]
for source_dir in reversed(SOURCE_DIRS):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


@pytest.fixture
def anyio_backend():
    return "asyncio"
