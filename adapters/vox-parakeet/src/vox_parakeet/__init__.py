from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ParakeetAdapter", "ParakeetNemoAdapter"]


def __getattr__(name: str) -> Any:
    if name == "ParakeetAdapter":
        return import_module("vox_parakeet.adapter").ParakeetAdapter
    if name == "ParakeetNemoAdapter":
        return import_module("vox_parakeet.nemo_adapter").ParakeetNemoAdapter
    raise AttributeError(name)
