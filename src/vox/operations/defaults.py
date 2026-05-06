from __future__ import annotations

from typing import Any


def resolve_default_model(model_type: str, registry: Any, store: Any | None = None) -> str | None:
    if store is not None:
        for m in store.list_models():
            if m.type.value == model_type:
                return m.full_name
    for name, tags in registry.available_models().items():
        for tag, entry in tags.items():
            if entry.get("type") == model_type:
                return f"{name}:{tag}"
    return None
