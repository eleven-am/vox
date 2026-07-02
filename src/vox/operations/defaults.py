from __future__ import annotations

from typing import Any

from vox.operations.errors import NoDefaultModelError


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


def resolve_requested_or_default_model(
    model_type: str,
    requested: str,
    registry: Any,
    store: Any | None = None,
) -> str:
    model = requested or resolve_default_model(model_type, registry, store) or ""
    if not model:
        raise NoDefaultModelError(model_type)
    return model
