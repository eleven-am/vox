from fastapi import HTTPException

from vox.core.types import parse_model_name


def get_default_model(model_type: str, registry, store=None) -> str:
    """Get the first available model of the given type — prefer pulled models."""
    if store is not None:
        for m in store.list_models():
            if m.type.value == model_type:
                return m.full_name
    for name, tags in registry.available_models().items():
        for tag, entry in tags.items():
            if entry.get("type") == model_type:
                return f"{name}:{tag}"
    raise HTTPException(
        status_code=400,
        detail=f"No model specified and no default {model_type.upper()} model available",
    )
