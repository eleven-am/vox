from __future__ import annotations

import inspect
from typing import Any

from vox.core.adapter import TTSAdapter


def call_accepts_keyword(call: Any, name: str) -> bool:
    try:
        signature = inspect.signature(call)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def validate_adapter_synthesis_params(adapter: TTSAdapter, params: dict[str, Any]) -> None:
    if not params:
        return

    supported = {param.name: param for param in adapter.synthesis_parameters()}
    unknown = sorted(set(params) - set(supported))
    if unknown:
        names = ", ".join(unknown)
        supported_names = ", ".join(sorted(supported)) or "none"
        raise ValueError(
            f"Unsupported synthesis parameter(s) for {adapter.info().name}: {names}. "
            f"Supported parameters: {supported_names}"
        )

    for name, value in params.items():
        spec = supported[name]
        expected_type = spec.type.lower()
        if expected_type == "number":
            valid = isinstance(value, int | float) and not isinstance(value, bool)
        elif expected_type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "boolean":
            valid = isinstance(value, bool)
        elif expected_type == "string":
            valid = isinstance(value, str)
        else:
            raise ValueError(
                f"Adapter {adapter.info().name} declares unsupported parameter type {spec.type!r} for {name}"
            )

        if not valid:
            raise ValueError(f"Synthesis parameter {name!r} for {adapter.info().name} must be {expected_type}")

        if expected_type in ("number", "integer"):
            numeric = float(value)
            if spec.min_value is not None and numeric < spec.min_value:
                raise ValueError(f"Synthesis parameter {name!r} for {adapter.info().name} must be >= {spec.min_value}")
            if spec.max_value is not None and numeric > spec.max_value:
                raise ValueError(f"Synthesis parameter {name!r} for {adapter.info().name} must be <= {spec.max_value}")
