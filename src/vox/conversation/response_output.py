from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("response output params must contain finite numbers")
        return value
    raise ValueError("response output params must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _frozen_params(params: Mapping[str, Any] | None) -> Mapping[str, Any]:
    frozen = _freeze_json(params or {})
    if not isinstance(frozen, Mapping):
        raise ValueError("response output params must be an object")
    return frozen


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"response output {field_name} must be a non-empty string")
    return value.strip()


def _required_text(value: str, field_name: str) -> str:
    resolved = _optional_text(value, field_name)
    if resolved is None:
        raise ValueError(f"response output {field_name} must be a non-empty string")
    return resolved


def _valid_speed(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("response output speed must be a number")
    speed = float(value)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("response output speed must be a finite number greater than zero")
    return speed


def _required_speed(value: float | int) -> float:
    resolved = _valid_speed(value)
    if resolved is None:
        raise ValueError("response output speed must be a finite number greater than zero")
    return resolved


@dataclass(frozen=True, slots=True)
class ResponseOutputOptions:
    model: str | None = None
    voice: str | None = None
    language: str | None = None
    speed: float | None = None
    params: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _optional_text(self.model, "model"))
        object.__setattr__(self, "voice", _optional_text(self.voice, "voice"))
        object.__setattr__(self, "language", _optional_text(self.language, "language"))
        object.__setattr__(self, "speed", _valid_speed(self.speed))
        if self.params is not None:
            object.__setattr__(self, "params", _frozen_params(self.params))

    def params_dict(self) -> dict[str, Any] | None:
        if self.params is None:
            return None
        return _thaw_json(self.params)


@dataclass(frozen=True, slots=True)
class ResponseOutputConfig:
    model: str
    voice: str | None
    language: str
    speed: float = 1.0
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        object.__setattr__(self, "voice", _optional_text(self.voice, "voice"))
        object.__setattr__(self, "language", _required_text(self.language, "language"))
        object.__setattr__(self, "speed", _required_speed(self.speed))
        object.__setattr__(self, "params", _frozen_params(self.params))

    def params_dict(self) -> dict[str, Any]:
        return _thaw_json(self.params)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "language": self.language,
            "speed": self.speed,
            "params": self.params_dict(),
        }
        if self.voice is not None:
            payload["voice"] = self.voice
        return payload


def resolve_response_output(
    options: ResponseOutputOptions | None,
    *,
    model: str,
    voice: str | None,
    language: str,
) -> ResponseOutputConfig:
    return ResponseOutputConfig(
        model=options.model if options is not None and options.model is not None else model,
        voice=options.voice if options is not None and options.voice is not None else voice,
        language=(options.language if options is not None and options.language is not None else language),
        speed=options.speed if options is not None and options.speed is not None else 1.0,
        params=(options.params_dict() if options is not None and options.params is not None else {}),
    )
