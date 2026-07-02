from __future__ import annotations

from dataclasses import dataclass

import pytest

from vox.operations.defaults import resolve_default_model, resolve_requested_or_default_model
from vox.operations.errors import NoDefaultModelError


@dataclass(frozen=True)
class TypeValue:
    value: str


@dataclass(frozen=True)
class StoredModel:
    full_name: str
    type: TypeValue


class Store:
    def __init__(self, models=()) -> None:
        self._models = tuple(models)

    def list_models(self):
        return list(self._models)


class Registry:
    def __init__(self, models=None) -> None:
        self._models = models or {}

    def available_models(self):
        return self._models


def test_resolve_default_model_prefers_installed_store_model():
    store = Store([
        StoredModel(full_name="parakeet:local", type=TypeValue("stt")),
    ])
    registry = Registry({
        "parakeet": {"remote": {"type": "stt"}},
    })

    assert resolve_default_model("stt", registry, store) == "parakeet:local"


def test_resolve_default_model_falls_back_to_registry_catalog():
    registry = Registry({
        "kokoro": {"v1.0": {"type": "tts"}},
    })

    assert resolve_default_model("tts", registry, None) == "kokoro:v1.0"


def test_resolve_requested_or_default_model_preserves_explicit_request():
    registry = Registry()

    assert resolve_requested_or_default_model("stt", "custom:model", registry, None) == "custom:model"


def test_resolve_requested_or_default_model_raises_canonical_operation_error():
    with pytest.raises(NoDefaultModelError, match="no model specified and no default STT model available"):
        resolve_requested_or_default_model("stt", "", Registry(), Store())
