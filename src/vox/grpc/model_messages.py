from __future__ import annotations

from typing import Any

from vox.core.types import ModelInfo
from vox.grpc import vox_pb2
from vox.operations.models import (
    PullEvent,
    ShowResult,
    list_models_payload,
    pull_event_payload,
    show_model_payload,
)


def pull_progress_message(event: PullEvent) -> vox_pb2.PullProgress:
    return vox_pb2.PullProgress(**pull_event_payload(event))


def list_models_response(models: list[ModelInfo]) -> vox_pb2.ListModelsResponse:
    payload = list_models_payload(models)
    return vox_pb2.ListModelsResponse(
        models=[vox_pb2.ModelInfo(**model) for model in payload["models"]]
    )


def show_model_response(result: ShowResult) -> vox_pb2.ShowResponse:
    payload = show_model_payload(result)
    return vox_pb2.ShowResponse(
        name=payload["name"],
        config=_string_config(payload["config"]),
        layers=[vox_pb2.LayerInfo(**layer) for layer in payload["layers"]],
    )


def delete_model_response() -> vox_pb2.DeleteResponse:
    return vox_pb2.DeleteResponse(status="success")


def _string_config(config: dict[str, Any]) -> dict[str, str]:
    return {key: value if isinstance(value, str) else str(value) for key, value in config.items()}
