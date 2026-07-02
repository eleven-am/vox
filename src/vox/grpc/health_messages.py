from __future__ import annotations

from vox.grpc import vox_pb2
from vox.operations.system import (
    HealthStatusResult,
    ListLoadedModelsResult,
    health_status_payload,
    list_loaded_models_payload,
)


def health_status_response(result: HealthStatusResult) -> vox_pb2.HealthResponse:
    return vox_pb2.HealthResponse(**health_status_payload(result))


def list_loaded_models_response(result: ListLoadedModelsResult) -> vox_pb2.ListLoadedResponse:
    payload = list_loaded_models_payload(result)
    return vox_pb2.ListLoadedResponse(
        models=[vox_pb2.LoadedModel(**model) for model in payload["models"]],
    )
