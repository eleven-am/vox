from __future__ import annotations

from vox.core.scheduler import Scheduler
from vox.grpc import vox_pb2_grpc
from vox.grpc.health_messages import health_status_response, list_loaded_models_response
from vox.operations.system import (
    get_health_status,
    health_status_request_from_fields,
    list_loaded_models,
    list_loaded_models_request_from_fields,
)


class HealthServicer(vox_pb2_grpc.HealthServiceServicer):

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    async def Health(self, request, context):
        result = get_health_status(request=health_status_request_from_fields())
        return health_status_response(result)

    async def ListLoaded(self, request, context):
        result = list_loaded_models(
            scheduler=self._scheduler,
            request=list_loaded_models_request_from_fields(),
        )
        return list_loaded_models_response(result)
