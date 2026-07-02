from __future__ import annotations

import logging

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2_grpc
from vox.grpc.model_messages import (
    delete_model_response,
    list_models_response,
    pull_error_message,
    pull_progress_message,
    show_model_response,
)
from vox.grpc.operation_errors import map_operation_errors_to_grpc
from vox.operations.errors import CatalogEntryNotFoundError
from vox.operations.models import (
    delete_model,
    list_models,
    model_reference_request_from_fields,
    pull_model,
    show_model,
)

logger = logging.getLogger(__name__)


class ModelServicer(vox_pb2_grpc.ModelServiceServicer):

    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Pull(self, request, context):
        try:
            events = pull_model(
                store=self._store,
                scheduler=self._scheduler,
                registry=self._registry,
                request=model_reference_request_from_fields(
                    name=request.name,
                    variant=request.variant or None,
                ),
            )
        except CatalogEntryNotFoundError as exc:
            yield pull_error_message(exc)
            return

        async for event in events:
            yield pull_progress_message(event)

    async def List(self, request, context):
        models = list_models(store=self._store)
        return list_models_response(models)

    async def Show(self, request, context):
        async with map_operation_errors_to_grpc(context):
            result = show_model(
                store=self._store,
                registry=self._registry,
                request=model_reference_request_from_fields(name=request.name),
            )

        return show_model_response(result)

    async def Delete(self, request, context):
        async with map_operation_errors_to_grpc(context):
            await delete_model(
                store=self._store,
                scheduler=self._scheduler,
                registry=self._registry,
                request=model_reference_request_from_fields(name=request.name),
            )
        return delete_model_response()
