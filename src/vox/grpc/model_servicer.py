from __future__ import annotations

import logging

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.model_messages import (
    delete_model_response,
    list_models_response,
    pull_progress_message,
    show_model_response,
)
from vox.grpc.operation_errors import operation_error_status
from vox.operations.errors import CatalogEntryNotFoundError, OperationError
from vox.operations.models import (
    delete_model,
    list_models,
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
                name=request.name,
            )
        except CatalogEntryNotFoundError as exc:
            yield vox_pb2.PullProgress(status="error", error=str(exc))
            return

        async for event in events:
            yield pull_progress_message(event)

    async def List(self, request, context):
        models = list_models(store=self._store)
        return list_models_response(models)

    async def Show(self, request, context):
        try:
            result = show_model(store=self._store, registry=self._registry, name=request.name)
        except OperationError as exc:
            code, msg = operation_error_status(exc)
            await context.abort(code, msg)
            return

        return show_model_response(result)

    async def Delete(self, request, context):
        try:
            await delete_model(
                store=self._store,
                scheduler=self._scheduler,
                registry=self._registry,
                name=request.name,
            )
        except OperationError as exc:
            code, msg = operation_error_status(exc)
            await context.abort(code, msg)
            return
        return delete_model_response()
