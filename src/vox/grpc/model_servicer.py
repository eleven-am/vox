from __future__ import annotations

import logging

import grpc

from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.operations.errors import (
    CatalogEntryNotFoundError,
    ModelInUseError,
    OperationError,
    StoredModelNotFoundError,
)
from vox.operations.models import (
    delete_model,
    list_models,
    pull_model,
    show_model,
)

logger = logging.getLogger(__name__)


def _model_op_error_status(exc: OperationError) -> tuple[grpc.StatusCode, str]:
    if isinstance(exc, (CatalogEntryNotFoundError, StoredModelNotFoundError)):
        return grpc.StatusCode.NOT_FOUND, str(exc)
    if isinstance(exc, ModelInUseError):
        return grpc.StatusCode.FAILED_PRECONDITION, str(exc)
    return grpc.StatusCode.INTERNAL, str(exc)


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
            yield vox_pb2.PullProgress(
                status=event.status,
                completed=event.completed,
                total=event.total,
                error=event.error,
            )

    async def List(self, request, context):
        models = list_models(store=self._store)
        return vox_pb2.ListModelsResponse(
            models=[
                vox_pb2.ModelInfo(
                    name=m.full_name,
                    type=m.type.value,
                    format=m.format.value,
                    architecture=m.architecture,
                    size_bytes=m.size_bytes,
                    description=m.description,
                )
                for m in models
            ]
        )

    async def Show(self, request, context):
        try:
            result = show_model(store=self._store, registry=self._registry, name=request.name)
        except OperationError as exc:
            code, msg = _model_op_error_status(exc)
            await context.abort(code, msg)
            return

        config_map = {k: (str(v) if not isinstance(v, str) else v) for k, v in result.config.items()}

        return vox_pb2.ShowResponse(
            name=result.name,
            config=config_map,
            layers=[
                vox_pb2.LayerInfo(
                    media_type=layer.media_type,
                    digest=layer.digest,
                    size=layer.size,
                    filename=layer.filename,
                )
                for layer in result.layers
            ],
        )

    async def Delete(self, request, context):
        try:
            await delete_model(
                store=self._store,
                scheduler=self._scheduler,
                registry=self._registry,
                name=request.name,
            )
        except OperationError as exc:
            code, msg = _model_op_error_status(exc)
            await context.abort(code, msg)
            return
        return vox_pb2.DeleteResponse(status="success")
