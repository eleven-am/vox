from __future__ import annotations

import grpc

from vox.core.scheduler import Scheduler
from vox.grpc import vox_pb2, vox_pb2_grpc


class HealthServicer(vox_pb2_grpc.HealthServiceServicer):

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    async def Health(self, request, context):
        return vox_pb2.HealthResponse(status="ok")

    async def ListLoaded(self, request, context):
        loaded = self._scheduler.list_loaded()
        models = [
            vox_pb2.LoadedModel(
                name=m.name,
                tag=m.tag,
                type=m.type.value,
                device=m.device,
                vram_bytes=m.vram_bytes,
                loaded_at=m.loaded_at,
                last_used=m.last_used,
                ref_count=m.ref_count,
            )
            for m in loaded
        ]
        return vox_pb2.ListLoadedResponse(models=models)
