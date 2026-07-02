from fastapi import APIRouter, Request

from vox.operations import system as system_operations
from vox.server.app_services import app_scheduler

router = APIRouter()


@router.get("/v1/health")
async def health():
    result = system_operations.get_health_status(
        request=system_operations.health_status_request_from_fields(),
    )
    return system_operations.health_status_payload(result)


@router.get("/v1/models/loaded")
async def list_running(request: Request):
    scheduler = app_scheduler(request)
    result = system_operations.list_loaded_models(
        scheduler=scheduler,
        request=system_operations.list_loaded_models_request_from_fields(),
    )
    return system_operations.list_loaded_models_payload(result)
