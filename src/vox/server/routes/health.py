from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/v1/health")
async def health():
    return {"status": "ok"}


@router.get("/api/health", include_in_schema=False)
async def legacy_health():
    return await health()


@router.get("/v1/models/loaded")
async def list_running(request: Request):
    scheduler = request.app.state.scheduler
    loaded = scheduler.list_loaded()
    return {
        "models": [
            {
                "name": m.name,
                "tag": m.tag,
                "type": m.type.value,
                "device": m.device,
                "vram_bytes": m.vram_bytes,
                "loaded_at": m.loaded_at,
                "last_used": m.last_used,
                "ref_count": m.ref_count,
            }
            for m in loaded
        ]
    }
