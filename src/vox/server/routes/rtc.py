"""Authenticated RTC session bootstrap."""

from __future__ import annotations

from fastapi import APIRouter, Request

from vox.operations.rtc_signaling import (
    create_rtc_session as create_rtc_session_operation,
)
from vox.operations.rtc_signaling import (
    rtc_session_bootstrap_payload,
)
from vox.server.app_services import app_rtc_registry
from vox.server.auth import require_api_key
from vox.server.rtc_sessions import parse_rtc_session_bootstrap_request

router = APIRouter()


@router.post("/v1/rtc/sessions")
async def create_rtc_session(request: Request) -> dict:
    require_api_key(request)
    registry = app_rtc_registry(request)
    try:
        body = await request.json()
    except Exception:
        body = None
    result = create_rtc_session_operation(
        registry=registry,
        request=parse_rtc_session_bootstrap_request(body),
    )
    return rtc_session_bootstrap_payload(result)
