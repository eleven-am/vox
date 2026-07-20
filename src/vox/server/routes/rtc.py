"""Authenticated RTC session bootstrap."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from vox.operations.rtc_signaling import (
    RtcSessionBootstrapRequest,
    rtc_session_bootstrap_payload,
)
from vox.operations.rtc_signaling import (
    create_rtc_session as create_rtc_session_operation,
)
from vox.server.app_services import app_rtc_registry
from vox.server.auth import require_api_key

router = APIRouter()


def parse_rtc_session_bootstrap_request(body: Any) -> RtcSessionBootstrapRequest:
    if isinstance(body, dict) and "browser_events" in body:
        return RtcSessionBootstrapRequest(
            control_transport="pondsocket",
            forward_browser_events=bool(body["browser_events"]),
        )
    return RtcSessionBootstrapRequest(control_transport="pondsocket")


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
