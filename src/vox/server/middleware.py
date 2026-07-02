"""FastAPI middleware — request ID + one structured access-log line.

Honors an incoming ``X-Request-ID`` header when present (so an upstream
gateway can correlate across services) and generates a fresh ID otherwise.
Always echoes the ID back on the response.

Health-style paths are skipped to keep the log quiet; everything else emits
one INFO line on completion with method, path, status, and duration.
"""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from vox.logging_context import bind_request_id, current_request_id, reset_request_id
from vox.server.auth import (
    MISSING_OR_INVALID_API_KEY,
    api_key_required,
    extract_api_key_from_http,
    is_api_key_authorized,
)

logger = logging.getLogger("vox.server.request")

HEADER = "X-Request-ID"

_QUIET_PATHS: frozenset[str] = frozenset({
    "/", "/health", "/healthz", "/readyz", "/v1/health",
})

_AUTH_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/", "/health", "/healthz", "/readyz", "/v1/health",
})


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Enforce the API key on every HTTP route when one is configured.

    When ``VOX_API_KEY`` is unset the server stays open (unchanged behavior).
    Health/probe paths are always exempt so orchestrator liveness checks work.
    Websocket connections are authenticated in their own routes.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if (
            api_key_required()
            and request.url.path not in _AUTH_EXEMPT_PATHS
            and not is_api_key_authorized(extract_api_key_from_http(request))
        ):
            return JSONResponse(
                {"detail": MISSING_OR_INVALID_API_KEY},
                status_code=401,
                headers={HEADER: current_request_id()},
            )
        return await call_next(request)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        token = bind_request_id(request.headers.get(HEADER))
        rid = current_request_id()
        start = time.perf_counter()
        path = request.url.path
        method = request.method
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[HEADER] = rid
            return response
        except Exception:
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "%s %s failed after %d ms", method, path, duration_ms,
            )
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            if path not in _QUIET_PATHS:
                logger.info(
                    "%s %s -> %d (%d ms)",
                    method, path, status_code, duration_ms,
                )
            reset_request_id(token)
