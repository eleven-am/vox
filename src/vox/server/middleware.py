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
from starlette.responses import Response

from vox.logging_context import new_request_id, request_id_var

logger = logging.getLogger("vox.server.request")

HEADER = "X-Request-ID"

_QUIET_PATHS: frozenset[str] = frozenset({
    "/", "/health", "/healthz", "/readyz", "/v1/health", "/api/health",
})


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(HEADER)
        rid = incoming.strip() if incoming and incoming.strip() else new_request_id()
        token = request_id_var.set(rid)
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
            request_id_var.reset(token)
