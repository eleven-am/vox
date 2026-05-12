from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import HTTPException, Request


def configured_api_key() -> str | None:
    value = os.environ.get("VOX_API_KEY", "").strip()
    return value or None


def _bearer_token(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip()
    if value.lower().startswith("bearer "):
        token = value[7:].strip()
        return token or None
    return None


def extract_api_key_from_http(request: Request) -> str | None:
    header = request.headers.get("authorization")
    token = _bearer_token(header)
    if token:
        return token
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip() or None
    query_value = request.query_params.get("api_key")
    if query_value:
        return query_value.strip() or None
    return None


def is_api_key_authorized(candidate: str | None) -> bool:
    expected = configured_api_key()
    if expected is None:
        return True
    if not candidate:
        return False
    return hmac.compare_digest(candidate, expected)


def require_api_key(request: Request) -> None:
    if is_api_key_authorized(extract_api_key_from_http(request)):
        return
    raise HTTPException(status_code=401, detail="missing or invalid API key")


def extract_api_key_from_connection(ctx: Any) -> str | None:
    token = _bearer_token(ctx.headers.get("authorization"))
    if token:
        return token
    header_value = ctx.headers.get("x-api-key")
    if header_value:
        return header_value.strip() or None
    query_value = ctx.query.get("api_key")
    if query_value:
        return query_value.strip() or None
    return None
