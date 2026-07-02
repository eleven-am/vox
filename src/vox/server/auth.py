from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import HTTPException, Request

AUTHORIZATION_HEADER = "authorization"
API_KEY_HEADER = "x-api-key"
API_KEY_QUERY_PARAM = "api_key"
MISSING_OR_INVALID_API_KEY = "missing or invalid API key"


def configured_api_key() -> str | None:
    value = os.environ.get("VOX_API_KEY", "").strip()
    return value or None


def bearer_token_from_authorization(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip()
    if value.lower().startswith("bearer "):
        token = value[7:].strip()
        return token or None
    return None


def bearer_token_from_http(request: Request) -> str | None:
    return bearer_token_from_authorization(request.headers.get(AUTHORIZATION_HEADER))


def extract_api_key_from_parts(headers: Any, query: Any) -> str | None:
    token = bearer_token_from_authorization(headers.get(AUTHORIZATION_HEADER))
    if token:
        return token
    api_key = headers.get(API_KEY_HEADER)
    if api_key:
        return api_key.strip() or None
    query_value = query.get(API_KEY_QUERY_PARAM)
    if query_value:
        return query_value.strip() or None
    return None


def extract_api_key_from_http(request: Request) -> str | None:
    return extract_api_key_from_parts(request.headers, request.query_params)


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
    raise HTTPException(status_code=401, detail=MISSING_OR_INVALID_API_KEY)


def extract_api_key_from_connection(ctx: Any) -> str | None:
    return extract_api_key_from_parts(ctx.headers, ctx.query)


def authorize_api_key_connection(ctx: Any) -> None:
    if is_api_key_authorized(extract_api_key_from_connection(ctx)):
        ctx.accept()
        return
    ctx.decline(401, MISSING_OR_INVALID_API_KEY)
