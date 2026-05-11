from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from vox.operations.errors import InvalidConfigError


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class LiveKitConfig:
    url: str
    api_key: str
    api_secret: str
    token_ttl_s: int = 120

    @property
    def join_token_ttl_s(self) -> int:
        return self.token_ttl_s

    @classmethod
    def from_env(cls) -> LiveKitConfig:
        url = os.environ.get("VOX_LIVEKIT_URL") or os.environ.get("LIVEKIT_URL") or ""
        api_key = os.environ.get("VOX_LIVEKIT_API_KEY") or os.environ.get("LIVEKIT_API_KEY") or ""
        api_secret = os.environ.get("VOX_LIVEKIT_API_SECRET") or os.environ.get("LIVEKIT_API_SECRET") or ""
        ttl_raw = (
            os.environ.get("VOX_LIVEKIT_TOKEN_TTL_SECONDS")
            or os.environ.get("LIVEKIT_TOKEN_TTL_SECONDS")
            or "120"
        )
        try:
            ttl_s = int(ttl_raw)
        except ValueError as exc:
            raise InvalidConfigError("VOX_LIVEKIT_TOKEN_TTL_SECONDS must be an integer") from exc
        if not url.strip():
            raise InvalidConfigError("VOX_LIVEKIT_URL or LIVEKIT_URL is required")
        if not api_key.strip():
            raise InvalidConfigError("VOX_LIVEKIT_API_KEY or LIVEKIT_API_KEY is required")
        if not api_secret.strip():
            raise InvalidConfigError("VOX_LIVEKIT_API_SECRET or LIVEKIT_API_SECRET is required")
        if ttl_s <= 0:
            raise InvalidConfigError("LiveKit token TTL must be positive")
        return cls(
            url=url.strip(),
            api_key=api_key.strip(),
            api_secret=api_secret.strip(),
            token_ttl_s=ttl_s,
        )


@dataclass(frozen=True)
class LiveKitIssuedToken:
    token: str
    expires_at: float

    @property
    def expires_at_iso(self) -> str:
        return datetime.fromtimestamp(self.expires_at, tz=UTC).isoformat()


class LiveKitTokenIssuer:
    def __init__(self, config: LiveKitConfig) -> None:
        self._config = config

    def issue_join_token(
        self,
        *,
        room: str,
        identity: str,
        name: str | None = None,
        ttl_s: int | None = None,
    ) -> LiveKitIssuedToken:
        now = int(time.time())
        ttl = int(ttl_s or self._config.token_ttl_s)
        expires_at = now + ttl
        claims: dict[str, Any] = {
            "iss": self._config.api_key,
            "sub": identity,
            "nbf": now,
            "exp": expires_at,
            "video": {
                "roomJoin": True,
                "room": room,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,
            },
        }
        if name:
            claims["name"] = name
        return LiveKitIssuedToken(
            token=self._encode_jwt(claims),
            expires_at=float(expires_at),
        )

    def _encode_jwt(self, claims: dict[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        claims_b64 = _b64url(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b64}.{claims_b64}".encode("ascii")
        signature = hmac.new(
            self._config.api_secret.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        return f"{header_b64}.{claims_b64}.{_b64url(signature)}"


def decode_unverified_livekit_token(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verification. Intended for tests and diagnostics only."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("token is not a JWT")
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode((payload + padding).encode("ascii")))
