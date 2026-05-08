from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time


def ice_servers_from_env(*, now: float | None = None) -> list[dict]:
    """Build browser RTCPeerConnection iceServers from Vox environment.

    Supports either static TURN credentials or coturn REST credentials. For
    coturn `use-auth-secret`, set VOX_RTC_TURN_SECRET and Vox will generate a
    short-lived username/password pair per session.
    """
    now = time.time() if now is None else now
    servers: list[dict] = []

    stun_urls = _csv_env("VOX_RTC_STUN_URLS")
    if stun_urls:
        servers.append({"urls": stun_urls})

    turn_urls = _csv_env("VOX_RTC_TURN_URLS")
    if not turn_urls:
        return servers

    static_username = os.environ.get("VOX_RTC_TURN_USERNAME")
    static_credential = os.environ.get("VOX_RTC_TURN_CREDENTIAL")
    shared_secret = os.environ.get("VOX_RTC_TURN_SECRET")

    if static_username and static_credential:
        servers.append({
            "urls": turn_urls,
            "username": static_username,
            "credential": static_credential,
        })
        return servers

    if shared_secret:
        ttl_s = _int_env("VOX_RTC_TURN_CREDENTIAL_TTL_SECONDS", default=3600)
        username = str(int(now) + ttl_s)
        credential = base64.b64encode(
            hmac.new(
                shared_secret.encode("utf-8"),
                username.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("ascii")
        servers.append({
            "urls": turn_urls,
            "username": username,
            "credential": credential,
        })

    return servers


def _csv_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
