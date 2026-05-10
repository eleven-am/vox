from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import socket
import time
from struct import unpack
from urllib.parse import urlparse

from aioice import stun


def ice_servers_from_env(*, now: float | None = None) -> list[dict]:
    """Build browser RTCPeerConnection iceServers from Vox environment.

    Supports either static TURN credentials or coturn REST credentials. For
    coturn `use-auth-secret`, set VOX_RTC_TURN_SECRET and Vox will generate a
    short-lived username/password pair per session.
    """
    return _ice_servers_from_env(
        now=now,
        stun_env="VOX_RTC_STUN_URLS",
        turn_env="VOX_RTC_TURN_URLS",
    )


def server_ice_servers_from_env(*, now: float | None = None) -> list[dict]:
    """Build the ICE servers used by Vox's server-side aiortc peer.

    Browsers need public ICE URLs, but a Vox pod may need to reach coturn by
    its in-cluster service name. When VOX_RTC_SERVER_TURN_URLS is set, use it
    for the server peer while reusing the existing TURN credentials.
    """
    if not os.environ.get("VOX_RTC_SERVER_TURN_URLS") and not os.environ.get("VOX_RTC_SERVER_STUN_URLS"):
        return ice_servers_from_env(now=now)
    return _ice_servers_from_env(
        now=now,
        stun_env="VOX_RTC_SERVER_STUN_URLS",
        turn_env="VOX_RTC_SERVER_TURN_URLS",
    )


def patch_aioice_turn_error_code_parser() -> None:
    """Allow aioice to handle coturn ERROR-CODE reason bytes defensively."""
    attr = stun.ATTRIBUTES_BY_TYPE.get(0x0009)
    if attr and attr[3] is _unpack_error_code_compat:
        return
    stun.ATTRIBUTES_BY_TYPE[0x0009] = (
        0x0009,
        "ERROR-CODE",
        stun.pack_error_code,
        _unpack_error_code_compat,
    )


def rewrite_private_relay_candidates(sdp: str) -> str:
    advertise_addrs = _relay_advertise_addrs_from_env()
    if not advertise_addrs:
        return sdp

    lines: list[str] = []
    for line in sdp.splitlines():
        lines.extend(_rewrite_candidate_line(line, advertise_addrs))
    return "\r\n".join(lines) + ("\r\n" if sdp.endswith(("\r\n", "\n")) else "")


def _ice_servers_from_env(*, now: float | None, stun_env: str, turn_env: str) -> list[dict]:
    now = time.time() if now is None else now
    servers: list[dict] = []

    stun_urls = _csv_env(stun_env)
    if stun_urls:
        servers.append({"urls": stun_urls})

    turn_urls = _csv_env(turn_env)
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


def _unpack_error_code_compat(data: bytes) -> tuple[int, str]:
    if len(data) < 4:
        raise ValueError("STUN error code is less than 4 bytes")
    _reserved, code_high, code_low = unpack("!HBB", data[0:4])
    return code_high * 100 + code_low, data[4:].decode("utf8", errors="replace")


def _relay_advertise_addrs_from_env() -> list[str]:
    turn_urls = _csv_env("VOX_RTC_TURN_URLS")
    addrs: list[str] = []
    for url in turn_urls:
        host = _turn_uri_host(url)
        if host:
            addr = _resolve_public_addr(host)
            if addr not in addrs:
                addrs.append(addr)
    return addrs


def _turn_uri_host(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"turn", "turns"}:
        return None
    if parsed.hostname:
        return parsed.hostname
    host_port = parsed.path.split("?", 1)[0]
    if not host_port:
        return None
    if host_port.startswith("[") and "]" in host_port:
        return host_port[1:host_port.index("]")]
    return host_port.rsplit(":", 1)[0]


def _resolve_public_addr(host: str) -> str:
    try:
        return socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_DGRAM)[0][4][0]
    except OSError:
        return host


def _rewrite_candidate_line(line: str, advertise_addrs: list[str]) -> list[str]:
    if not line.startswith("a=candidate:") or " typ relay" not in line:
        return [line]

    parts = line.split()
    if len(parts) < 8 or not _is_private_addr(parts[4]):
        return [line]

    lines: list[str] = []
    for addr in advertise_addrs:
        rewritten = parts.copy()
        rewritten[4] = addr
        lines.append(" ".join(rewritten))
    return lines or [line]


def _is_private_addr(value: str) -> bool:
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback or addr.is_link_local


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
