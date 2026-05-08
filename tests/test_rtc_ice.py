from __future__ import annotations

import base64
import hashlib
import hmac

from vox.server.rtc_ice import ice_servers_from_env


def test_ice_servers_empty_without_env(monkeypatch):
    for key in (
        "VOX_RTC_STUN_URLS",
        "VOX_RTC_TURN_URLS",
        "VOX_RTC_TURN_USERNAME",
        "VOX_RTC_TURN_CREDENTIAL",
        "VOX_RTC_TURN_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)

    assert ice_servers_from_env(now=1000.0) == []


def test_stun_urls_are_returned(monkeypatch):
    monkeypatch.setenv("VOX_RTC_STUN_URLS", "stun:turn.horus:3478, stun:backup.example:3478")
    monkeypatch.delenv("VOX_RTC_TURN_URLS", raising=False)

    assert ice_servers_from_env(now=1000.0) == [{
        "urls": ["stun:turn.horus:3478", "stun:backup.example:3478"],
    }]


def test_static_turn_credentials_are_returned(monkeypatch):
    monkeypatch.delenv("VOX_RTC_STUN_URLS", raising=False)
    monkeypatch.setenv("VOX_RTC_TURN_URLS", "turn:turn.horus:3478?transport=udp")
    monkeypatch.setenv("VOX_RTC_TURN_USERNAME", "user")
    monkeypatch.setenv("VOX_RTC_TURN_CREDENTIAL", "pass")
    monkeypatch.setenv("VOX_RTC_TURN_SECRET", "ignored-when-static-is-set")

    assert ice_servers_from_env(now=1000.0) == [{
        "urls": ["turn:turn.horus:3478?transport=udp"],
        "username": "user",
        "credential": "pass",
    }]


def test_turn_rest_credentials_are_generated_from_shared_secret(monkeypatch):
    monkeypatch.delenv("VOX_RTC_STUN_URLS", raising=False)
    monkeypatch.delenv("VOX_RTC_TURN_USERNAME", raising=False)
    monkeypatch.delenv("VOX_RTC_TURN_CREDENTIAL", raising=False)
    monkeypatch.setenv("VOX_RTC_TURN_URLS", "turn:turn.horus:3478?transport=udp,turns:turn.horus:5349")
    monkeypatch.setenv("VOX_RTC_TURN_SECRET", "shared-secret")
    monkeypatch.setenv("VOX_RTC_TURN_CREDENTIAL_TTL_SECONDS", "60")

    servers = ice_servers_from_env(now=1000.0)

    username = "1060"
    expected = base64.b64encode(
        hmac.new(b"shared-secret", username.encode("utf-8"), hashlib.sha1).digest()
    ).decode("ascii")
    assert servers == [{
        "urls": ["turn:turn.horus:3478?transport=udp", "turns:turn.horus:5349"],
        "username": username,
        "credential": expected,
    }]
