from __future__ import annotations

import base64
import hashlib
import hmac

import pytest
from aiortc.rtcicetransport import RTCIceCandidate

from vox.server.rtc_ice import (
    InvalidIceCandidateError,
    ice_servers_from_env,
    local_candidate_events,
    parse_browser_ice_candidate,
    rewrite_private_relay_candidates,
    server_ice_servers_from_env,
)


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

    assert ice_servers_from_env(now=1000.0) == [
        {
            "urls": ["stun:turn.horus:3478", "stun:backup.example:3478"],
        }
    ]


def test_static_turn_credentials_are_returned(monkeypatch):
    monkeypatch.delenv("VOX_RTC_STUN_URLS", raising=False)
    monkeypatch.setenv("VOX_RTC_TURN_URLS", "turn:turn.horus:3478?transport=udp")
    monkeypatch.setenv("VOX_RTC_TURN_USERNAME", "user")
    monkeypatch.setenv("VOX_RTC_TURN_CREDENTIAL", "pass")
    monkeypatch.setenv("VOX_RTC_TURN_SECRET", "ignored-when-static-is-set")

    assert ice_servers_from_env(now=1000.0) == [
        {
            "urls": ["turn:turn.horus:3478?transport=udp"],
            "username": "user",
            "credential": "pass",
        }
    ]


def test_turn_rest_credentials_are_generated_from_shared_secret(monkeypatch):
    monkeypatch.delenv("VOX_RTC_STUN_URLS", raising=False)
    monkeypatch.delenv("VOX_RTC_TURN_USERNAME", raising=False)
    monkeypatch.delenv("VOX_RTC_TURN_CREDENTIAL", raising=False)
    monkeypatch.setenv("VOX_RTC_TURN_URLS", "turn:turn.horus:3478?transport=udp,turns:turn.horus:5349")
    monkeypatch.setenv("VOX_RTC_TURN_SECRET", "shared-secret")
    monkeypatch.setenv("VOX_RTC_TURN_CREDENTIAL_TTL_SECONDS", "60")

    servers = ice_servers_from_env(now=1000.0)

    username = "1060"
    expected = base64.b64encode(hmac.new(b"shared-secret", username.encode("utf-8"), hashlib.sha1).digest()).decode(
        "ascii"
    )
    assert servers == [
        {
            "urls": ["turn:turn.horus:3478?transport=udp", "turns:turn.horus:5349"],
            "username": username,
            "credential": expected,
        }
    ]


def test_server_ice_uses_browser_config_when_not_overridden(monkeypatch):
    monkeypatch.delenv("VOX_RTC_SERVER_STUN_URLS", raising=False)
    monkeypatch.delenv("VOX_RTC_SERVER_TURN_URLS", raising=False)
    monkeypatch.setenv("VOX_RTC_STUN_URLS", "stun:turn.maix.ovh:3478")
    monkeypatch.setenv("VOX_RTC_TURN_URLS", "turn:turn.maix.ovh:3478?transport=udp")
    monkeypatch.setenv("VOX_RTC_TURN_USERNAME", "user")
    monkeypatch.setenv("VOX_RTC_TURN_CREDENTIAL", "pass")

    assert server_ice_servers_from_env(now=1000.0) == [
        {"urls": ["stun:turn.maix.ovh:3478"]},
        {
            "urls": ["turn:turn.maix.ovh:3478?transport=udp"],
            "username": "user",
            "credential": "pass",
        },
    ]


def test_server_ice_can_use_internal_turn_url_with_same_secret(monkeypatch):
    monkeypatch.delenv("VOX_RTC_TURN_USERNAME", raising=False)
    monkeypatch.delenv("VOX_RTC_TURN_CREDENTIAL", raising=False)
    monkeypatch.setenv("VOX_RTC_STUN_URLS", "stun:turn.maix.ovh:3478")
    monkeypatch.setenv("VOX_RTC_TURN_URLS", "turn:turn.maix.ovh:3478?transport=udp")
    monkeypatch.setenv("VOX_RTC_SERVER_TURN_URLS", "turn:coturn.coturn.svc.cluster.local:3478?transport=udp")
    monkeypatch.setenv("VOX_RTC_TURN_SECRET", "shared-secret")
    monkeypatch.setenv("VOX_RTC_TURN_CREDENTIAL_TTL_SECONDS", "60")

    servers = server_ice_servers_from_env(now=1000.0)

    username = "1060"
    expected = base64.b64encode(hmac.new(b"shared-secret", username.encode("utf-8"), hashlib.sha1).digest()).decode(
        "ascii"
    )
    assert servers == [
        {
            "urls": ["turn:coturn.coturn.svc.cluster.local:3478?transport=udp"],
            "username": username,
            "credential": expected,
        }
    ]


def test_private_relay_candidates_are_rewritten_to_public_turn_addr(monkeypatch):
    monkeypatch.setenv("VOX_RTC_TURN_URLS", "turn:turn.maix.ovh:3478?transport=udp")
    monkeypatch.setattr("vox.server.rtc_ice._resolve_public_addr", lambda host: "176.149.222.82")
    sdp = "\r\n".join(
        [
            "v=0",
            "a=candidate:host 1 udp 2130706431 10.244.0.205 42599 typ host",
            "a=candidate:relay 1 udp 16777215 10.244.0.130 49159 typ relay raddr 10.244.0.205 rport 47668",
            "",
        ]
    )

    rewritten = rewrite_private_relay_candidates(sdp)

    assert "10.244.0.130 49159 typ relay" not in rewritten
    assert "176.149.222.82 49159 typ relay" in rewritten
    assert "10.244.0.205 42599 typ host" in rewritten


def test_private_relay_candidates_are_duplicated_for_all_browser_turn_addrs(monkeypatch):
    monkeypatch.setenv(
        "VOX_RTC_TURN_URLS",
        ",".join(
            [
                "turn:turn.maix.ovh:3478?transport=udp",
                "turn:172.198.1.55:3478?transport=udp",
                "turn:turn.maix.ovh:3478?transport=tcp",
            ]
        ),
    )
    addrs = {
        "turn.maix.ovh": "176.149.222.82",
        "172.198.1.55": "172.198.1.55",
    }
    monkeypatch.setattr("vox.server.rtc_ice._resolve_public_addr", lambda host: addrs[host])
    sdp = "\r\n".join(
        [
            "v=0",
            "a=candidate:relay 1 udp 16777215 10.244.0.130 49159 typ relay raddr 10.244.0.205 rport 47668",
            "",
        ]
    )

    rewritten = rewrite_private_relay_candidates(sdp)

    assert "10.244.0.130 49159 typ relay" not in rewritten
    assert "176.149.222.82 49159 typ relay" in rewritten
    assert "172.198.1.55 49159 typ relay" in rewritten
    assert rewritten.count("49159 typ relay") == 2


def test_local_candidate_events_preserves_mid_and_mline(monkeypatch):
    monkeypatch.delenv("VOX_RTC_TURN_URLS", raising=False)
    candidate = RTCIceCandidate(
        component=1,
        foundation="audio-host",
        ip="10.0.0.1",
        port=40000,
        priority=2130706431,
        protocol="udp",
        type="host",
        sdpMid="audio",
        sdpMLineIndex=0,
    )

    assert local_candidate_events(candidate) == [
        {
            "type": "rtc.ice_candidate",
            "candidate": {
                "candidate": "candidate:audio-host 1 udp 2130706431 10.0.0.1 40000 typ host",
                "sdpMid": "audio",
                "sdpMLineIndex": 0,
            },
        },
    ]


def test_parse_browser_ice_candidate_accepts_null_end_marker():
    assert parse_browser_ice_candidate({"candidate": None}) is None


def test_parse_browser_ice_candidate_preserves_dict_mid_and_mline():
    ice = parse_browser_ice_candidate(
        {
            "candidate": {
                "candidate": "candidate:host 1 udp 2130706431 10.0.0.1 40000 typ host",
                "sdpMid": "audio",
                "sdpMLineIndex": 0,
            }
        }
    )

    assert ice is not None
    assert ice.ip == "10.0.0.1"
    assert ice.port == 40000
    assert ice.sdpMid == "audio"
    assert ice.sdpMLineIndex == 0


def test_parse_browser_ice_candidate_accepts_string_with_top_level_mid():
    ice = parse_browser_ice_candidate(
        {
            "candidate": "candidate:host 1 udp 2130706431 10.0.0.1 40001 typ host",
            "sdpMid": "data",
            "sdpMLineIndex": 1,
        }
    )

    assert ice is not None
    assert ice.ip == "10.0.0.1"
    assert ice.port == 40001
    assert ice.sdpMid == "data"
    assert ice.sdpMLineIndex == 1


def test_parse_browser_ice_candidate_rejects_malformed_candidate():
    with pytest.raises(InvalidIceCandidateError, match="invalid ICE candidate"):
        parse_browser_ice_candidate({"candidate": {"candidate": "not a candidate"}})
