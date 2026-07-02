from __future__ import annotations

from vox.server.rtc_registry import RtcSessionRegistry
from vox.server.rtc_sessions import (
    RtcSessionBootstrapRequest,
    create_rtc_session_bootstrap,
    parse_rtc_session_bootstrap_request,
)


def test_parse_rtc_session_bootstrap_request_defaults_to_browser_events_unchanged():
    assert parse_rtc_session_bootstrap_request(None) == RtcSessionBootstrapRequest()
    assert parse_rtc_session_bootstrap_request({"other": False}) == RtcSessionBootstrapRequest()


def test_parse_rtc_session_bootstrap_request_preserves_browser_event_choice():
    assert parse_rtc_session_bootstrap_request({"browser_events": False}) == (
        RtcSessionBootstrapRequest(forward_browser_events=False)
    )
    assert parse_rtc_session_bootstrap_request({"browser_events": 1}) == (
        RtcSessionBootstrapRequest(forward_browser_events=True)
    )


def test_create_rtc_session_bootstrap_owns_browser_payload_shape(monkeypatch):
    monkeypatch.setenv("VOX_RTC_STUN_URLS", "stun:turn.example.test:3478")
    registry = RtcSessionRegistry(join_token_ttl_s=60)

    payload = create_rtc_session_bootstrap(
        registry=registry,
        request=RtcSessionBootstrapRequest(forward_browser_events=False),
    )

    record = registry.get(payload["session_id"], now=0)
    assert record is not None
    assert record.forward_browser_events is False
    assert payload["session_id"].startswith("rtc_")
    assert payload["client_token"].startswith("rtc_client_")
    assert payload["join_token_ttl_seconds"] == 60
    assert payload["expires_at"]
    assert payload["ice_servers"] == [{"urls": ["stun:turn.example.test:3478"]}]
