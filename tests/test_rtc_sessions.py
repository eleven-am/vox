from __future__ import annotations

from vox.server.rtc_registry import RtcSessionRegistry
from vox.server.rtc_sessions import (
    RTC_CANDIDATE_TOKEN_FIELDS,
    RTC_OFFER_TOKEN_FIELDS,
    RtcSessionBootstrapRequest,
    create_rtc_session_bootstrap,
    parse_rtc_session_bootstrap_request,
    rtc_token_from_authorization_or_body,
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


def test_rtc_token_from_authorization_or_body_prefers_authorization_token():
    assert (
        rtc_token_from_authorization_or_body(
            "from-header",
            {"client_token": "from-body"},
            body_fields=RTC_OFFER_TOKEN_FIELDS,
        )
        == "from-header"
    )


def test_rtc_token_from_authorization_or_body_reads_named_offer_token_field():
    assert (
        rtc_token_from_authorization_or_body(
            None,
            {"client_token": "rtc_client_123"},
            body_fields=RTC_OFFER_TOKEN_FIELDS,
        )
        == "rtc_client_123"
    )


def test_rtc_token_from_authorization_or_body_reads_candidate_compatibility_aliases():
    assert (
        rtc_token_from_authorization_or_body(
            None,
            {"media_token": "rtc_media_primary", "token": "rtc_media_compat"},
            body_fields=RTC_CANDIDATE_TOKEN_FIELDS,
        )
        == "rtc_media_primary"
    )
    assert (
        rtc_token_from_authorization_or_body(
            None,
            {"token": "rtc_media_compat"},
            body_fields=RTC_CANDIDATE_TOKEN_FIELDS,
        )
        == "rtc_media_compat"
    )


def test_rtc_token_from_authorization_or_body_rejects_missing_or_invalid_body():
    assert rtc_token_from_authorization_or_body(None, None, body_fields=RTC_OFFER_TOKEN_FIELDS) == ""
    assert rtc_token_from_authorization_or_body(None, [], body_fields=RTC_OFFER_TOKEN_FIELDS) == ""
    assert (
        rtc_token_from_authorization_or_body(
            None,
            {"client_token": "   "},
            body_fields=RTC_OFFER_TOKEN_FIELDS,
        )
        == ""
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
