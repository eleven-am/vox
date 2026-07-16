from __future__ import annotations

from vox.operations.rtc_signaling import RtcSessionBootstrapRequest
from vox.server.rtc_sessions import parse_rtc_session_bootstrap_request


def test_parse_rtc_session_bootstrap_request_defaults_to_browser_events_unchanged():
    expected = RtcSessionBootstrapRequest(control_transport="pondsocket")
    assert parse_rtc_session_bootstrap_request(None) == expected
    assert parse_rtc_session_bootstrap_request({"other": False}) == expected


def test_parse_rtc_session_bootstrap_request_preserves_browser_event_choice():
    assert parse_rtc_session_bootstrap_request({"browser_events": False}) == (
        RtcSessionBootstrapRequest(
            control_transport="pondsocket",
            forward_browser_events=False,
        )
    )
    assert parse_rtc_session_bootstrap_request({"browser_events": 1}) == (
        RtcSessionBootstrapRequest(
            control_transport="pondsocket",
            forward_browser_events=True,
        )
    )
