from __future__ import annotations

from typing import Any

from vox.operations.rtc_signaling import RtcSessionBootstrapRequest


def parse_rtc_session_bootstrap_request(body: Any) -> RtcSessionBootstrapRequest:
    if isinstance(body, dict) and "browser_events" in body:
        return RtcSessionBootstrapRequest(
            control_transport="pondsocket",
            forward_browser_events=bool(body["browser_events"]),
        )
    return RtcSessionBootstrapRequest(control_transport="pondsocket")
