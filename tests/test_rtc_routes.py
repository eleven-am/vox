from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.operations.rtc_signaling import RtcSessionBootstrapRequest
from vox.server.routes.rtc import parse_rtc_session_bootstrap_request
from vox.server.routes.rtc import router as rtc_router


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(rtc_router)
    return app


def test_create_rtc_session_returns_ephemeral_binding():
    client = TestClient(_build_app())

    response = client.post("/v1/rtc/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"].startswith("rtc_")
    assert "client_token" not in payload
    assert payload["attach_ttl_seconds"] == 120
    assert payload["expires_at"]
    assert payload["ice_servers"] == []


def test_create_rtc_session_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    client = TestClient(_build_app())

    unauthorized = client.post("/v1/rtc/sessions")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["detail"] == "missing or invalid API key"

    authorized = client.post(
        "/v1/rtc/sessions",
        headers={"authorization": "Bearer secret"},
    )
    assert authorized.status_code == 200
    assert authorized.json()["session_id"].startswith("rtc_")


def test_direct_http_signaling_routes_do_not_exist():
    client = TestClient(_build_app())
    session_id = client.post("/v1/rtc/sessions").json()["session_id"]

    assert client.post(f"/v1/rtc/sessions/{session_id}/offer", json={}).status_code == 404
    assert client.post(f"/v1/rtc/sessions/{session_id}/candidates", json={}).status_code == 404
    assert client.get(f"/v1/rtc/sessions/{session_id}/events").status_code == 404


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
