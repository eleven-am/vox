from __future__ import annotations

import pytest

import vox.operations.rtc_signaling as rtc_operations
from vox.operations.errors import InvalidRtcCandidateError, RtcSessionNotFoundError
from vox.operations.rtc_signaling import (
    RtcCandidateRequest,
    RtcOfferRequest,
    RtcSessionBootstrapRequest,
    add_server_rtc_candidate,
    close_rtc_session,
    create_rtc_session,
    exchange_server_rtc_offer,
    rtc_session_bootstrap_payload,
)
from vox.server.rtc_ice import InvalidIceCandidateError
from vox.server.rtc_registry import RtcSessionRegistry


def test_create_rtc_session_owns_bootstrap_shape(monkeypatch):
    monkeypatch.setenv("VOX_RTC_STUN_URLS", "stun:turn.example.test:3478")
    registry = RtcSessionRegistry(attach_ttl_s=60)

    result = create_rtc_session(
        registry=registry,
        request=RtcSessionBootstrapRequest(
            control_transport="pondsocket",
            forward_browser_events=False,
        ),
    )
    payload = rtc_session_bootstrap_payload(result)

    record = registry.get(result.session_id, now=0)
    assert record is not None
    assert record.expected_control_transport == "pondsocket"
    assert record.forward_browser_events is False
    assert payload["session_id"].startswith("rtc_")
    assert "client_token" not in payload
    assert payload["attach_ttl_seconds"] == 60
    assert payload["expires_at"]
    assert payload["ice_servers"] == [{"urls": ["stun:turn.example.test:3478"]}]


@pytest.mark.asyncio
async def test_server_offer_attaches_session_without_a_secondary_token(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    captured = {}

    async def fake_answer(**kwargs):
        captured.update(kwargs)
        return {
            "session_id": record.session_id,
            "type": "answer",
            "sdp": "answer-sdp",
        }

    monkeypatch.setattr(rtc_operations, "create_browser_rtc_answer", fake_answer)

    result = await exchange_server_rtc_offer(
        registry=registry,
        request=RtcOfferRequest(record.session_id, "offer", "offer-sdp"),
    )

    assert result.session_id == record.session_id
    assert result.sdp == "answer-sdp"
    assert captured == {
        "registry": registry,
        "record": record,
        "offer": {"type": "offer", "sdp": "offer-sdp"},
    }
    assert record.browser_attached is True


@pytest.mark.asyncio
async def test_server_offer_cannot_attach_unknown_or_already_attached_session():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    assert registry.attach_browser_session(record.session_id) is record

    for session_id in ("rtc_missing", record.session_id):
        with pytest.raises(RtcSessionNotFoundError):
            await exchange_server_rtc_offer(
                registry=registry,
                request=RtcOfferRequest(session_id, "offer", "sdp"),
            )


@pytest.mark.asyncio
async def test_invalid_candidate_is_a_transport_neutral_operation_error(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    record.browser_attached = True
    record.rtc_peer = object()

    async def reject_candidate(**_kwargs):
        raise InvalidIceCandidateError("bad candidate")

    monkeypatch.setattr(rtc_operations, "add_browser_rtc_candidate", reject_candidate)

    with pytest.raises(InvalidRtcCandidateError):
        await add_server_rtc_candidate(
            registry=registry,
            request=RtcCandidateRequest(record.session_id, "candidate:not-valid"),
        )


def test_close_rtc_session_is_idempotent():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")

    assert close_rtc_session(registry=registry, session_id=record.session_id).closed is True
    assert close_rtc_session(registry=registry, session_id=record.session_id).closed is False
