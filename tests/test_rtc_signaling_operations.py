from __future__ import annotations

from types import SimpleNamespace

import pytest

import vox.operations.rtc_signaling as rtc_operations
from vox.operations.errors import InvalidRtcCandidateError, RtcSessionNotFoundError
from vox.operations.rtc_signaling import (
    RtcCandidateRequest,
    RtcOfferRequest,
    RtcSessionBootstrapRequest,
    add_server_rtc_candidate,
    create_rtc_session,
    exchange_server_rtc_offer,
    rtc_session_bootstrap_payload,
)
from vox.server.rtc_ice import InvalidIceCandidateError
from vox.server.rtc_registry import RtcSessionRegistry


class _PreparedAnswer:
    def __init__(self, record, sdp: str) -> None:
        self.session_id = record.session_id
        self.answer_type = "answer"
        self.sdp = sdp
        self.attachment = SimpleNamespace(attempt_id="attempt")
        self._record = record

    async def commit(self) -> None:
        self._record.browser_attached = True

    async def rollback(self) -> None:
        self._record.browser_attached = False


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
        return _PreparedAnswer(record, "answer-sdp")

    monkeypatch.setattr(rtc_operations, "create_browser_rtc_answer", fake_answer)

    result = await exchange_server_rtc_offer(
        registry=registry,
        request=RtcOfferRequest(record.session_id, "offer", "offer-sdp"),
    )

    assert result.session_id == record.session_id
    assert result.sdp == "answer-sdp"
    await result.commit()
    assert captured == {
        "registry": registry,
        "record": record,
        "offer": {"type": "offer", "sdp": "offer-sdp"},
        "restart": False,
        "generation": None,
    }
    assert record.browser_attached is True


@pytest.mark.asyncio
async def test_failed_server_offer_does_not_consume_browser_attachment(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    attempts = 0

    async def answer(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("invalid offer")
        return _PreparedAnswer(record, "answer-sdp")

    monkeypatch.setattr(rtc_operations, "create_browser_rtc_answer", answer)

    with pytest.raises(RuntimeError, match="invalid offer"):
        await exchange_server_rtc_offer(
            registry=registry,
            request=RtcOfferRequest(record.session_id, "offer", "broken"),
        )

    assert record.browser_attached is False

    result = await exchange_server_rtc_offer(
        registry=registry,
        request=RtcOfferRequest(record.session_id, "offer", "valid"),
    )
    await result.commit()

    assert result.sdp == "answer-sdp"
    assert record.browser_attached is True


@pytest.mark.asyncio
async def test_server_restart_replaces_an_attached_browser_session(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    record.browser_attached = True
    captured = {}

    async def answer(**kwargs):
        captured.update(kwargs)
        return _PreparedAnswer(record, "replacement-answer")

    monkeypatch.setattr(rtc_operations, "create_browser_rtc_answer", answer)

    result = await exchange_server_rtc_offer(
        registry=registry,
        request=RtcOfferRequest(
            record.session_id,
            "offer",
            "replacement-offer",
            restart=True,
            generation=12,
        ),
    )
    await result.commit()

    assert result.sdp == "replacement-answer"
    assert captured["restart"] is True
    assert captured["generation"] == 12
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
