from __future__ import annotations

import asyncio
import logging

import pytest

import vox.operations.rtc_runtime as rtc_runtime_module
from tests.fakes import FakeScheduler
from vox.operations.conversation import ConversationSessionConfig
from vox.operations.conversation_commands import (
    ClientEventCommand,
    ResponseStartCommand,
    SessionUpdateCommand,
)
from vox.operations.errors import (
    InvalidConfigError,
    RtcControlAlreadyAttachedError,
    RtcControlTransportMismatchError,
)
from vox.operations.rtc_runtime import (
    RtcCandidateCommand,
    RtcOfferCommand,
    RtcRuntime,
)
from vox.operations.rtc_signaling import RtcCandidateResult, RtcOfferAnswer
from vox.server.rtc_media import emit_media_event
from vox.server.rtc_registry import RtcSessionRegistry


@pytest.mark.asyncio
async def test_runtime_emits_attached_before_preexisting_session_events():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    emitted: list[dict] = []
    await record.media_events.put({"type": "rtc.connection_state", "state": "new"})

    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append(emitted, event),
    )
    await asyncio.sleep(0)

    assert emitted == []

    await runtime.start()
    await asyncio.sleep(0)

    assert [event["type"] for event in emitted[:2]] == [
        "rtc.session.attached",
        "rtc.connection_state",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_buffers_candidate_before_offer_and_preserves_order(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    emitted: list[dict] = []
    applied = []

    async def fake_exchange(**_kwargs):
        return RtcOfferAnswer(
            session_id=record.session_id,
            answer_type="answer",
            sdp="answer-sdp",
        )

    async def fake_add(**kwargs):
        applied.append(kwargs["request"])
        return RtcCandidateResult(ok=True)

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    monkeypatch.setattr(rtc_runtime_module, "add_server_rtc_candidate", fake_add)
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append(emitted, event),
    )

    await runtime.start()
    await runtime.dispatch(
        RtcCandidateCommand(
            candidate="candidate:first",
            sdp_mid="audio",
            sdp_m_line_index=0,
        )
    )
    await runtime.dispatch(RtcCandidateCommand(candidate=None))
    assert applied == []

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-sdp"))

    assert [request.candidate for request in applied] == ["candidate:first", None]
    assert emitted[0]["type"] == "rtc.session.attached"
    assert emitted[1] == {
        "type": "rtc.answer",
        "session_id": record.session_id,
        "answer": {"type": "answer", "sdp": "answer-sdp"},
    }
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_allows_setup_commands_before_offer_but_not_response_commands():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append([], event),
    )
    config = ConversationSessionConfig(stt_model="stt:1", tts_model="tts:1")

    await runtime.dispatch(SessionUpdateCommand(config=config))
    await runtime.dispatch(ClientEventCommand(event="ui.ready", payload={"call": "pending"}))

    assert record.orchestrator.config == config
    assert len(record.pending_client_events) == 1
    with pytest.raises(InvalidConfigError, match="send rtc.offer first"):
        await runtime.dispatch(ResponseStartCommand())
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_ignores_duplicate_completion_and_rejects_late_candidate(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    applied = []

    async def fake_add(**kwargs):
        applied.append(kwargs["request"])
        return RtcCandidateResult(ok=True)

    monkeypatch.setattr(rtc_runtime_module, "add_server_rtc_candidate", fake_add)
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="grpc",
        emit=lambda event: _append([], event),
    )
    record.remote_description_set = True
    record.rtc_peer = object()

    await runtime.dispatch(RtcCandidateCommand(candidate=None))
    await runtime.dispatch(RtcCandidateCommand(candidate=None))
    with pytest.raises(InvalidConfigError, match="after end-of-candidates"):
        await runtime.dispatch(RtcCandidateCommand(candidate="candidate:late"))

    assert len(applied) == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_ice_restart_replaces_peer_and_keeps_control_transport(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    emitted: list[dict] = []
    offers: list[RtcOfferCommand] = []

    class OldPeer:
        closed = 0

        async def close(self):
            self.closed += 1

    old_peer = OldPeer()

    async def fake_exchange(**kwargs):
        offers.append(
            RtcOfferCommand(
                offer_type=kwargs["request"].offer_type,
                sdp=kwargs["request"].sdp,
            )
        )
        record.browser_attached = True
        return RtcOfferAnswer(
            session_id=record.session_id,
            answer_type="answer",
            sdp=f"answer-{len(offers)}",
        )

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="grpc",
        emit=lambda event: _append(emitted, event),
    )
    await runtime.start()
    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-1"))
    record.rtc_peer = old_peer

    with pytest.raises(InvalidConfigError, match="set restart=true"):
        await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-2"))

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-2", restart=True))

    assert old_peer.closed == 1
    assert record.attached_control_transport == "grpc"
    assert record.control_attached is True
    assert record.remote_description_set is True
    assert record.ice_restart_in_progress is False
    assert [event.get("answer", {}).get("sdp") for event in emitted if event["type"] == "rtc.answer"] == [
        "answer-1",
        "answer-2",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_only_one_transport_can_control_a_session():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    first = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append([], event),
    )

    with pytest.raises(RtcControlTransportMismatchError, match="requires pondsocket"):
        RtcRuntime(
            scheduler=FakeScheduler(),
            registry=registry,
            session_id=record.session_id,
            transport="grpc",
            emit=lambda event: _append([], event),
        )

    assert record.attached_control_transport == "pondsocket"
    await first.close()


@pytest.mark.asyncio
async def test_duplicate_control_attach_has_an_explicit_conflict():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    first = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="grpc",
        emit=lambda event: _append([], event),
    )

    with pytest.raises(RtcControlAlreadyAttachedError, match="already has"):
        RtcRuntime(
            scheduler=FakeScheduler(),
            registry=registry,
            session_id=record.session_id,
            transport="grpc",
            emit=lambda event: _append([], event),
        )

    await first.close()


@pytest.mark.asyncio
async def test_runtime_closes_after_background_signaling_failure():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    emitted: list[dict] = []
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="grpc",
        emit=lambda event: _append(emitted, event),
    )
    await runtime.start()

    await record.media_events.put({"type": "rtc.signaling_error", "message": "TURN gathering failed"})
    await asyncio.wait_for(runtime.wait_closed(), timeout=1)

    assert [event["type"] for event in emitted] == [
        "rtc.session.attached",
        "rtc.signaling_error",
        "rtc.session.closed",
    ]
    assert emitted[-1]["reason"] == "signaling_error"
    assert registry.get(record.session_id) is None


@pytest.mark.asyncio
async def test_runtime_emits_closed_after_conversation_cleanup(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    emitted: list[dict] = []
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append(emitted, event),
    )
    await runtime.start()

    async def close_conversation():
        emitted.append({"type": "conversation.cleanup.completed"})

    monkeypatch.setattr(runtime.conversation, "close", close_conversation)
    await runtime.close(reason="client_closed")

    assert [event["type"] for event in emitted] == [
        "rtc.session.attached",
        "conversation.cleanup.completed",
        "rtc.session.closed",
    ]
    assert runtime.is_closed is True


class _PreparedWire:
    def __init__(self, wire: dict) -> None:
        self.wire = wire
        self.done = False


def _runtime_with_emit(registry: RtcSessionRegistry, record, emit) -> RtcRuntime:
    return RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=emit,
    )


@pytest.mark.asyncio
async def test_runtime_retries_lifecycle_critical_emit_once(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    attempts: list[str] = []
    failures = {"response.cancelled": 1}

    async def emit(event: dict) -> bool:
        event_type = event["type"]
        attempts.append(event_type)
        if failures.get(event_type, 0) > 0:
            failures[event_type] -= 1
            return False
        return True

    monkeypatch.setattr(
        rtc_runtime_module,
        "prepare_rtc_control_event",
        lambda **_kwargs: _PreparedWire({"type": "response.cancelled", "response_id": "resp_1"}),
    )
    runtime = _runtime_with_emit(registry, record, emit)

    await runtime._emit_conversation_event(object())

    assert attempts == ["response.cancelled", "response.cancelled"]
    await runtime.close()


@pytest.mark.asyncio
async def test_typed_conversation_handler_retries_critical_and_receives_event_and_wire(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    calls: list[tuple[object, str]] = []
    failures = {"response.cancelled": 1}
    marker = object()

    wire_emits: list[str] = []

    async def emit(event: dict) -> bool:
        wire_emits.append(event["type"])
        return True

    async def emit_conversation(event: object, wire: dict) -> bool:
        calls.append((event, wire["type"]))
        if failures.get(wire["type"], 0) > 0:
            failures[wire["type"]] -= 1
            return False
        return True

    monkeypatch.setattr(
        rtc_runtime_module,
        "prepare_rtc_control_event",
        lambda **_kwargs: _PreparedWire({"type": "response.cancelled", "response_id": "resp_1"}),
    )
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=emit,
        emit_conversation=emit_conversation,
    )

    await runtime._emit_conversation_event(marker)

    assert [wire_type for _, wire_type in calls] == ["response.cancelled", "response.cancelled"]
    assert all(event is marker for event, _ in calls)
    assert "response.cancelled" not in wire_emits
    await runtime.close()


@pytest.mark.asyncio
async def test_typed_conversation_handler_lost_after_retry_logs_error(monkeypatch, caplog):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")

    wire_emits: list[str] = []

    async def emit(event: dict) -> bool:
        wire_emits.append(event["type"])
        return True

    async def emit_conversation(event: object, wire: dict) -> bool:
        return False

    monkeypatch.setattr(
        rtc_runtime_module,
        "prepare_rtc_control_event",
        lambda **_kwargs: _PreparedWire({"type": "response.cancelled", "response_id": "resp_1"}),
    )
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=emit,
        emit_conversation=emit_conversation,
    )

    with caplog.at_level("ERROR"):
        await runtime._emit_conversation_event(object())

    assert any("Lost lifecycle event response.cancelled" in message for message in caplog.messages)
    assert "response.cancelled" not in wire_emits
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_logs_error_when_critical_emit_lost_after_retry(monkeypatch, caplog):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    attempts: list[str] = []

    async def emit(event: dict) -> bool:
        attempts.append(event["type"])
        return event["type"] != "response.audio.clear"

    monkeypatch.setattr(
        rtc_runtime_module,
        "prepare_rtc_control_event",
        lambda **_kwargs: _PreparedWire({"type": "response.audio.clear", "response_id": "resp_1"}),
    )
    runtime = _runtime_with_emit(registry, record, emit)

    with caplog.at_level(logging.ERROR, logger="vox.operations.rtc_runtime"):
        await runtime._emit_conversation_event(object())

    assert attempts.count("response.audio.clear") == 2
    errors = [entry for entry in caplog.records if entry.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "response.audio.clear" in errors[0].getMessage()
    assert record.session_id in errors[0].getMessage()
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_does_not_retry_non_critical_emit(monkeypatch, caplog):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    attempts: list[str] = []

    async def emit(event: dict) -> bool:
        attempts.append(event["type"])
        return False

    monkeypatch.setattr(
        rtc_runtime_module,
        "prepare_rtc_control_event",
        lambda **_kwargs: _PreparedWire(
            {"type": "response.audio.delta", "response_id": "resp_1", "delta": "AAAA"}
        ),
    )
    runtime = _runtime_with_emit(registry, record, emit)

    with caplog.at_level(logging.ERROR, logger="vox.operations.rtc_runtime"):
        await runtime._emit_conversation_event(object())

    assert attempts == ["response.audio.delta"]
    assert [entry for entry in caplog.records if entry.levelno == logging.ERROR] == []
    await runtime.close()


async def _wait_for_emitted(emitted: list[dict], event_type: str, *, timeout: float = 1.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for event in emitted:
            if event["type"] == event_type:
                return event
        await asyncio.sleep(0)
    raise AssertionError(f"event {event_type} was not emitted")


async def _wait_for_candidate(emitted: list[dict], name: str, *, timeout: float = 1.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for event in emitted:
            if event["type"] == "rtc.ice_candidate" and event.get("candidate", {}).get("candidate") == name:
                return event
        await asyncio.sleep(0)
    raise AssertionError(f"candidate {name} was not emitted")


@pytest.mark.asyncio
async def test_runtime_labels_candidate_gathered_during_exchange(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    emitted: list[dict] = []
    exchanges = 0

    async def fake_exchange(**_kwargs):
        nonlocal exchanges
        exchanges += 1
        record.browser_attached = True
        await emit_media_event(
            record, {"type": "rtc.ice_candidate", "candidate": {"candidate": f"mid-{exchanges}"}}
        )
        return RtcOfferAnswer(session_id=record.session_id, answer_type="answer", sdp=f"answer-{exchanges}")

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append(emitted, event),
    )
    await runtime.start()

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-1", generation=1))
    first = await _wait_for_candidate(emitted, "mid-1")
    assert first["generation"] == 1

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-2", restart=True, generation=2))
    second = await _wait_for_candidate(emitted, "mid-2")
    assert second["generation"] == 2
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_stamps_offer_generation_on_answer_candidate_and_error(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    emitted: list[dict] = []

    async def fake_exchange(**_kwargs):
        return RtcOfferAnswer(session_id=record.session_id, answer_type="answer", sdp="answer-sdp")

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append(emitted, event),
    )
    await runtime.start()

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-sdp", generation=7))
    answer = await _wait_for_emitted(emitted, "rtc.answer")
    assert answer["generation"] == 7

    await emit_media_event(
        record, {"type": "rtc.ice_candidate", "candidate": {"candidate": "candidate:1", "sdpMid": "0"}}
    )
    candidate = await _wait_for_emitted(emitted, "rtc.ice_candidate")
    assert candidate["generation"] == 7

    await emit_media_event(record, {"type": "rtc.signaling_error", "message": "TURN gathering failed"})
    await asyncio.wait_for(runtime.wait_closed(), timeout=1)
    error = await _wait_for_emitted(emitted, "rtc.signaling_error")
    assert error["generation"] == 7


@pytest.mark.asyncio
async def test_runtime_labels_candidates_with_current_offer_generation(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    emitted: list[dict] = []

    async def fake_exchange(**_kwargs):
        record.browser_attached = True
        return RtcOfferAnswer(session_id=record.session_id, answer_type="answer", sdp="answer-sdp")

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append(emitted, event),
    )
    await runtime.start()

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-1", generation=1))
    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": {"candidate": "candidate:1"}})
    first = await _wait_for_emitted(emitted, "rtc.ice_candidate")
    assert first["generation"] == 1

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-2", restart=True, generation=2))
    emitted.clear()
    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": {"candidate": "candidate:2"}})
    second = await _wait_for_emitted(emitted, "rtc.ice_candidate")
    assert second["generation"] == 2
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_forwards_pre_restart_candidate_with_its_enqueue_generation(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    emitted: list[dict] = []

    async def fake_exchange(**_kwargs):
        record.browser_attached = True
        return RtcOfferAnswer(session_id=record.session_id, answer_type="answer", sdp="answer-sdp")

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append(emitted, event),
    )
    await runtime.start()

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-1", generation=1))
    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": {"candidate": "pre-restart"}})

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-2", restart=True, generation=2))
    assert record.negotiation_generation == 2

    candidate = await _wait_for_emitted(emitted, "rtc.ice_candidate")
    assert candidate["candidate"]["candidate"] == "pre-restart"
    assert candidate["generation"] == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_omits_generation_when_offer_has_none(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    emitted: list[dict] = []

    async def fake_exchange(**_kwargs):
        return RtcOfferAnswer(session_id=record.session_id, answer_type="answer", sdp="answer-sdp")

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    runtime = RtcRuntime(
        scheduler=FakeScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=lambda event: _append(emitted, event),
    )
    await runtime.start()

    await runtime.dispatch(RtcOfferCommand(offer_type="offer", sdp="offer-sdp"))
    answer = await _wait_for_emitted(emitted, "rtc.answer")
    assert "generation" not in answer

    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": {"candidate": "candidate:1"}})
    candidate = await _wait_for_emitted(emitted, "rtc.ice_candidate")
    assert "generation" not in candidate
    await runtime.close()


async def _append(events: list[dict], event: dict) -> None:
    events.append(event)
