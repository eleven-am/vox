from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import suppress

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("pondsocket")
pytest.importorskip("pondsocket_asgi")

import vox.grpc.conversation_servicer as grpc_servicer_module
import vox.server.pondsocket_events as pondsocket_events_module
import vox.server.pondsocket_gateway as pondsocket_gateway_module
from tests.fakes import FakeScheduler
from tests.test_conversation_session import HangingReleaseScheduler, ScriptedTTSAdapter
from vox.conversation import TurnEvent, TurnEventType
from vox.grpc import vox_pb2
from vox.grpc.conversation_servicer import ConversationServicer
from vox.operations.conversation import ConversationOrchestrator
from vox.server.pondsocket_gateway import install_pondsocket_gateway
from vox.server.rtc_registry import RtcSessionRegistry
from vox.streaming.types import SpeechStarted, SpeechStopped, StreamTranscript

_WIRE_TYPES = frozenset(
    {
        "session.created",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
        "turn.eou.predicted",
        "response.created",
        "response.committed",
        "response.audio.delta",
        "response.audio.clear",
        "response.done",
        "response.cancelled",
        "interruption.detected",
        "interruption.false_positive",
        "turn.state_changed",
        "error",
    }
)

_GRPC_KIND_TO_WIRE = {
    "session_created": "session.created",
    "speech_started": "input_audio_buffer.speech_started",
    "speech_stopped": "input_audio_buffer.speech_stopped",
    "transcript_delta": "conversation.item.input_audio_transcription.delta",
    "transcript_done": "conversation.item.input_audio_transcription.completed",
    "turn_eou_predicted": "turn.eou.predicted",
    "response_created": "response.created",
    "response_committed": "response.committed",
    "audio_delta": "response.audio.delta",
    "audio_clear": "response.audio.clear",
    "response_done": "response.done",
    "response_cancelled": "response.cancelled",
    "interruption_detected": "interruption.detected",
    "interruption_false_positive": "interruption.false_positive",
    "state_changed": "turn.state_changed",
    "error": "error",
}

_NORMALIZED_STRING_FIELDS = (
    "response_id",
    "generation_id",
    "code",
    "reason",
    "state",
    "previous_state",
    "message",
)


def _clean(value):
    return value if value else None


def _normalize_pondsocket_frame(frame: dict) -> dict | None:
    event_name = frame.get("event")
    if event_name not in _WIRE_TYPES:
        return None
    payload = frame.get("payload") or {}
    wire: dict = {"type": event_name}
    for key in _NORMALIZED_STRING_FIELDS:
        if key in payload:
            wire[key] = _clean(payload.get(key))
    if "recoverable" in payload:
        wire["recoverable"] = bool(payload["recoverable"])
    if "sequence" in payload:
        wire["sequence"] = int(payload["sequence"])
    return wire


def _normalize_grpc_message(message: vox_pb2.ConverseServerMessage) -> dict | None:
    kind = message.WhichOneof("msg")
    wire_type = _GRPC_KIND_TO_WIRE.get(kind or "")
    if wire_type is None:
        return None
    body = getattr(message, kind)
    fields = body.DESCRIPTOR.fields_by_name
    wire: dict = {"type": wire_type}
    for key in _NORMALIZED_STRING_FIELDS:
        if key in fields:
            wire[key] = _clean(getattr(body, key))
    if "recoverable" in fields:
        wire["recoverable"] = bool(body.recoverable)
    if "sequence" in fields:
        wire["sequence"] = int(body.sequence)
    return wire


def _grpc_client_message(event_name: str, payload: dict) -> vox_pb2.ConverseClientMessage:
    if event_name == "session.update":
        session = payload["session"]
        update = vox_pb2.ConversationSessionUpdate(
            stt_model=session["stt_model"],
            tts_model=session["tts_model"],
            voice=session.get("voice") or "",
        )
        policy = session.get("turn_policy")
        if policy:
            update.policy.CopyFrom(vox_pb2.ConversationTurnPolicy(**policy))
        return vox_pb2.ConverseClientMessage(session_update=update)
    if event_name == "response.start":
        return vox_pb2.ConverseClientMessage(
            response_start=vox_pb2.ConversationResponseStart(
                generation_id=payload.get("generation_id") or "",
            )
        )
    if event_name == "response.delta":
        return vox_pb2.ConverseClientMessage(
            response_delta=vox_pb2.ConversationResponseDelta(
                delta=payload.get("delta") or "",
                generation_id=payload.get("generation_id") or "",
            )
        )
    if event_name == "response.commit":
        return vox_pb2.ConverseClientMessage(
            response_commit=vox_pb2.ConversationResponseCommit(
                generation_id=payload.get("generation_id") or "",
            )
        )
    if event_name == "response.cancel":
        return vox_pb2.ConverseClientMessage(
            response_cancel=vox_pb2.ConversationResponseCancel(
                generation_id=payload.get("generation_id") or "",
            )
        )
    raise AssertionError(f"unsupported scenario command: {event_name}")


def _session_payload(**policy_overrides) -> dict:
    policy = {
        "min_interrupt_duration_ms": 50,
        "max_endpointing_delay_ms": 200,
        "speaking_interrupt_min_duration_ms": 50,
    }
    policy.update(policy_overrides)
    return {
        "session": {
            "stt_model": "fake-stt:1",
            "tts_model": "fake-tts:1",
            "voice": "default",
            "turn_policy": policy,
        }
    }


def _is(type_: str, **fields):
    def predicate(wire: dict) -> bool:
        if wire["type"] != type_:
            return False
        return all(wire.get(key) == value for key, value in fields.items())

    return predicate


def _types(events: list[dict], *interesting: str) -> list[str]:
    return [wire["type"] for wire in events if wire["type"] in interesting]


async def _drain_session(session, max_iterations: int = 40) -> None:
    for _ in range(max_iterations):
        await asyncio.sleep(0)
        if session._event_queue.empty():
            break


async def _commit_user_turn(session) -> None:
    await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
    await _drain_session(session)


async def _begin_user_speech(session, *, utterance_id: int, timestamp_ms: int) -> None:
    await session._forward_stream_event(SpeechStarted(timestamp_ms=timestamp_ms, utterance_id=utterance_id))
    await _drain_session(session)


async def _end_user_speech_without_transcript(session, *, utterance_id: int, timestamp_ms: int) -> None:
    await session._forward_stream_event(
        SpeechStopped(timestamp_ms=timestamp_ms, expects_transcript=False, utterance_id=utterance_id)
    )
    await _drain_session(session)


async def _push_confirming_partial(session, *, utterance_id: int, text: str, start_ms: int, end_ms: int) -> None:
    await session._forward_stream_event(
        StreamTranscript(
            text=text,
            is_partial=True,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_duration_ms=end_ms - start_ms,
            utterance_id=utterance_id,
        )
    )
    await _drain_session(session)


async def _break_output_flush(session) -> None:
    def broken_flush() -> None:
        raise RuntimeError("flush exploded")

    session._audio_output.flush = broken_flush


async def _current_tts_task(session):
    return session._tts_task


class _FakeGrpcContext:
    def cancelled(self) -> bool:
        return False


class PondSocketTransport:
    name = "pondsocket"

    def __init__(self, scheduler, monkeypatch, *, channel: str = "/conversation/adversarial") -> None:
        self.captured: list[tuple[ConversationOrchestrator, asyncio.AbstractEventLoop]] = []

        def capture(**kwargs):
            orchestrator = ConversationOrchestrator(**kwargs)
            self.captured.append((orchestrator, asyncio.get_running_loop()))
            return orchestrator

        monkeypatch.setattr(pondsocket_gateway_module, "ConversationOrchestrator", capture)
        app = FastAPI()
        app.state.scheduler = scheduler
        app.state.rtc_registry = RtcSessionRegistry()
        assert install_pondsocket_gateway(app)
        self._client = TestClient(app)
        self._ws_ctx = self._client.websocket_connect("/v1/socket")
        self._ws = self._ws_ctx.__enter__()
        connect = self._receive_frame()
        assert connect["action"] == "CONNECT"
        self._channel = channel
        self._request_seq = 0
        self._send_frame(action="JOIN_CHANNEL", event="join", payload={})
        ack = self._receive_frame()
        assert ack["event"] == "ACKNOWLEDGE"
        self.events: list[dict] = []

    def _send_frame(self, *, action: str, event: str, payload: dict) -> None:
        self._request_seq += 1
        self._ws.send_text(
            json.dumps(
                {
                    "action": action,
                    "channelName": self._channel,
                    "requestId": f"req-{self._request_seq}",
                    "event": event,
                    "payload": payload,
                }
            )
        )

    def _receive_frame(self) -> dict:
        message = self._ws.receive()
        assert message.get("text")
        return json.loads(message["text"])

    def send(self, event_name: str, payload: dict) -> None:
        self._send_frame(action="BROADCAST", event=event_name, payload=payload)

    def expect(self, predicate, *, max_events: int = 200) -> dict:
        for _ in range(max_events):
            wire = _normalize_pondsocket_frame(self._receive_frame())
            if wire is None:
                continue
            self.events.append(wire)
            if predicate(wire):
                return wire
        raise AssertionError(f"expected event not observed; saw {self.events}")

    @property
    def session(self):
        orchestrator, _ = self.captured[-1]
        session = orchestrator._session
        assert session is not None
        return session

    def run_on_session_loop(self, coroutine, *, timeout_s: float = 5.0):
        _, loop = self.captured[-1]
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout_s)

    def call_on_session_loop(self, fn) -> None:
        _, loop = self.captured[-1]
        loop.call_soon_threadsafe(fn)

    def close(self) -> None:
        with suppress(Exception):
            self._ws_ctx.__exit__(None, None, None)


class GrpcTransport:
    name = "grpc"

    def __init__(self, scheduler, monkeypatch) -> None:
        self.captured: list[tuple[ConversationOrchestrator, asyncio.AbstractEventLoop]] = []

        def capture(**kwargs):
            orchestrator = ConversationOrchestrator(**kwargs)
            self.captured.append((orchestrator, asyncio.get_running_loop()))
            return orchestrator

        monkeypatch.setattr(grpc_servicer_module, "ConversationOrchestrator", capture)
        self.events: list[dict] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._submit(self._start(scheduler))

    def _submit(self, coroutine, *, timeout_s: float = 10.0):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout_s)

    async def _start(self, scheduler) -> None:
        self._client_queue: asyncio.Queue = asyncio.Queue()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        servicer = ConversationServicer(store=None, registry=None, scheduler=scheduler)
        self._pump_task = asyncio.create_task(self._pump(servicer))

    async def _pump(self, servicer: ConversationServicer) -> None:
        async def client_stream():
            while True:
                item = await self._client_queue.get()
                if item is None:
                    return
                yield item

        async for message in servicer.Converse(client_stream(), _FakeGrpcContext()):
            wire = _normalize_grpc_message(message)
            if wire is not None:
                await self._event_queue.put(wire)

    async def _finish_pump(self) -> None:
        with suppress(Exception):
            await self._pump_task

    def send(self, event_name: str, payload: dict) -> None:
        message = _grpc_client_message(event_name, payload)
        self._submit(self._client_queue.put(message))

    def expect(self, predicate, *, max_events: int = 200) -> dict:
        deadline = time.monotonic() + 10.0
        for _ in range(max_events):
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"expected event not observed in time; saw {self.events}"
            wire = self._submit(
                asyncio.wait_for(self._event_queue.get(), timeout=remaining),
                timeout_s=remaining + 1.0,
            )
            self.events.append(wire)
            if predicate(wire):
                return wire
        raise AssertionError(f"expected event not observed; saw {self.events}")

    @property
    def session(self):
        orchestrator, _ = self.captured[-1]
        session = orchestrator._session
        assert session is not None
        return session

    def run_on_session_loop(self, coroutine, *, timeout_s: float = 5.0):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout_s)

    def call_on_session_loop(self, fn) -> None:
        self._loop.call_soon_threadsafe(fn)

    def close(self) -> None:
        with suppress(Exception):
            self._submit(self._client_queue.put(None))
            self._submit(self._finish_pump(), timeout_s=10.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._loop.is_closed():
            self._loop.close()


@pytest.fixture(params=["pondsocket", "grpc"])
def transport_factory(request, monkeypatch):
    transports: list[PondSocketTransport | GrpcTransport] = []

    def factory(scheduler):
        transport = (
            PondSocketTransport(scheduler, monkeypatch)
            if request.param == "pondsocket"
            else GrpcTransport(scheduler, monkeypatch)
        )
        transports.append(transport)
        return transport

    yield factory
    for transport in reversed(transports):
        transport.close()


class TestConfirmedBargeInMidStream:
    def test_barge_in_emits_clear_cancel_and_stale_errors_then_recovers(self, transport_factory):
        transport = transport_factory(FakeScheduler(ScriptedTTSAdapter(chunks=60, inter_chunk_delay=0.01)))
        transport.send("session.update", _session_payload())
        transport.expect(_is("session.created"))

        transport.send("response.start", {"generation_id": "gen-1"})
        created = transport.expect(_is("response.created", generation_id="gen-1"))
        response_id = created["response_id"]
        assert response_id

        transport.send(
            "response.delta",
            {"delta": "A long reply that keeps going with plenty of sentences.", "generation_id": "gen-1"},
        )
        transport.expect(_is("turn.state_changed", state="speaking"))
        transport.expect(_is("response.audio.delta"))

        transport.run_on_session_loop(_begin_user_speech(transport.session, utterance_id=1, timestamp_ms=1_000))
        transport.run_on_session_loop(
            _push_confirming_partial(
                transport.session,
                utterance_id=1,
                text="wait please stop",
                start_ms=1_000,
                end_ms=1_600,
            )
        )

        detected = transport.expect(_is("interruption.detected", generation_id="gen-1"))
        assert detected["response_id"] == response_id
        cleared = transport.expect(_is("response.audio.clear", generation_id="gen-1"))
        assert cleared["response_id"] == response_id
        cancelled = transport.expect(_is("response.cancelled", generation_id="gen-1"))
        assert cancelled["response_id"] == response_id

        transport.send("response.delta", {"delta": "late delta", "generation_id": "gen-1"})
        late_delta_error = transport.expect(_is("error", code="response_stale_generation"))
        assert late_delta_error["generation_id"] == "gen-1"
        assert late_delta_error["recoverable"] is True

        transport.send("response.commit", {"generation_id": "gen-1"})
        late_commit_error = transport.expect(
            _is("error", code="response_stale_generation", generation_id="gen-1")
        )
        assert late_commit_error["recoverable"] is True

        transport.run_on_session_loop(
            _end_user_speech_without_transcript(transport.session, utterance_id=1, timestamp_ms=1_800)
        )
        transport.run_on_session_loop(_commit_user_turn(transport.session))

        transport.send("response.start", {"generation_id": "gen-2"})
        transport.expect(_is("response.created", generation_id="gen-2"))
        transport.send("response.delta", {"delta": "Recovered turn.", "generation_id": "gen-2"})
        transport.send("response.commit", {"generation_id": "gen-2"})
        done = transport.expect(_is("response.done", generation_id="gen-2"))
        assert done["response_id"] != response_id

        lifecycle = _types(
            transport.events,
            "response.created",
            "interruption.detected",
            "response.audio.clear",
            "response.cancelled",
            "response.done",
        )
        assert lifecycle == [
            "response.created",
            "interruption.detected",
            "response.audio.clear",
            "response.cancelled",
            "response.created",
            "response.done",
        ]
        assert all(
            wire.get("generation_id") == "gen-2" for wire in transport.events if wire["type"] == "response.done"
        )


class TestStartRejectedDuringUserSpeech:
    def test_typed_rejection_then_retry_succeeds_after_false_positive(self, transport_factory):
        transport = transport_factory(FakeScheduler(ScriptedTTSAdapter(chunks=3, inter_chunk_delay=0.005)))
        transport.send("session.update", _session_payload(min_interrupt_duration_ms=5_000))
        transport.expect(_is("session.created"))

        transport.run_on_session_loop(_commit_user_turn(transport.session))
        transport.expect(_is("turn.state_changed", state="thinking"))

        transport.run_on_session_loop(_begin_user_speech(transport.session, utterance_id=7, timestamp_ms=2_000))
        transport.expect(_is("input_audio_buffer.speech_started"))

        transport.send("response.start", {"generation_id": "gen-reject"})
        rejected = transport.expect(_is("error", code="response_rejected_user_speech"))
        assert rejected["generation_id"] == "gen-reject"
        assert rejected["recoverable"] is True

        transport.run_on_session_loop(
            _end_user_speech_without_transcript(transport.session, utterance_id=7, timestamp_ms=2_400)
        )
        false_positive = transport.expect(_is("interruption.false_positive"))
        assert false_positive.get("reason") == "no_transcript"

        transport.send("response.start", {"generation_id": "gen-retry"})
        transport.expect(_is("response.created", generation_id="gen-retry"))
        transport.send("response.delta", {"delta": "Hello again.", "generation_id": "gen-retry"})
        transport.send("response.commit", {"generation_id": "gen-retry"})
        transport.expect(_is("response.committed", generation_id="gen-retry"))
        transport.expect(_is("response.done", generation_id="gen-retry"))

        assert _types(transport.events, "response.cancelled") == []
        assert _types(transport.events, "response.created", "response.done") == [
            "response.created",
            "response.done",
        ]

    def test_retry_succeeds_after_acoustic_false_positive_before_vad_stops(self, transport_factory):
        transport = transport_factory(FakeScheduler(ScriptedTTSAdapter(chunks=3, inter_chunk_delay=0.005)))
        transport.send("session.update", _session_payload(min_interrupt_duration_ms=5_000))
        transport.expect(_is("session.created"))

        transport.run_on_session_loop(_commit_user_turn(transport.session))
        transport.expect(_is("turn.state_changed", state="thinking"))
        transport.run_on_session_loop(_begin_user_speech(transport.session, utterance_id=8, timestamp_ms=3_000))
        transport.expect(_is("input_audio_buffer.speech_started"))

        transport.send("response.start", {"generation_id": "gen-reject"})
        transport.expect(
            _is(
                "error",
                code="response_rejected_user_speech",
                generation_id="gen-reject",
            )
        )

        transport.run_on_session_loop(transport.session._evaluate_interrupt_candidate())
        false_positive = transport.expect(_is("interruption.false_positive"))
        assert false_positive.get("reason") == "insufficient_acoustic_evidence"
        assert transport.session._input_speech_active is True

        transport.send("response.start", {"generation_id": "gen-retry"})
        transport.expect(_is("response.created", generation_id="gen-retry"))
        transport.send("response.delta", {"delta": "The retry was admitted.", "generation_id": "gen-retry"})
        transport.send("response.commit", {"generation_id": "gen-retry"})
        transport.expect(_is("response.committed", generation_id="gen-retry"))
        transport.expect(_is("response.done", generation_id="gen-retry"))

    def test_every_rejected_start_has_a_correlated_error(self, transport_factory):
        transport = transport_factory(FakeScheduler(ScriptedTTSAdapter(chunks=1)))
        transport.send("session.update", _session_payload())
        transport.expect(_is("session.created"))

        transport.run_on_session_loop(_commit_user_turn(transport.session))
        transport.expect(_is("turn.state_changed", state="thinking"))
        transport.run_on_session_loop(_begin_user_speech(transport.session, utterance_id=9, timestamp_ms=4_000))
        transport.expect(_is("input_audio_buffer.speech_started"))

        for generation_id in ("gen-reject-1", "gen-reject-2"):
            transport.send("response.start", {"generation_id": generation_id})
            rejected = transport.expect(
                _is(
                    "error",
                    code="response_rejected_user_speech",
                    generation_id=generation_id,
                )
            )
            assert rejected["recoverable"] is True


class TestRecoverableVsFatalErrors:
    def test_stale_delta_without_start_is_recoverable_and_session_survives(self, transport_factory):
        transport = transport_factory(FakeScheduler(ScriptedTTSAdapter(chunks=3, inter_chunk_delay=0.005)))
        transport.send("session.update", _session_payload())
        transport.expect(_is("session.created"))

        transport.send("response.delta", {"delta": "ghost delta", "generation_id": "gen-ghost"})
        ghost = transport.expect(_is("error", code="response_stale_generation"))
        assert ghost["generation_id"] == "gen-ghost"
        assert ghost["recoverable"] is True

        transport.send("response.start", {"generation_id": "gen-a"})
        transport.expect(_is("response.created", generation_id="gen-a"))
        transport.send("response.delta", {"delta": "still alive", "generation_id": "gen-a"})
        transport.send("response.commit", {"generation_id": "gen-a"})
        transport.expect(_is("response.committed", generation_id="gen-a"))
        transport.expect(_is("response.done", generation_id="gen-a"))

    def test_action_failure_during_cancel_surfaces_fatal_session_failed(self, transport_factory):
        transport = transport_factory(FakeScheduler(ScriptedTTSAdapter(chunks=60, inter_chunk_delay=0.01)))
        transport.send("session.update", _session_payload())
        transport.expect(_is("session.created"))

        transport.send("response.start", {"generation_id": "gen-b"})
        created = transport.expect(_is("response.created", generation_id="gen-b"))
        transport.send(
            "response.delta",
            {"delta": "A long reply heading for a broken output path.", "generation_id": "gen-b"},
        )
        transport.expect(_is("turn.state_changed", state="speaking"))

        transport.run_on_session_loop(_break_output_flush(transport.session))
        transport.send("response.cancel", {"generation_id": "gen-b"})

        fatal = transport.expect(_is("error", code="session_failed"))
        assert fatal["recoverable"] is False
        cleared = transport.expect(_is("response.audio.clear", generation_id="gen-b"))
        assert cleared["response_id"] == created["response_id"]
        cancelled = transport.expect(_is("response.cancelled", generation_id="gen-b"))
        assert cancelled["response_id"] == created["response_id"]
        transport.expect(_is("turn.state_changed", state="idle"))

        assert _types(transport.events, "response.done") == []


class TestStubbornTtsTeardownFromTransport:
    def test_session_stays_responsive_to_transport_commands_during_hang(self, transport_factory):
        adapter = ScriptedTTSAdapter(chunks=60, inter_chunk_delay=0.01)
        scheduler = HangingReleaseScheduler(adapter)
        transport = transport_factory(scheduler)
        transport.send("session.update", _session_payload())
        transport.expect(_is("session.created"))

        transport.send("response.start", {"generation_id": "gen-slow"})
        created = transport.expect(_is("response.created", generation_id="gen-slow"))
        transport.send(
            "response.delta",
            {"delta": "A very long reply that will be cancelled mid-flight.", "generation_id": "gen-slow"},
        )
        transport.expect(_is("turn.state_changed", state="speaking"))

        old_task = transport.run_on_session_loop(_current_tts_task(transport.session))
        assert old_task is not None

        cancel_started = time.monotonic()
        transport.send("response.cancel", {"generation_id": "gen-slow"})
        cleared = transport.expect(_is("response.audio.clear", generation_id="gen-slow"))
        assert cleared["response_id"] == created["response_id"]
        cancelled = transport.expect(_is("response.cancelled", generation_id="gen-slow"))
        assert cancelled["response_id"] == created["response_id"]
        assert time.monotonic() - cancel_started < 3.0
        assert not old_task.done()

        transport.send("response.start", {"generation_id": "gen-next"})
        transport.expect(_is("response.created", generation_id="gen-next"))
        assert not old_task.done()

        transport.call_on_session_loop(scheduler.release_gate.set)
        transport.send("response.delta", {"delta": "Fresh reply after the hang.", "generation_id": "gen-next"})
        transport.send("response.commit", {"generation_id": "gen-next"})
        done = transport.expect(_is("response.done", generation_id="gen-next"))
        assert done["response_id"] != created["response_id"]


class TestPondSocketLostLifecycleDelivery:
    def test_lost_cancelled_broadcast_retries_and_backend_sequence_stays_coherent(self, monkeypatch):
        scheduler = FakeScheduler(ScriptedTTSAdapter(chunks=60, inter_chunk_delay=0.01))
        transport = PondSocketTransport(scheduler, monkeypatch)
        try:
            real_broadcast = pondsocket_events_module.broadcast_wire_to_user
            attempts: list[str] = []

            async def flaky_broadcast(channel, user_id, wire):
                if wire.get("type") == "response.cancelled":
                    attempts.append(wire["type"])
                    if len(attempts) == 1:
                        raise RuntimeError("simulated transport drop")
                await real_broadcast(channel, user_id, wire)

            monkeypatch.setattr(pondsocket_events_module, "broadcast_wire_to_user", flaky_broadcast)

            transport.send("session.update", _session_payload())
            transport.expect(_is("session.created"))
            transport.send("response.start", {"generation_id": "gen-lossy"})
            created = transport.expect(_is("response.created", generation_id="gen-lossy"))
            transport.send(
                "response.delta",
                {"delta": "A long reply crossing a lossy transport.", "generation_id": "gen-lossy"},
            )
            transport.expect(_is("turn.state_changed", state="speaking"))

            transport.send("response.cancel", {"generation_id": "gen-lossy"})
            cancelled = transport.expect(_is("response.cancelled", generation_id="gen-lossy"))
            assert cancelled["response_id"] == created["response_id"]
            assert attempts == ["response.cancelled", "response.cancelled"]

            transport.send("response.start", {"generation_id": "gen-after-loss"})
            transport.expect(_is("response.created", generation_id="gen-after-loss"))
            transport.send(
                "response.delta",
                {"delta": "Recovered after the drop.", "generation_id": "gen-after-loss"},
            )
            transport.send("response.commit", {"generation_id": "gen-after-loss"})
            transport.expect(_is("response.committed", generation_id="gen-after-loss"))
            transport.expect(_is("response.done", generation_id="gen-after-loss"))

            assert len([wire for wire in transport.events if wire["type"] == "response.cancelled"]) == 1
            assert _types(
                transport.events,
                "response.audio.clear",
                "response.cancelled",
                "response.created",
                "response.done",
            ) == [
                "response.created",
                "response.audio.clear",
                "response.cancelled",
                "response.created",
                "response.done",
            ]
        finally:
            transport.close()
