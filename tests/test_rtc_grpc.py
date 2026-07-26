from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

import vox.operations.rtc_runtime as rtc_runtime_module
from tests.fakes import FakeScheduler
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.grpc import vox_pb2
from vox.grpc.rtc_servicer import RtcServicer
from vox.operations.rtc_signaling import RtcCandidateResult, RtcOfferAnswer
from vox.server.rtc_registry import RtcSessionRegistry


class FakeDataChannel:
    def __init__(self) -> None:
        self.readyState = "open"
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)


class ScriptedTTS(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted",
            type=ModelType.TTS,
            architectures=("scripted",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *args, **kwargs): ...
    def unload(self): ...

    @property
    def is_loaded(self):
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **kwargs):
        yield SynthesizeChunk(
            audio=np.full(256, 0.02, dtype=np.float32).tobytes(),
            sample_rate=24_000,
            is_final=False,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class FakeContext:
    def cancelled(self) -> bool:
        return False


def _servicer(registry: RtcSessionRegistry) -> RtcServicer:
    return RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=registry,
    )


def _attach(session_id: str) -> vox_pb2.RtcControlClientMessage:
    return vox_pb2.RtcControlClientMessage(attach=vox_pb2.RtcControlAttach(session_id=session_id))


def _offer(
    sdp: str = "offer-sdp",
    *,
    restart: bool = False,
    generation: int | None = None,
) -> vox_pb2.RtcControlClientMessage:
    offer = vox_pb2.RtcControlOffer(
        offer=vox_pb2.RtcSessionDescription(type="offer", sdp=sdp),
        restart=restart,
    )
    if generation is not None:
        offer.generation = generation
    return vox_pb2.RtcControlClientMessage(offer=offer)


async def _collect_all(servicer: RtcServicer, messages, *, timeout: float = 2.0):
    async def client_stream():
        for item in messages:
            yield item

    output = []

    async def run():
        async for server_message in servicer.Control(client_stream(), FakeContext()):
            output.append(server_message)

    await asyncio.wait_for(run(), timeout=timeout)
    return output


async def _drive_until(
    servicer: RtcServicer,
    messages,
    predicate,
    *,
    timeout: float = 2.0,
):
    client_queue: asyncio.Queue = asyncio.Queue()
    for message in messages:
        await client_queue.put(message)

    async def client_stream():
        while True:
            item = await client_queue.get()
            if item is None:
                return
            yield item

    output = []

    async def run():
        generator = servicer.Control(client_stream(), FakeContext())
        try:
            async for server_message in generator:
                output.append(server_message)
                if predicate(server_message):
                    break
        finally:
            await generator.aclose()

    await asyncio.wait_for(run(), timeout=timeout)
    return output


def _conversation_kind(message: vox_pb2.RtcControlServerMessage) -> str | None:
    if message.WhichOneof("msg") != "conversation":
        return None
    return message.conversation.WhichOneof("msg")


@pytest.mark.asyncio
async def test_rtc_grpc_create_session_returns_private_bootstrap():
    registry = RtcSessionRegistry(attach_ttl_s=45)

    response = await _servicer(registry).CreateSession(
        vox_pb2.RtcCreateSessionRequest(browser_events=False),
        FakeContext(),
    )

    record = registry.get(response.session_id, now=0)
    assert record is not None
    assert record.expected_control_transport == "grpc"
    assert record.forward_browser_events is False
    assert not hasattr(response, "client_token")
    assert response.attach_ttl_seconds == 45


@pytest.mark.asyncio
async def test_rtc_grpc_full_signaling_uses_one_ordered_stream(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    applied_candidates = []

    async def fake_exchange(**_kwargs):
        return RtcOfferAnswer(
            session_id=record.session_id,
            answer_type="answer",
            sdp="answer-sdp",
        )

    async def fake_add(**kwargs):
        applied_candidates.append(kwargs["request"])
        return RtcCandidateResult(ok=True)

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    monkeypatch.setattr(rtc_runtime_module, "add_server_rtc_candidate", fake_add)
    output = await _collect_all(
        _servicer(registry),
        [
            _attach(record.session_id),
            vox_pb2.RtcControlClientMessage(
                candidate=vox_pb2.RtcIceCandidate(
                    candidate="candidate:first",
                    sdp_mid="0",
                    sdp_m_line_index=0,
                    username_fragment="ufrag",
                    generation=4,
                )
            ),
            vox_pb2.RtcControlClientMessage(candidates_complete=vox_pb2.RtcIceCandidatesComplete(generation=4)),
            _offer(generation=4),
            vox_pb2.RtcControlClientMessage(close=vox_pb2.RtcControlClose(reason="test_complete")),
        ],
    )

    assert [message.WhichOneof("msg") for message in output] == [
        "attached",
        "answer",
        "closed",
    ]
    assert output[0].attached.session_id == record.session_id
    assert output[0].attached.provider == "grpc"
    assert output[1].answer.answer.sdp == "answer-sdp"
    assert output[1].answer.generation == 4
    assert [candidate.candidate for candidate in applied_candidates] == [
        "candidate:first",
        None,
    ]
    assert applied_candidates[0].sdp_m_line_index == 0
    assert [candidate.generation for candidate in applied_candidates] == [4, 4]


@pytest.mark.asyncio
async def test_rtc_grpc_failed_answer_delivery_rolls_back_offer(monkeypatch):
    import vox.grpc.rtc_servicer as rtc_servicer_module

    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    committed = 0
    rolled_back = 0
    original_put = rtc_servicer_module.put_grpc_output_queue

    async def commit() -> None:
        nonlocal committed
        committed += 1

    async def rollback() -> None:
        nonlocal rolled_back
        rolled_back += 1

    async def fake_exchange(**_kwargs):
        return RtcOfferAnswer(
            session_id=record.session_id,
            answer_type="answer",
            sdp="answer-sdp",
            commit_callback=commit,
            rollback_callback=rollback,
        )

    async def fail_answer(queue, item, *, consumer_closed):
        if item.WhichOneof("msg") == "answer":
            return False
        return await original_put(queue, item, consumer_closed=consumer_closed)

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    monkeypatch.setattr(rtc_servicer_module, "put_grpc_output_queue", fail_answer)

    output = await _collect_all(
        _servicer(registry),
        [_attach(record.session_id), _offer(generation=4)],
    )

    assert [message.WhichOneof("msg") for message in output] == ["attached", "error", "closed"]
    assert committed == 0
    assert rolled_back == 1


@pytest.mark.asyncio
async def test_rtc_grpc_server_candidate_and_completion_are_typed_messages():
    from vox.grpc.rtc_messages import rtc_runtime_event_pb

    candidate = rtc_runtime_event_pb(
        {
            "type": "rtc.ice_candidate",
            "generation": 6,
            "candidate": {
                "candidate": "candidate:server",
                "sdpMid": "audio",
                "sdpMLineIndex": 0,
            },
        }
    )
    complete = rtc_runtime_event_pb({"type": "rtc.ice_candidate", "candidate": None, "generation": 6})

    assert candidate.WhichOneof("msg") == "candidate"
    assert candidate.candidate.sdp_mid == "audio"
    assert candidate.candidate.sdp_m_line_index == 0
    assert candidate.candidate.generation == 6
    assert complete.WhichOneof("msg") == "candidates_complete"
    assert complete.candidates_complete.generation == 6


def test_rtc_grpc_answer_preserves_negotiation_generation():
    from vox.grpc.rtc_messages import rtc_runtime_event_pb

    answer = rtc_runtime_event_pb(
        {
            "type": "rtc.answer",
            "session_id": "rtc_1",
            "generation": 11,
            "answer": {"type": "answer", "sdp": "answer-sdp"},
        }
    )

    assert answer.answer.generation == 11


def test_rtc_grpc_signaling_error_preserves_negotiation_generation():
    from vox.grpc.rtc_messages import rtc_runtime_event_pb

    error = rtc_runtime_event_pb(
        {
            "type": "rtc.signaling_error",
            "message": "failed",
            "generation": 13,
        }
    )

    assert error.error.generation == 13


@pytest.mark.asyncio
async def test_rtc_grpc_requires_attach_first():
    output = await _collect_all(
        _servicer(RtcSessionRegistry()),
        [_offer()],
    )

    assert output[0].WhichOneof("msg") == "error"
    assert "attach first" in output[0].error.message


@pytest.mark.asyncio
async def test_rtc_grpc_dispatch_error_preserves_offer_generation():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    output = await _collect_all(
        _servicer(registry),
        [_attach(record.session_id), _offer(restart=True, generation=17)],
    )

    error = next(message.error for message in output if message.WhichOneof("msg") == "error")
    assert error.generation == 17


@pytest.mark.asyncio
async def test_rtc_grpc_rejects_unknown_or_already_controlled_session():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    assert registry.attach_control(record.session_id, transport="grpc") is record

    missing = await _collect_all(_servicer(registry), [_attach("rtc_missing")])
    duplicate = await _collect_all(_servicer(registry), [_attach(record.session_id)])

    assert missing[0].WhichOneof("msg") == "error"
    assert "not found" in missing[0].error.message
    assert duplicate[0].WhichOneof("msg") == "error"
    assert "already has" in duplicate[0].error.message


@pytest.mark.asyncio
async def test_rtc_grpc_rejects_pondsocket_owned_session():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")

    output = await _collect_all(_servicer(registry), [_attach(record.session_id)])

    assert output[0].WhichOneof("msg") == "error"
    assert output[0].error.message == (f"RTC session '{record.session_id}' requires pondsocket control; received grpc")
    assert registry.get(record.session_id) is record
    assert record.control_attached is False


@pytest.mark.asyncio
async def test_rtc_grpc_conversation_events_are_nested_after_offer(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    async def fake_exchange(**_kwargs):
        return RtcOfferAnswer(
            session_id=record.session_id,
            answer_type="answer",
            sdp="answer-sdp",
        )

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    output = await _drive_until(
        _servicer(registry),
        [
            _attach(record.session_id),
            _offer(),
            vox_pb2.RtcControlClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                    voice="default",
                )
            ),
        ],
        predicate=lambda message: _conversation_kind(message) == "session_created",
    )

    created = next(
        message.conversation.session_created for message in output if _conversation_kind(message) == "session_created"
    )
    assert created.turn_profile == "default"
    assert created.policy.aec_warmup_ms == 750


@pytest.mark.asyncio
async def test_rtc_grpc_relays_client_event_to_browser_after_offer(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    channel = FakeDataChannel()
    record.data_channel = channel

    async def fake_exchange(**_kwargs):
        return RtcOfferAnswer(
            session_id=record.session_id,
            answer_type="answer",
            sdp="answer-sdp",
        )

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)
    await _collect_all(
        _servicer(registry),
        [
            _attach(record.session_id),
            _offer(),
            vox_pb2.RtcControlClientMessage(
                client_event=vox_pb2.RtcClientEvent(
                    event="render.image",
                    payload_json=json.dumps({"url": "https://example.com/a.png"}),
                )
            ),
        ],
    )

    assert json.loads(channel.sent[0]) == {
        "event": "render.image",
        "payload": {"url": "https://example.com/a.png"},
    }


@pytest.mark.asyncio
async def test_rtc_grpc_emits_browser_events_as_typed_events():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    async def enqueue_browser_event():
        while not record.control_attached:
            await asyncio.sleep(0)
        await record.control_events.put(
            {
                "type": "browser.event",
                "session_id": record.session_id,
                "event": "ui.select",
                "payload": {"id": "choice-a"},
            }
        )

    task = asyncio.create_task(enqueue_browser_event())
    output = await _drive_until(
        _servicer(registry),
        [_attach(record.session_id)],
        predicate=lambda message: message.WhichOneof("msg") == "browser_event",
    )
    await task

    event = next(message.browser_event for message in output if message.WhichOneof("msg") == "browser_event")
    assert event.event == "ui.select"
    assert json.loads(event.payload_json) == {"id": "choice-a"}


@pytest.mark.asyncio
async def test_rtc_grpc_stream_end_closes_session_once():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    output = await _collect_all(_servicer(registry), [_attach(record.session_id)])

    assert [message.WhichOneof("msg") for message in output] == ["attached", "closed"]
    assert registry.get(record.session_id) is None
    assert record.closed is True


@pytest.mark.asyncio
async def test_rtc_grpc_signaling_failure_closes_the_control_stream():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    client_queue: asyncio.Queue[vox_pb2.RtcControlClientMessage] = asyncio.Queue()
    await client_queue.put(_attach(record.session_id))

    async def client_stream():
        while True:
            yield await client_queue.get()

    async def fail_signaling():
        while not record.control_attached:
            await asyncio.sleep(0)
        await record.media_events.put({"type": "rtc.signaling_error", "message": "TURN gathering failed"})

    failure = asyncio.create_task(fail_signaling())
    output: list[vox_pb2.RtcControlServerMessage] = []

    async def collect_until_closed():
        async for message in _servicer(registry).Control(client_stream(), FakeContext()):
            output.append(message)

    await asyncio.wait_for(collect_until_closed(), timeout=1)
    await failure

    assert [message.WhichOneof("msg") for message in output] == [
        "attached",
        "error",
        "closed",
    ]
    assert registry.get(record.session_id) is None
