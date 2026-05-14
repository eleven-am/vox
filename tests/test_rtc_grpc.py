from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

import vox.grpc.rtc_servicer as rtc_servicer_module
from tests.fakes import FakeScheduler
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.grpc import vox_pb2
from vox.grpc.rtc_servicer import RtcServicer
from vox.operations.conversation import ConvAudioClearEvent, ConvDoneEvent
from vox.server.rtc_registry import RtcSessionRegistry


class FakeDataChannel:
    def __init__(self) -> None:
        self.readyState = "open"
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)


class FakeAudioOutputTrack:
    def __init__(self) -> None:
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1


class ScriptedTTS(TTSAdapter):
    def __init__(self, chunks: int = 2) -> None:
        self._chunks = chunks

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted",
            type=ModelType.TTS,
            architectures=("scripted",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self):
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **_):
        for _ in range(self._chunks):
            yield SynthesizeChunk(
                audio=np.full(256, 0.02, dtype=np.float32).tobytes(),
                sample_rate=24_000,
                is_final=False,
            )
            await asyncio.sleep(0.005)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class FakeContext:
    def cancelled(self) -> bool:
        return False


async def _drive_until(servicer, messages, predicate, *, timeout: float = 2.0, max_items: int = 50):
    client_queue: asyncio.Queue = asyncio.Queue()
    for msg in messages:
        await client_queue.put(msg)

    async def client_stream():
        while True:
            item = await client_queue.get()
            if item is None:
                return
            yield item

    out = []

    async def run():
        gen = servicer.Control(client_stream(), FakeContext())
        try:
            async for server_msg in gen:
                out.append(server_msg)
                if predicate(server_msg):
                    break
                if len(out) >= max_items:
                    break
        finally:
            await client_queue.put(None)

    await asyncio.wait_for(run(), timeout=timeout)
    return out


async def _collect_all(servicer, messages, *, timeout: float = 2.0):
    async def client_stream():
        for item in messages:
            yield item

    out = []

    async def run():
        async for server_msg in servicer.Control(client_stream(), FakeContext()):
            out.append(server_msg)

    await asyncio.wait_for(run(), timeout=timeout)
    return out


@pytest.mark.asyncio
async def test_rtc_grpc_attach_emits_attached_event():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session()
    servicer = RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=registry,
    )

    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.RtcControlClientMessage(
                attach=vox_pb2.RtcControlAttach(session_id=record.session_id),
            )
        ],
        predicate=lambda m: m.WhichOneof("msg") == "rtc_session_attached",
    )

    attached = next(m for m in out if m.WhichOneof("msg") == "rtc_session_attached")
    assert attached.rtc_session_attached.session_id == record.session_id
    assert attached.rtc_session_attached.provider == "webrtc"


@pytest.mark.asyncio
async def test_rtc_grpc_session_update_emits_session_created():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session()
    servicer = RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=registry,
    )

    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.RtcControlClientMessage(
                attach=vox_pb2.RtcControlAttach(session_id=record.session_id),
            ),
            vox_pb2.RtcControlClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                    voice="default",
                ),
            ),
        ],
        predicate=lambda m: m.WhichOneof("msg") == "session_created",
    )

    assert any(m.WhichOneof("msg") == "rtc_session_attached" for m in out)
    created = next(m.session_created for m in out if m.WhichOneof("msg") == "session_created")
    assert created.turn_profile == "default"
    assert created.policy.aec_warmup_ms == 750


@pytest.mark.asyncio
async def test_rtc_grpc_control_drops_audio_delta_events():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session()
    servicer = RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=registry,
    )

    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.RtcControlClientMessage(
                attach=vox_pb2.RtcControlAttach(session_id=record.session_id),
            ),
            vox_pb2.RtcControlClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                    voice="default",
                    sample_rate=48_000,
                ),
            ),
            vox_pb2.RtcControlClientMessage(
                response_delta=vox_pb2.ConversationResponseDelta(delta="hello"),
            ),
            vox_pb2.RtcControlClientMessage(
                response_commit=vox_pb2.ConversationResponseCommit(),
            ),
        ],
        predicate=lambda m: m.WhichOneof("msg") == "response_done",
    )

    assert out
    assert not any(m.WhichOneof("msg") == "audio_delta" for m in out)
    assert any(m.WhichOneof("msg") == "response_done" for m in out)


@pytest.mark.asyncio
async def test_rtc_grpc_audio_clear_clears_webrtc_output_track(monkeypatch):
    class ClearOnlyOrchestrator:
        config = None

        def __init__(self, **_):
            pass

        async def events(self):
            yield ConvAudioClearEvent(response_id="resp_1")
            yield ConvDoneEvent()

        async def end_of_stream(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(rtc_servicer_module, "ConversationOrchestrator", ClearOnlyOrchestrator)

    registry = RtcSessionRegistry()
    record, _ = registry.create_session()
    track = FakeAudioOutputTrack()
    record.audio_output_track = track
    servicer = RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=registry,
    )

    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.RtcControlClientMessage(
                attach=vox_pb2.RtcControlAttach(session_id=record.session_id),
            )
        ],
        predicate=lambda m: m.WhichOneof("msg") == "audio_clear",
    )

    assert track.cleared == 1
    clear = next(m.audio_clear for m in out if m.WhichOneof("msg") == "audio_clear")
    assert clear.response_id == "resp_1"


@pytest.mark.asyncio
async def test_rtc_grpc_requires_attach_first():
    registry = RtcSessionRegistry()
    servicer = RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=registry,
    )

    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.RtcControlClientMessage(
                session_update=vox_pb2.ConversationSessionUpdate(
                    stt_model="x:1",
                    tts_model="y:1",
                ),
            )
        ],
        predicate=lambda m: m.WhichOneof("msg") == "error",
    )

    err = next(m for m in out if m.WhichOneof("msg") == "error")
    assert "attach first" in err.error.message


@pytest.mark.asyncio
async def test_rtc_grpc_rejects_unknown_session():
    servicer = RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=RtcSessionRegistry(),
    )

    out = await _drive_until(
        servicer,
        messages=[
            vox_pb2.RtcControlClientMessage(
                attach=vox_pb2.RtcControlAttach(session_id="rtc_missing"),
            )
        ],
        predicate=lambda m: m.WhichOneof("msg") == "error",
    )

    err = next(m for m in out if m.WhichOneof("msg") == "error")
    assert "unknown, expired, or already attached RTC session" in err.error.message


@pytest.mark.asyncio
async def test_rtc_grpc_relays_client_event_to_browser_data_channel():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session()
    record.data_channel = FakeDataChannel()
    servicer = RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=registry,
    )

    out = await _collect_all(
        servicer,
        messages=[
            vox_pb2.RtcControlClientMessage(
                attach=vox_pb2.RtcControlAttach(session_id=record.session_id),
            ),
            vox_pb2.RtcControlClientMessage(
                client_event=vox_pb2.RtcClientEvent(
                    event="render.image",
                    payload_json=json.dumps({"url": "https://example.com/a.png"}),
                ),
            ),
        ],
    )

    assert any(m.WhichOneof("msg") == "rtc_session_attached" for m in out)
    assert record.data_channel.sent
    assert json.loads(record.data_channel.sent[0]) == {
        "event": "render.image",
        "payload": {
            "url": "https://example.com/a.png",
        },
    }


@pytest.mark.asyncio
async def test_rtc_grpc_emits_browser_events_to_backend():
    registry = RtcSessionRegistry()
    record, _ = registry.create_session()
    servicer = RtcServicer(
        scheduler=FakeScheduler(ScriptedTTS()),
        rtc_registry=registry,
    )

    async def enqueue_browser_event():
        while not record.control_attached:
            await asyncio.sleep(0.005)
        assert record.control_events is not None
        await record.control_events.put(
            {
                "type": "browser.event",
                "session_id": record.session_id,
                "event": "ui.select",
                "payload": {"id": "choice-a"},
            }
        )

    task = asyncio.create_task(enqueue_browser_event())
    try:
        out = await _drive_until(
            servicer,
            messages=[
                vox_pb2.RtcControlClientMessage(
                    attach=vox_pb2.RtcControlAttach(session_id=record.session_id),
                )
            ],
            predicate=lambda m: m.WhichOneof("msg") == "client_event",
        )
    finally:
        await task

    evt = next(m for m in out if m.WhichOneof("msg") == "client_event")
    assert evt.client_event.event == "browser.event"
    assert json.loads(evt.client_event.payload_json) == {
        "session_id": record.session_id,
        "event": "ui.select",
        "payload": {"id": "choice-a"},
    }
