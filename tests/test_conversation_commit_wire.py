"""Wire-level tests for the new response.committed event on both transports."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.fakes import FakeScheduler
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.grpc import vox_pb2
from vox.grpc.conversation_servicer import ConversationServicer
from vox.server.pondsocket_gateway import install_pondsocket_gateway
from vox.server.rtc_registry import RtcSessionRegistry


class QuickTTS(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="quick", type=ModelType.TTS,
            architectures=("quick",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **_):
        yield SynthesizeChunk(
            audio=np.zeros(256, dtype=np.float32).tobytes(),
            sample_rate=24_000, is_final=False,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


DummyScheduler = FakeScheduler


class FakeContext:
    def cancelled(self): return False


def _ws_app() -> FastAPI:
    app = FastAPI()
    app.state.scheduler = DummyScheduler(QuickTTS())
    app.state.rtc_registry = RtcSessionRegistry()
    assert install_pondsocket_gateway(app)
    return app


def _send_channel_message(ws, *, action: str, channel_name: str, event: str, payload: dict, request_id: str) -> None:
    ws.send_json({
        "action": action,
        "channelName": channel_name,
        "requestId": request_id,
        "event": event,
        "payload": payload,
    })


def _receive_json(ws) -> dict:
    payload = ws.receive_json()
    assert isinstance(payload, dict)
    return payload


def _drain_until(ws, predicate, max_events: int = 50) -> list[dict]:
    events: list[dict] = []
    for _ in range(max_events):
        payload = _receive_json(ws)
        events.append(payload)
        if predicate(payload):
            break
    return events


class TestWsCommitWire:
    def test_response_commit_emits_committed_event(self):
        client = TestClient(_ws_app())
        with client.websocket_connect("/v1/socket") as ws:
            _receive_json(ws)
            _send_channel_message(
                ws,
                action="JOIN_CHANNEL",
                channel_name="/conversation/commit-wire",
                event="join",
                payload={},
                request_id="join-conversation",
            )
            _receive_json(ws)

            _send_channel_message(
                ws,
                action="BROADCAST",
                channel_name="/conversation/commit-wire",
                event="session.update",
                payload={"session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"}},
                request_id="session-update",
            )
            _receive_json(ws)

            _send_channel_message(
                ws,
                action="BROADCAST",
                channel_name="/conversation/commit-wire",
                event="response.start",
                payload={},
                request_id="response-start",
            )
            _send_channel_message(
                ws,
                action="BROADCAST",
                channel_name="/conversation/commit-wire",
                event="response.delta",
                payload={"delta": "hi there"},
                request_id="response-delta",
            )
            _send_channel_message(
                ws,
                action="BROADCAST",
                channel_name="/conversation/commit-wire",
                event="response.commit",
                payload={},
                request_id="response-commit",
            )

            events = _drain_until(ws, lambda e: e.get("event") == "response.done")
            types = [e["event"] for e in events]
            assert "response.committed" in types
            assert "response.done" in types
            assert types.index("response.committed") < types.index("response.done")


class TestGrpcCommitWire:
    @pytest.mark.asyncio
    async def test_response_commit_emits_committed_message(self):
        servicer = ConversationServicer(store=None, registry=None, scheduler=DummyScheduler(QuickTTS()))

        client_queue: asyncio.Queue = asyncio.Queue()
        await client_queue.put(vox_pb2.ConverseClientMessage(
            session_update=vox_pb2.ConversationSessionUpdate(
                stt_model="x:1", tts_model="y:1", voice="default",
            ),
        ))
        await client_queue.put(vox_pb2.ConverseClientMessage(
            response_start=vox_pb2.ConversationResponseStart(),
        ))
        await client_queue.put(vox_pb2.ConverseClientMessage(
            response_delta=vox_pb2.ConversationResponseDelta(delta="hi there"),
        ))
        await client_queue.put(vox_pb2.ConverseClientMessage(
            response_commit=vox_pb2.ConversationResponseCommit(),
        ))

        async def client_stream():
            while True:
                item = await client_queue.get()
                if item is None:
                    return
                yield item

        out: list[vox_pb2.ConverseServerMessage] = []

        async def run():
            try:
                async for server_msg in servicer.Converse(client_stream(), FakeContext()):
                    out.append(server_msg)
                    if server_msg.WhichOneof("msg") == "response_done":
                        break
            finally:
                await client_queue.put(None)

        await asyncio.wait_for(run(), timeout=2.0)

        kinds = [m.WhichOneof("msg") for m in out]
        assert "response_committed" in kinds
        assert "response_done" in kinds
        assert kinds.index("response_committed") < kinds.index("response_done")
