from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import vox.operations.rtc_runtime as rtc_runtime_module

pytest.importorskip("pondsocket")
pytest.importorskip("pondsocket_asgi")

from tests.fakes import FakeScheduler
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.operations.rtc_signaling import RtcOfferAnswer
from vox.server.pondsocket_gateway import install_pondsocket_gateway
from vox.server.routes.rtc import router as rtc_router
from vox.server.rtc_registry import RtcSessionRegistry


class ScriptedTTS(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted-tts",
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
        yield SynthesizeChunk(
            audio=np.full(512, 0.01, dtype=np.float32).tobytes(),
            sample_rate=24_000,
            is_final=False,
        )
        await asyncio.sleep(0.01)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


def _build_app() -> FastAPI:
    app = FastAPI()
    app.state.scheduler = FakeScheduler(ScriptedTTS())
    app.state.rtc_registry = RtcSessionRegistry()
    app.include_router(rtc_router)
    assert install_pondsocket_gateway(app)
    return app


def _send_channel_message(ws, *, action: str, channel_name: str, event: str, payload: dict, request_id: str) -> None:
    ws.send_text(
        json.dumps(
            {
                "action": action,
                "channelName": channel_name,
                "requestId": request_id,
                "event": event,
                "payload": payload,
            }
        )
    )


def _receive_json(ws) -> dict:
    payload = ws.receive()
    assert payload.get("text")
    return json.loads(payload["text"])


def _drain_until(ws, predicate, max_events: int = 50) -> list[dict]:
    events: list[dict] = []
    for _ in range(max_events):
        try:
            msg = ws.receive()
        except WebSocketDisconnect:
            break
        if msg.get("type") == "websocket.disconnect":
            break
        if "text" in msg and msg["text"]:
            payload = json.loads(msg["text"])
            events.append(payload)
            if predicate(payload):
                break
    return events


def test_pondsocket_conversation_channel_matches_ws_flow():
    client = TestClient(_build_app())

    with client.websocket_connect("/v1/socket") as ws:
        connect = _receive_json(ws)
        assert connect["action"] == "CONNECT"
        assert connect["event"] == "CONNECTION"

        _send_channel_message(
            ws,
            action="JOIN_CHANNEL",
            channel_name="/conversation/demo",
            event="join",
            payload={},
            request_id="join-conversation",
        )
        ack = _receive_json(ws)
        assert ack["event"] == "ACKNOWLEDGE"
        assert ack["channelName"] == "/conversation/demo"

        _send_channel_message(
            ws,
            action="BROADCAST",
            channel_name="/conversation/demo",
            event="session.update",
            payload={
                "session": {
                    "stt_model": "fake-stt:1",
                    "tts_model": "fake-tts:1",
                    "voice": "default",
                    "turn_profile": "browser_default",
                },
            },
            request_id="session-update",
        )
        created = _receive_json(ws)
        assert created["event"] == "session.created"
        assert created["payload"]["session"]["turn_profile"] == "browser_default"

        _send_channel_message(
            ws,
            action="BROADCAST",
            channel_name="/conversation/demo",
            event="response.start",
            payload={},
            request_id="start-1",
        )
        _send_channel_message(
            ws,
            action="BROADCAST",
            channel_name="/conversation/demo",
            event="response.delta",
            payload={"delta": "hello"},
            request_id="delta-1",
        )
        _send_channel_message(
            ws,
            action="BROADCAST",
            channel_name="/conversation/demo",
            event="response.commit",
            payload={},
            request_id="commit-1",
        )

        events = _drain_until(ws, lambda e: e.get("event") == "response.done")
        deltas = [e for e in events if e.get("event") == "response.audio.delta"]
        assert deltas
        decoded = base64.b64decode(deltas[0]["payload"]["audio"])
        assert decoded


def test_pondsocket_socket_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    client = TestClient(_build_app())

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/socket"):
        pass

    with client.websocket_connect("/v1/socket?api_key=secret") as ws:
        connect = _receive_json(ws)
        assert connect["action"] == "CONNECT"
        assert connect["event"] == "CONNECTION"


def test_pondsocket_socket_can_multiplex_conversation_and_rtc_channels():
    client = TestClient(_build_app())
    rtc_session = client.post("/v1/rtc/sessions").json()

    with client.websocket_connect("/v1/socket") as ws:
        _receive_json(ws)

        _send_channel_message(
            ws,
            action="JOIN_CHANNEL",
            channel_name="/conversation/alpha",
            event="join",
            payload={},
            request_id="join-conversation",
        )
        conv_ack = _receive_json(ws)
        assert conv_ack["event"] == "ACKNOWLEDGE"
        assert conv_ack["channelName"] == "/conversation/alpha"

        _send_channel_message(
            ws,
            action="JOIN_CHANNEL",
            channel_name=f"/rtc/{rtc_session['session_id']}",
            event="join",
            payload={},
            request_id="join-rtc",
        )
        rtc_ack = _receive_json(ws)
        rtc_attached = _receive_json(ws)
        assert rtc_ack["event"] == "ACKNOWLEDGE"
        assert rtc_attached["event"] == "rtc.session.attached"
        assert rtc_attached["payload"]["session_id"] == rtc_session["session_id"]

        _send_channel_message(
            ws,
            action="BROADCAST",
            channel_name=f"/rtc/{rtc_session['session_id']}",
            event="session.update",
            payload={
                "session": {
                    "stt_model": "fake-stt:1",
                    "tts_model": "fake-tts:1",
                    "voice": "default",
                    "turn_profile": "speakerphone",
                },
            },
            request_id="rtc-session-update",
        )

        created = _receive_json(ws)
        assert created["event"] == "session.created"
        assert created["channelName"] == f"/rtc/{rtc_session['session_id']}"
        assert created["payload"]["session_id"] == rtc_session["session_id"]
        assert created["payload"]["session"]["turn_profile"] == "speakerphone"


def test_pondsocket_rtc_offer_reaches_rtc_runtime_and_returns_answer(monkeypatch):
    app = _build_app()
    client = TestClient(app)
    rtc_session = client.post("/v1/rtc/sessions").json()

    async def fake_exchange(*, registry, request):
        assert registry is app.state.rtc_registry
        assert request.session_id == rtc_session["session_id"]
        assert request.offer_type == "offer"
        assert request.sdp == "offer-sdp"
        return RtcOfferAnswer(
            session_id=request.session_id,
            answer_type="answer",
            sdp="answer-sdp",
        )

    monkeypatch.setattr(rtc_runtime_module, "exchange_server_rtc_offer", fake_exchange)

    with client.websocket_connect("/v1/socket") as ws:
        _receive_json(ws)
        channel_name = f"/rtc/{rtc_session['session_id']}"
        _send_channel_message(
            ws,
            action="JOIN_CHANNEL",
            channel_name=channel_name,
            event="join",
            payload={},
            request_id="join-rtc",
        )
        assert _receive_json(ws)["event"] == "ACKNOWLEDGE"
        assert _receive_json(ws)["event"] == "rtc.session.attached"

        _send_channel_message(
            ws,
            action="BROADCAST",
            channel_name=channel_name,
            event="rtc.offer",
            payload={"offer": {"type": "offer", "sdp": "offer-sdp"}},
            request_id="rtc-offer",
        )

        answer = _receive_json(ws)
        assert answer["event"] == "rtc.answer"
        assert answer["channelName"] == channel_name
        assert answer["payload"]["answer"] == {
            "type": "answer",
            "sdp": "answer-sdp",
        }


def test_pondsocket_rejects_grpc_owned_session_without_consuming_it():
    app = _build_app()
    record = app.state.rtc_registry.create_session(control_transport="grpc")
    client = TestClient(app)

    with client.websocket_connect("/v1/socket") as ws:
        _receive_json(ws)
        _send_channel_message(
            ws,
            action="JOIN_CHANNEL",
            channel_name=f"/rtc/{record.session_id}",
            event="join",
            payload={},
            request_id="join-mismatch",
        )
        rejected = _receive_json(ws)

    assert rejected == {
        "action": "SYSTEM",
        "event": "UNAUTHORIZED",
        "channelName": f"/rtc/{record.session_id}",
        "requestId": "join-mismatch",
        "payload": {
            "code": 409,
            "message": (f"RTC session '{record.session_id}' requires grpc control; received pondsocket"),
        },
    }
    assert app.state.rtc_registry.get(record.session_id) is record
    assert record.control_attached is False


def test_legacy_ws_routes_are_not_mounted():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/v1/conversation"):
        pass

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"/v1/rtc/sessions/{session['session_id']}/control"),
    ):
        pass
