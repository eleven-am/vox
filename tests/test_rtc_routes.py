from __future__ import annotations

import asyncio
import json

import numpy as np
from aiortc import RTCPeerConnection
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.fakes import FakeScheduler
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.server.routes.rtc import router as rtc_router


class FakeDataChannel:
    def __init__(self) -> None:
        self.readyState = "open"
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)


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
    def is_loaded(self): return True
    def list_voices(self): return [VoiceInfo(id="default", name="Default")]

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
    app.include_router(rtc_router)
    return app


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


async def _make_offer() -> dict:
    peer = RTCPeerConnection()
    try:
        peer.addTransceiver("audio", direction="sendrecv")
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        return {
            "type": peer.localDescription.type,
            "sdp": peer.localDescription.sdp,
        }
    finally:
        await peer.close()


def test_create_rtc_session_returns_ephemeral_binding():
    client = TestClient(_build_app())

    response = client.post("/v1/rtc/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"].startswith("rtc_")
    assert payload["client_token"].startswith("rtc_client_")
    assert payload["join_token_ttl_seconds"] == 120
    assert payload["expires_at"]
    assert payload["ice_servers"] == []


def test_rtc_offer_returns_answer_media_token_and_events_url():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()
    offer = asyncio.run(_make_offer())

    response = client.post(
        f"/v1/rtc/sessions/{session['session_id']}/offer",
        json={
            **offer,
            "client_token": session["client_token"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session["session_id"]
    assert payload["media_token"].startswith("rtc_media_")
    assert payload["events_url"].startswith(f"/v1/rtc/sessions/{session['session_id']}/events?token=rtc_media_")
    assert payload["type"] == "answer"
    assert "m=audio" in payload["sdp"]


def test_rtc_offer_rejects_invalid_client_token():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()
    offer = asyncio.run(_make_offer())

    response = client.post(
        f"/v1/rtc/sessions/{session['session_id']}/offer",
        json={
            **offer,
            "client_token": "wrong",
        },
    )

    assert response.status_code == 401


def test_rtc_candidate_endpoint_accepts_end_of_candidates():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()
    offer = asyncio.run(_make_offer())

    answer = client.post(
        f"/v1/rtc/sessions/{session['session_id']}/offer",
        json={
            **offer,
            "client_token": session["client_token"],
        },
    ).json()
    response = client.post(
        f"/v1/rtc/sessions/{session['session_id']}/candidates",
        json={
            "media_token": answer["media_token"],
            "candidate": None,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_rtc_control_session_update_emits_bound_session_created():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()

    with client.websocket_connect(f"/v1/rtc/sessions/{session['session_id']}/control") as ws:
        attached = ws.receive_json()
        assert attached == {
            "type": "rtc.session.attached",
            "session_id": session["session_id"],
        }

        ws.send_json({
            "type": "session.update",
            "session": {
                "stt_model": "fake-stt:1",
                "tts_model": "fake-tts:1",
                "voice": "default",
                "turn_profile": "browser_default",
            },
        })
        msg = ws.receive_json()

    assert msg["type"] == "session.created"
    assert msg["session_id"] == session["session_id"]
    assert msg["session"]["stt_model"] == "fake-stt:1"
    assert msg["session"]["turn_profile"] == "browser_default"


def test_rtc_control_drops_audio_delta_events():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()

    with client.websocket_connect(f"/v1/rtc/sessions/{session['session_id']}/control") as ws:
        ws.receive_json()
        ws.send_json({
            "type": "session.update",
            "session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"},
        })
        ws.receive_json()
        ws.send_json({"type": "response.delta", "delta": "hello"})
        ws.send_json({"type": "response.commit"})

        events = _drain_until(ws, lambda e: e.get("type") == "response.done")

    assert events
    assert not any(e["type"] == "response.audio.delta" for e in events)
    assert any(e["type"] == "response.done" for e in events)
    assert all(e.get("session_id") == session["session_id"] for e in events)


def test_rtc_control_rejects_audio_append_messages():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()

    with client.websocket_connect(f"/v1/rtc/sessions/{session['session_id']}/control") as ws:
        ws.receive_json()
        ws.send_json({
            "type": "session.update",
            "session": {"stt_model": "x:1", "tts_model": "y:1"},
        })
        ws.receive_json()
        ws.send_json({"type": "input_audio_buffer.append", "audio": "AAAA"})
        msg = ws.receive_json()

    assert msg["type"] == "error"
    assert "unknown control message type" in msg["message"]


def test_rtc_control_relays_client_event_to_browser_data_channel():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()
    record = client.app.state.rtc_registry.get(session["session_id"])
    assert record is not None
    channel = FakeDataChannel()
    record.data_channel = channel

    with client.websocket_connect(f"/v1/rtc/sessions/{session['session_id']}/control") as ws:
        ws.receive_json()
        ws.send_json({
            "type": "client.event",
            "event": "render.url",
            "payload": {"url": "https://example.com"},
        })

    assert channel.sent
    assert json.loads(channel.sent[0]) == {
        "event": "render.url",
        "payload": {
            "url": "https://example.com",
        },
    }


def test_rtc_control_emits_browser_client_events_to_backend():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()
    record = client.app.state.rtc_registry.get(session["session_id"])
    assert record is not None
    assert record.control_events is not None

    with client.websocket_connect(f"/v1/rtc/sessions/{session['session_id']}/control") as ws:
        attached = ws.receive_json()
        assert attached["type"] == "rtc.session.attached"
        asyncio.run(record.control_events.put({
            "type": "client.event",
            "session_id": session["session_id"],
            "event": "render.image",
            "payload": {"image": "https://example.com/image.png"},
        }))
        msg = ws.receive_json()

    assert msg == {
        "type": "client.event",
        "session_id": session["session_id"],
        "event": "render.image",
        "payload": {"image": "https://example.com/image.png"},
    }
