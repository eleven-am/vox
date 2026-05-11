from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.fakes import FakeScheduler
from vox.operations.conversation import (
    ConvDoneEvent,
    ConversationSessionConfig,
    ConvEvent,
    ConvResponseCommittedEvent,
    ConvResponseCreatedEvent,
    ConvResponseDoneEvent,
    ConvSessionCreatedEvent,
)
from vox.server.livekit_config import LiveKitConfig, decode_unverified_livekit_token
from vox.server.routes import rtc as rtc_routes
from vox.server.routes.rtc import router as rtc_router


class FakeLiveKitConversation:
    def __init__(self, **kwargs) -> None:
        import asyncio

        self.config: ConversationSessionConfig | None = None
        self.events_queue: asyncio.Queue[ConvEvent] = asyncio.Queue()
        self.response_counter = 0
        self.pending_response_id = ""
        self.text = ""

    async def start_session(self, config: ConversationSessionConfig) -> None:
        self.config = config
        await self.events_queue.put(ConvSessionCreatedEvent(config=config))

    async def start_response(self) -> None:
        self.response_counter += 1
        self.pending_response_id = f"resp_{self.response_counter}"
        self.text = ""
        await self.events_queue.put(ConvResponseCreatedEvent(response_id=self.pending_response_id))

    async def append_response_text(self, text: str) -> None:
        if not self.pending_response_id:
            await self.start_response()
        self.text += text

    async def commit_response(self) -> None:
        response_id = self.pending_response_id or "resp_1"
        self.pending_response_id = ""
        await self.events_queue.put(ConvResponseCommittedEvent(response_id=response_id))
        await self.events_queue.put(ConvResponseDoneEvent(response_id=response_id))

    async def cancel_response(self) -> None:
        pass

    async def report_error(self, message: str) -> None:
        pass

    async def end_of_stream(self) -> None:
        await self.events_queue.put(ConvDoneEvent())

    async def close(self) -> None:
        pass

    async def events(self) -> AsyncIterator[ConvEvent]:
        while True:
            event = await self.events_queue.get()
            yield event
            if isinstance(event, ConvDoneEvent):
                return


def _build_app() -> FastAPI:
    app = FastAPI()
    app.state.scheduler = FakeScheduler()
    app.state.livekit_config = LiveKitConfig(
        url="ws://livekit.test",
        api_key="test-key",
        api_secret="test-secret",
        token_ttl_s=120,
    )
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


def test_create_rtc_session_returns_livekit_join_binding():
    client = TestClient(_build_app())

    response = client.post("/v1/rtc/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "livekit"
    assert payload["session_id"].startswith("rtc_")
    assert payload["room"].startswith("vox-")
    assert payload["livekit_url"] == "ws://livekit.test"
    assert payload["join_token_ttl_seconds"] == 120
    assert payload["control_url"] == f"/v1/rtc/sessions/{payload['session_id']}/control"
    claims = decode_unverified_livekit_token(payload["client_token"])
    assert claims["iss"] == "test-key"
    assert claims["sub"] == payload["participant_identity"]
    assert claims["video"]["room"] == payload["room"]
    assert claims["video"]["roomJoin"] is True
    assert claims["video"]["canPublish"] is True
    assert claims["video"]["canSubscribe"] is True


def test_create_rtc_session_requires_livekit_config(monkeypatch):
    for key in (
        "VOX_LIVEKIT_URL",
        "LIVEKIT_URL",
        "VOX_LIVEKIT_API_KEY",
        "LIVEKIT_API_KEY",
        "VOX_LIVEKIT_API_SECRET",
        "LIVEKIT_API_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    app = FastAPI()
    app.state.scheduler = FakeScheduler()
    app.include_router(rtc_router)
    client = TestClient(app)

    response = client.post("/v1/rtc/sessions")

    assert response.status_code == 503
    assert "LIVEKIT_URL" in response.json()["detail"]


def test_rtc_offer_endpoint_is_gone():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()

    response = client.post(
        f"/v1/rtc/sessions/{session['session_id']}/offer",
        json={"type": "offer", "sdp": "v=0"},
    )

    assert response.status_code == 410
    assert "LiveKit-backed" in response.json()["detail"]


def test_rtc_candidate_endpoint_is_gone():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()

    response = client.post(
        f"/v1/rtc/sessions/{session['session_id']}/candidates",
        json={"candidate": None},
    )

    assert response.status_code == 410
    assert "LiveKit handles ICE" in response.json()["detail"]


def test_rtc_control_session_update_emits_bound_session_created(monkeypatch):
    monkeypatch.setattr(rtc_routes, "LiveKitConversation", FakeLiveKitConversation)
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()

    with client.websocket_connect(f"/v1/rtc/sessions/{session['session_id']}/control") as ws:
        attached = ws.receive_json()
        assert attached == {
            "type": "rtc.session.attached",
            "session_id": session["session_id"],
            "provider": "livekit",
            "room": session["room"],
        }

        ws.send_json({
            "type": "session.update",
            "session": {
                "stt_model": "fake-stt:1",
                "tts_model": "fake-tts:1",
                "voice": "default",
            },
        })
        msg = ws.receive_json()

    assert msg["type"] == "session.created"
    assert msg["session_id"] == session["session_id"]
    assert msg["session"]["stt_model"] == "fake-stt:1"


def test_rtc_control_uses_livekit_media_instead_of_audio_delta(monkeypatch):
    monkeypatch.setattr(rtc_routes, "LiveKitConversation", FakeLiveKitConversation)
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


def test_rtc_control_rejects_audio_append_messages(monkeypatch):
    monkeypatch.setattr(rtc_routes, "LiveKitConversation", FakeLiveKitConversation)
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
