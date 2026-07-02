from __future__ import annotations

import asyncio

import numpy as np
from aiortc import RTCPeerConnection
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.fakes import FakeScheduler
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.server.routes.rtc import router as rtc_router


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


def test_create_rtc_session_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("VOX_API_KEY", "secret")
    client = TestClient(_build_app())

    unauthorized = client.post("/v1/rtc/sessions")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["detail"] == "missing or invalid API key"

    authorized = client.post(
        "/v1/rtc/sessions",
        headers={"authorization": "Bearer secret"},
    )
    assert authorized.status_code == 200
    assert authorized.json()["session_id"].startswith("rtc_")


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


def test_rtc_offer_accepts_client_token_from_bearer_header():
    client = TestClient(_build_app())
    session = client.post("/v1/rtc/sessions").json()
    offer = asyncio.run(_make_offer())

    response = client.post(
        f"/v1/rtc/sessions/{session['session_id']}/offer",
        headers={"authorization": f"Bearer {session['client_token']}"},
        json=offer,
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == session["session_id"]


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


def test_rtc_candidate_endpoint_accepts_media_token_from_bearer_header():
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
        headers={"authorization": f"Bearer {answer['media_token']}"},
        json={"candidate": None},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
