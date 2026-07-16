from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from aioice.ice import Candidate
from aiortc import RTCConfiguration, RTCPeerConnection

from vox.server.rtc_registry import RtcSessionRegistry
from vox.server.rtc_signaling import (
    add_browser_rtc_candidate,
    create_browser_rtc_answer,
    ingest_media_audio,
    rtc_configuration,
)
from vox.webrtc.trickle import TrickleConnection


def test_rtc_configuration_maps_ice_server_dicts():
    config = rtc_configuration(
        [
            {"urls": ["turn:example.com:3478"], "username": "user", "credential": "pass"},
            {"urls": ["stun:example.com:3478"]},
        ]
    )

    assert len(config.iceServers) == 2
    assert config.iceServers[0].urls == ["turn:example.com:3478"]
    assert config.iceServers[0].username == "user"
    assert config.iceServers[0].credential == "pass"
    assert config.iceServers[1].urls == ["stun:example.com:3478"]
    assert config.iceServers[1].username is None
    assert config.iceServers[1].credential is None


@pytest.mark.asyncio
async def test_ingest_media_audio_waits_until_orchestrator_is_configured():
    record = SimpleNamespace(orchestrator=SimpleNamespace(config=None, calls=[]))

    await ingest_media_audio(record, b"abc", 16_000)

    assert record.orchestrator.calls == []


@pytest.mark.asyncio
async def test_ingest_media_audio_forwards_to_configured_orchestrator():
    class Orchestrator:
        config = object()

        def __init__(self) -> None:
            self.calls = []

        async def ingest_pcm16(self, pcm16: bytes, sample_rate: int | None = None) -> None:
            self.calls.append((pcm16, sample_rate))

    orchestrator = Orchestrator()
    record = SimpleNamespace(orchestrator=orchestrator)

    await ingest_media_audio(record, b"abc", 16_000)

    assert orchestrator.calls == [(b"abc", 16_000)]


@pytest.mark.asyncio
async def test_add_browser_rtc_candidate_applies_end_of_candidates():
    class Peer:
        def __init__(self) -> None:
            self.candidates = []

        async def addIceCandidate(self, candidate):
            self.candidates.append(candidate)

    peer = Peer()

    payload = await add_browser_rtc_candidate(
        record=SimpleNamespace(rtc_peer=peer),
        candidate={"candidate": None},
    )

    assert payload == {"ok": True}
    assert peer.candidates == [None]


@pytest.mark.asyncio
async def test_answer_returns_and_candidate_is_queued_before_gathering_finishes(monkeypatch):
    gathering_started = asyncio.Event()
    release_gathering = asyncio.Event()

    async def controlled_candidates(self, component, addresses, timeout=5):
        candidate = Candidate(
            foundation="host",
            component=component,
            transport="udp",
            priority=2130706431,
            host="192.0.2.10",
            port=50000,
            type="host",
        )
        self._record_candidate(candidate)
        gathering_started.set()
        await release_gathering.wait()
        return [candidate]

    browser = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    browser.addTransceiver("audio", direction="sendonly")
    browser.createDataChannel("vox")
    offer = await browser.createOffer()

    monkeypatch.setattr(TrickleConnection, "get_component_candidates", controlled_candidates)
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    assert registry.attach_browser_session(record.session_id) is record

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": offer.sdp},
    )
    await asyncio.wait_for(gathering_started.wait(), timeout=1)

    assert "a=candidate:" not in result["sdp"]
    assert any(
        event.get("type") == "rtc.ice_candidate" and event.get("candidate") is not None
        for event in list(record.media_events._queue)
    )
    assert any(not task.done() for task in record.media_tasks)

    release_gathering.set()
    local_description_tasks = [
        task for task in record.media_tasks if task.get_coro().__name__ == "_apply_local_description"
    ]
    await asyncio.wait_for(
        asyncio.gather(*local_description_tasks, return_exceptions=True),
        timeout=1,
    )
    await browser.close()
    await registry.close_all()
