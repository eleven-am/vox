from __future__ import annotations

from types import SimpleNamespace

import pytest

from vox.server.rtc_signaling import (
    RtcSignalingError,
    create_browser_rtc_answer,
    ingest_media_audio,
    rtc_configuration,
)


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
async def test_create_browser_rtc_answer_rejects_invalid_client_token():
    class Registry:
        def attach_browser(self, _client_token: str):
            return None

    with pytest.raises(RtcSignalingError) as exc_info:
        await create_browser_rtc_answer(
            registry=Registry(),
            session_id="rtc_123",
            client_token="wrong",
            offer={},
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid or expired client token"


@pytest.mark.asyncio
async def test_create_browser_rtc_answer_closes_mismatched_session():
    class Registry:
        def __init__(self) -> None:
            self.closed = []

        def attach_browser(self, _client_token: str):
            return SimpleNamespace(session_id="rtc_other"), "rtc_media_123"

        def close(self, session_id: str) -> None:
            self.closed.append(session_id)

    registry = Registry()

    with pytest.raises(RtcSignalingError) as exc_info:
        await create_browser_rtc_answer(
            registry=registry,
            session_id="rtc_expected",
            client_token="rtc_client_123",
            offer={},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "RTC session not found"
    assert registry.closed == ["rtc_other"]
