from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from aioice.ice import Candidate
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

import vox.server.rtc_signaling as rtc_signaling_module
from vox.server.rtc_registry import RtcSessionRegistry
from vox.server.rtc_signaling import (
    add_browser_rtc_candidate,
    bind_peer_connection_handlers,
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


def test_attachment_without_peer_is_not_active():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    attachment = rtc_signaling_module._PeerAttachment(
        peer=None,
        audio_output_track=None,
        input_audio_track=None,
        data_channel=None,
        browser_attached=False,
        generation=None,
    )

    assert rtc_signaling_module._attachment_is_active(record, attachment) is False


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

    assert record.input_audio_track is not None
    assert "a=candidate:" not in result.sdp
    events = []
    while not record.media_events.empty():
        events.append(record.media_events.get_nowait())
    assert any(event.get("type") == "rtc.ice_candidate" and event.get("candidate") is not None for event in events)
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


@pytest.mark.asyncio
async def test_remote_description_events_belong_to_candidate_peer(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")

    class Track:
        kind = "audio"

        def __init__(self) -> None:
            self.stopped = False
            self.release = asyncio.Event()

        async def recv(self):
            await self.release.wait()
            raise MediaStreamError

        def stop(self) -> None:
            self.stopped = True

    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.handlers = {}
            self.closed = False

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        def close(self) -> None:
            self.closed = True

    class Peer:
        def __init__(self, **_kwargs) -> None:
            self.handlers = {}
            self.track = Track()
            self.channel = Channel()
            self.localDescription = None

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            self.handlers["track"](self.track)
            self.handlers["datachannel"](self.channel)

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, description) -> None:
            self.localDescription = description

        async def close(self) -> None:
            return None

    peer = Peer()
    monkeypatch.setattr(rtc_signaling_module, "TrickleRTCPeerConnection", lambda **_kwargs: peer)

    await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "offer-sdp"},
        generation=7,
    )
    await asyncio.sleep(0)

    assert record.rtc_peer is peer
    assert record.input_audio_track is peer.track
    assert record.data_channel is peer.channel
    assert peer.track.stopped is False
    assert peer.channel.closed is False
    await registry.close_all()


@pytest.mark.asyncio
async def test_failed_replacement_answer_preserves_active_peer(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")

    class OldPeer:
        pass

    class FailingPeer:
        def __init__(self, **_kwargs) -> None:
            self.closed = False

        def addTrack(self, _track) -> None:
            return None

        def on(self, _name):
            return lambda handler: handler

        async def setRemoteDescription(self, _description) -> None:
            raise RuntimeError("invalid replacement SDP")

        async def close(self) -> None:
            self.closed = True

    old_peer = OldPeer()
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 2
    monkeypatch.setattr(rtc_signaling_module, "TrickleRTCPeerConnection", FailingPeer)

    with pytest.raises(RuntimeError, match="invalid replacement SDP"):
        await create_browser_rtc_answer(
            registry=registry,
            record=record,
            offer={"type": "offer", "sdp": "broken"},
            restart=True,
            generation=3,
        )

    assert record.rtc_peer is old_peer
    assert record.browser_attached is True
    assert record.negotiation_generation == 2


@pytest.mark.asyncio
async def test_failed_offer_close_failure_remains_registry_owned(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")

    class OldPeer:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    class FailingPeer:
        def __init__(self, **_kwargs) -> None:
            self.close_calls = 0

        def addTrack(self, _track) -> None:
            return None

        def on(self, _name):
            return lambda handler: handler

        async def setRemoteDescription(self, _description) -> None:
            raise RuntimeError("invalid replacement SDP")

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("close failed")

    old_peer = OldPeer()
    failed_peer = FailingPeer()
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 2
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: failed_peer,
    )

    with pytest.raises(RuntimeError, match="invalid replacement SDP"):
        await create_browser_rtc_answer(
            registry=registry,
            record=record,
            offer={"type": "offer", "sdp": "broken"},
            restart=True,
            generation=3,
        )

    assert record.closed is True
    await registry.drain_teardowns()

    assert old_peer.close_calls == 1
    assert failed_peer.close_calls == 2
    assert record.retired_rtc_attachments == []


@pytest.mark.asyncio
async def test_local_description_close_failure_remains_registry_owned():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    class Peer:
        def __init__(self) -> None:
            self.close_calls = 0

        async def setLocalDescription(self, _description) -> None:
            raise RuntimeError("local description failed")

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("close failed")

    class ActivePeer:
        async def close(self) -> None:
            return None

    peer = Peer()
    attachment = rtc_signaling_module._PeerAttachment(
        peer=peer,
        audio_output_track=None,
        input_audio_track=None,
        data_channel=None,
        browser_attached=True,
        generation=2,
    )
    record.rtc_peer = ActivePeer()
    record.negotiation_generation = 1
    record.pending_rtc_attachment = attachment

    await rtc_signaling_module._apply_local_description(
        record,
        registry,
        attachment,
        RTCSessionDescription(type="answer", sdp="answer-sdp"),
    )

    assert record.closed is True
    await registry.drain_teardowns()

    assert peer.close_calls == 2
    assert record.retired_rtc_attachments == []


@pytest.mark.asyncio
async def test_successful_replacement_swaps_before_closing_old_peer(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, **_kwargs) -> None:
            self.handlers = {}
            self.closed = 0
            self.local_descriptions = []

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, description) -> None:
            self.local_descriptions.append(description)

        async def close(self) -> None:
            self.closed += 1

    old_peer = Peer()
    replacement = Peer()
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )

    await result.commit()

    assert result.sdp == "answer-sdp"
    assert record.rtc_peer is replacement
    assert record.negotiation_generation == 2
    assert record.browser_attached is True
    assert replacement.local_descriptions
    assert old_peer.closed == 1
    await registry.close_all()


@pytest.mark.asyncio
async def test_replacement_keeps_active_peer_authoritative_until_commit(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, **_kwargs) -> None:
            self.handlers = {}
            self.closed = 0

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            self.closed += 1

    old_peer = Peer()
    replacement = Peer()
    old_track = object()
    old_channel = object()
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.input_audio_track = old_track
    record.data_channel = old_channel
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )

    assert record.rtc_peer is old_peer
    assert record.input_audio_track is old_track
    assert record.data_channel is old_channel
    assert record.negotiation_generation == 1
    assert old_peer.closed == 0

    await result.commit()

    assert record.rtc_peer is replacement
    assert record.negotiation_generation == 2
    assert old_peer.closed == 1
    await registry.close_all()


@pytest.mark.asyncio
async def test_replacement_close_failure_terminates_owned_session(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    close_failed = asyncio.Event()

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, *, fail_first_close: bool = False) -> None:
            self.handlers = {}
            self.close_calls = 0
            self.fail_first_close = fail_first_close

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            self.close_calls += 1
            if self.fail_first_close and self.close_calls == 1:
                close_failed.set()
                raise RuntimeError("close failed")

    old_peer = Peer(fail_first_close=True)
    replacement = Peer()
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )
    await result.commit()

    assert record.rtc_peer is replacement
    assert record.retired_rtc_attachments == [result.previous]

    await close_failed.wait()
    await asyncio.sleep(0)
    assert record.closed is True
    await registry.drain_teardowns()

    assert old_peer.close_calls == 2
    assert replacement.close_calls == 1
    assert record.retired_rtc_attachments == []


@pytest.mark.asyncio
async def test_restart_rejects_while_previous_peer_is_retiring(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_done = asyncio.Event()

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, *, blocking_close: bool = False) -> None:
            self.handlers = {}
            self.blocking_close = blocking_close

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            if self.blocking_close:
                close_started.set()
                await close_release.wait()
                close_done.set()

    old_peer = Peer(blocking_close=True)
    replacement = Peer()
    created = 0

    def create_peer(**_kwargs):
        nonlocal created
        created += 1
        return replacement

    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(rtc_signaling_module, "TrickleRTCPeerConnection", create_peer)

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )
    await result.commit()
    await close_started.wait()

    with pytest.raises(RuntimeError, match="previous RTC peer is still closing"):
        await create_browser_rtc_answer(
            registry=registry,
            record=record,
            offer={"type": "offer", "sdp": "next"},
            restart=True,
            generation=3,
        )

    assert created == 1
    assert len(record.retired_rtc_attachments) == 1

    close_release.set()
    await close_done.wait()
    await asyncio.sleep(0)

    assert record.retired_rtc_attachments == []
    await registry.close_all()


@pytest.mark.asyncio
async def test_rollback_close_failure_terminates_owned_session(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    close_failed = asyncio.Event()

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, *, fail_first_close: bool = False) -> None:
            self.handlers = {}
            self.close_calls = 0
            self.fail_first_close = fail_first_close

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            self.close_calls += 1
            if self.fail_first_close and self.close_calls == 1:
                close_failed.set()
                raise RuntimeError("close failed")

    old_peer = Peer()
    replacement = Peer(fail_first_close=True)
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )
    await result.rollback()

    assert record.rtc_peer is old_peer
    assert record.retired_rtc_attachments == [result.attachment]

    await close_failed.wait()
    await asyncio.sleep(0)
    assert record.closed is True
    await registry.drain_teardowns()

    assert old_peer.close_calls == 1
    assert replacement.close_calls == 2
    assert record.retired_rtc_attachments == []


@pytest.mark.asyncio
async def test_cancelled_replacement_commit_does_not_restore_closed_peer(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    close_started = asyncio.Event()

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, *, blocking_close: bool = False, **_kwargs) -> None:
            self.handlers = {}
            self.closed = False
            self.blocking_close = blocking_close

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            self.closed = True
            if self.blocking_close:
                self.blocking_close = False
                close_started.set()
                await asyncio.Event().wait()

    old_peer = Peer(blocking_close=True)
    replacement = Peer()
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )
    commit_task = asyncio.create_task(result.commit())
    await close_started.wait()
    commit_task.cancel()
    await commit_task
    await result.rollback()

    assert record.rtc_peer is replacement
    assert record.negotiation_generation == 2
    assert old_peer.closed is True
    assert replacement.closed is False
    await registry.close_all()


@pytest.mark.asyncio
async def test_session_close_during_replacement_closes_both_owned_peers(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, **_kwargs) -> None:
            self.handlers = {}
            self.closed = 0

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            self.closed += 1

    old_peer = Peer()
    replacement = Peer()
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )

    await registry.close_all()
    await result.rollback()

    assert record.closed is True
    assert old_peer.closed == 1
    assert replacement.closed == 1


@pytest.mark.asyncio
async def test_replacement_preserves_output_track_pending_audio(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, **_kwargs) -> None:
            self.handlers = {}
            self.closed = 0

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            self.closed += 1

    old_peer = Peer()
    replacement = Peer()
    queue = rtc_signaling_module.create_rtc_audio_queue()
    output_track = rtc_signaling_module.RtcAudioOutputTrack(queue)
    await output_track.enqueue(b"\x01\x00" * 1_600, 16_000)
    await output_track.recv()
    assert output_track.stats()["pending_samples"] == 1_280

    record.audio_output = queue
    record.audio_output_track = output_track
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )

    assert record.audio_output_track is output_track
    assert output_track.stats()["pending_samples"] == 1_280
    commit = getattr(result, "commit", None)
    if commit is not None:
        await commit()
    await registry.close_all()


@pytest.mark.asyncio
async def test_replacement_sender_cannot_consume_audio_before_handoff(
    monkeypatch,
):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self) -> None:
            self.handlers = {}
            self.track = None
            self.closed = 0

        def addTrack(self, track) -> None:
            self.track = track

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            self.closed += 1

    old_peer = Peer()
    replacement = Peer()
    queue = rtc_signaling_module.create_rtc_audio_queue()
    output_track = rtc_signaling_module.RtcAudioOutputTrack(queue)
    record.audio_output = queue
    record.audio_output_track = output_track
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )
    await output_track.enqueue(b"\x01\x00" * 1_600, 16_000)
    replacement_recv = asyncio.create_task(replacement.track.recv())
    await asyncio.sleep(0)

    assert replacement_recv.done() is False

    await result.commit()
    frame = await asyncio.wait_for(replacement_recv, timeout=1)
    assert frame.pts == 0
    assert old_peer.closed == 1
    await registry.close_all()


@pytest.mark.asyncio
async def test_cancelled_answer_publication_rolls_back_pending_attachment(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    offer_task: asyncio.Task | None = None

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self) -> None:
            self.handlers = {}
            self.closed = 0

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            assert offer_task is not None
            offer_task.cancel()

        async def close(self) -> None:
            self.closed += 1

    old_peer = Peer()
    replacement = Peer()
    record.audio_output = rtc_signaling_module.create_rtc_audio_queue()
    record.audio_output_track = rtc_signaling_module.RtcAudioOutputTrack(record.audio_output)
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: replacement,
    )

    offer_task = asyncio.create_task(
        create_browser_rtc_answer(
            registry=registry,
            record=record,
            offer={"type": "offer", "sdp": "replacement"},
            restart=True,
            generation=2,
        )
    )

    try:
        with pytest.raises(asyncio.CancelledError):
            await offer_task
        await asyncio.sleep(0)

        assert record.pending_rtc_attachment is None
        assert record.rtc_peer is old_peer
        assert record.negotiation_generation == 1
        assert replacement.closed == 1
    finally:
        await registry.close_all()


@pytest.mark.asyncio
async def test_stale_replacement_rollback_preserves_new_peer_playout_observer(monkeypatch):
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    observed: list[bytes] = []

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self, **_kwargs) -> None:
            self.handlers = {}

        def addTrack(self, _track) -> None:
            return None

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def setRemoteDescription(self, _description) -> None:
            return None

        async def createAnswer(self):
            return RTCSessionDescription(type="answer", sdp="answer-sdp")

        async def setLocalDescription(self, _description) -> None:
            return None

        async def close(self) -> None:
            return None

    old_peer = Peer()
    candidate_peer = Peer()
    newer_peer = Peer()
    queue = rtc_signaling_module.create_rtc_audio_queue()
    output_track = rtc_signaling_module.RtcAudioOutputTrack(queue)
    record.audio_output = queue
    record.audio_output_track = output_track
    record.browser_attached = True
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    monkeypatch.setattr(
        rtc_signaling_module,
        "TrickleRTCPeerConnection",
        lambda **_kwargs: candidate_peer,
    )

    result = await create_browser_rtc_answer(
        registry=registry,
        record=record,
        offer={"type": "offer", "sdp": "replacement"},
        restart=True,
        generation=2,
    )
    record.rtc_peer = newer_peer
    record.negotiation_generation = 3
    output_track.set_playout_observer(lambda pcm16, _sample_rate: observed.append(pcm16))

    await result.rollback()
    await output_track.enqueue(b"\x01\x00" * 320, 16_000)
    await output_track.recv()

    assert observed
    assert record.rtc_peer is newer_peer
    assert record.negotiation_generation == 3
    await registry.close_all()


@pytest.mark.asyncio
async def test_stale_local_description_task_cannot_mutate_replacement():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")

    class Peer:
        calls = 0

        async def setLocalDescription(self, _description) -> None:
            self.calls += 1

        async def close(self) -> None:
            return None

    old_peer = Peer()
    replacement = Peer()
    record.browser_attached = True
    record.rtc_peer = replacement
    record.negotiation_generation = 2

    await rtc_signaling_module._apply_local_description(
        record,
        registry,
        rtc_signaling_module._PeerAttachment(
            peer=old_peer,
            audio_output_track=None,
            input_audio_track=None,
            data_channel=None,
            browser_attached=True,
            generation=1,
        ),
        RTCSessionDescription(type="answer", sdp="stale"),
    )

    assert old_peer.calls == 0
    assert record.rtc_peer is replacement
    assert registry.get(record.session_id) is record
    assert record.media_events.empty()
    await registry.close_all()


@pytest.mark.asyncio
async def test_stale_peer_and_data_channel_callbacks_cannot_close_replacement():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="pondsocket")
    record.browser_attached = True

    class Peer:
        def __init__(self) -> None:
            self.handlers = {}
            self.connectionState = "connected"
            self.iceConnectionState = "connected"
            self.iceGatheringState = "complete"

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def close(self) -> None:
            return None

    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.handlers = {}

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        def close(self) -> None:
            return None

    old_peer = Peer()
    record.rtc_peer = old_peer
    record.negotiation_generation = 1
    bind_peer_connection_handlers(
        record=record,
        session_id=record.session_id,
        registry=registry,
        peer=old_peer,
        generation=1,
    )
    old_channel = Channel()
    old_peer.handlers["datachannel"](old_channel)

    replacement_peer = Peer()
    replacement_channel = Channel()
    record.rtc_peer = replacement_peer
    record.data_channel = replacement_channel
    record.negotiation_generation = 2
    old_peer.connectionState = "closed"

    await old_peer.handlers["connectionstatechange"]()
    old_channel.handlers["close"]()
    await asyncio.sleep(0)

    assert registry.get(record.session_id) is record
    assert record.rtc_peer is replacement_peer
    assert record.data_channel is replacement_channel
    assert record.control_events.empty()
    assert record.media_events.empty()
    await registry.close_all()


@pytest.mark.asyncio
async def test_only_one_audio_track_can_own_input_pipeline():
    registry = RtcSessionRegistry()
    record = registry.create_session(control_transport="grpc")
    record.browser_attached = True

    class Peer:
        connectionState = "connected"
        iceConnectionState = "connected"
        iceGatheringState = "complete"

        def __init__(self) -> None:
            self.handlers = {}

        def on(self, name):
            def register(handler):
                self.handlers[name] = handler
                return handler

            return register

        async def close(self) -> None:
            return None

    class Track:
        kind = "audio"

        def __init__(self) -> None:
            self.stopped = False

        async def recv(self):
            raise MediaStreamError

        def stop(self) -> None:
            self.stopped = True

    peer = Peer()
    record.rtc_peer = peer
    record.negotiation_generation = 4
    bind_peer_connection_handlers(
        record=record,
        session_id=record.session_id,
        registry=registry,
        peer=peer,
        generation=4,
    )
    first = Track()
    second = Track()

    peer.handlers["track"](first)
    peer.handlers["track"](second)
    await asyncio.sleep(0)

    assert first.stopped is False
    assert second.stopped is True
    await registry.close_all()
