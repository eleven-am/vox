from __future__ import annotations

import asyncio

import pytest
from aioice.ice import Candidate, TransportPolicy
from aiortc import RTCConfiguration, RTCSessionDescription

from vox.webrtc.trickle import TrickleConnection, TrickleRTCPeerConnection


class _FakeSocket:
    def setsockopt(self, *_args) -> None:
        pass


class _FakeTransport:
    def get_extra_info(self, name: str):
        if name == "socket":
            return _FakeSocket()
        if name == "sockname":
            return ("192.0.2.10", 50000)
        return None


class _FakeProtocol:
    def __init__(self) -> None:
        self.transport = _FakeTransport()
        self.local_candidate = None


@pytest.mark.asyncio
async def test_host_candidate_is_emitted_before_delayed_discovery_finishes(monkeypatch):
    release_discovery = asyncio.Event()
    emitted: list[Candidate] = []
    connection = TrickleConnection(
        ice_controlling=False,
        stun_server=("stun.example.test", 3478),
        on_candidate=emitted.append,
    )
    connection._transport_policy = TransportPolicy.ALL

    loop = asyncio.get_running_loop()

    async def create_endpoint(*_args, **_kwargs):
        return _FakeTransport(), _FakeProtocol()

    async def delayed_srflx(protocol, _server):
        await release_discovery.wait()
        return (
            Candidate(
                foundation="srflx",
                component=1,
                transport="udp",
                priority=1,
                host="198.51.100.10",
                port=51000,
                type="srflx",
                related_address=protocol.local_candidate.host,
                related_port=protocol.local_candidate.port,
            ),
            None,
        )

    monkeypatch.setattr(loop, "create_datagram_endpoint", create_endpoint)
    monkeypatch.setattr("vox.webrtc.trickle.server_reflexive_candidate", delayed_srflx)

    gathering = asyncio.create_task(connection.get_component_candidates(1, ["192.0.2.10"]))
    await asyncio.sleep(0)

    assert [candidate.type for candidate in emitted] == ["host"]
    assert gathering.done() is False

    release_discovery.set()
    await gathering
    assert [candidate.type for candidate in emitted] == ["host", "srflx"]


@pytest.mark.asyncio
async def test_candidate_protocol_is_registered_before_candidate_is_emitted(monkeypatch):
    protocol_counts: list[int] = []
    connection: TrickleConnection
    connection = TrickleConnection(
        ice_controlling=False,
        on_candidate=lambda _candidate: protocol_counts.append(len(connection._protocols)),
    )
    connection._transport_policy = TransportPolicy.ALL
    loop = asyncio.get_running_loop()

    async def create_endpoint(*_args, **_kwargs):
        return _FakeTransport(), _FakeProtocol()

    monkeypatch.setattr(loop, "create_datagram_endpoint", create_endpoint)

    await connection.get_component_candidates(1, ["192.0.2.10"])

    assert protocol_counts == [1]


@pytest.mark.asyncio
async def test_discovered_relay_protocol_is_registered_before_emission():
    protocol = _FakeProtocol()
    protocol_counts: list[int] = []
    connection: TrickleConnection
    connection = TrickleConnection(
        ice_controlling=False,
        on_candidate=lambda _candidate: protocol_counts.append(len(connection._protocols)),
    )

    async def discover():
        return (
            Candidate(
                foundation="relay",
                component=1,
                transport="udp",
                priority=1,
                host="198.51.100.10",
                port=52000,
                type="relay",
            ),
            protocol,
        )

    await connection._run_discovery(discover)

    assert protocol_counts == [1]


@pytest.mark.asyncio
async def test_answer_exists_while_local_gathering_is_in_progress(monkeypatch):
    gather_started = asyncio.Event()
    release_gather = asyncio.Event()

    async def delayed_gather(self):
        gather_started.set()
        await release_gather.wait()

    monkeypatch.setattr(TrickleConnection, "gather_candidates", delayed_gather)

    browser = TrickleRTCPeerConnection(RTCConfiguration(iceServers=[]))
    browser.createDataChannel("vox")
    offer = await browser.createOffer()
    browser_set_local = asyncio.create_task(browser.setLocalDescription(offer))
    await gather_started.wait()

    server = TrickleRTCPeerConnection(RTCConfiguration(iceServers=[]))
    await server.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type="offer"))
    answer = await server.createAnswer()
    server_set_local = asyncio.create_task(server.setLocalDescription(answer))
    await asyncio.sleep(0)

    assert answer.type == "answer"
    assert "a=ice-ufrag:" in answer.sdp
    assert "a=candidate:" not in answer.sdp
    assert server_set_local.done() is False

    release_gather.set()
    await asyncio.gather(browser_set_local, server_set_local)
    await asyncio.gather(browser.close(), server.close())
