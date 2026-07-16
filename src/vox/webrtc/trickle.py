"""Incremental local ICE gathering for aiortc.

aiortc 1.14 / aioice 0.10 expose half-trickle: remote candidates may be added
incrementally, but local candidates are only published after gathering ends.
Vox owns this small compatibility boundary so signaling transports can provide
real bidirectional trickle without polling or runtime monkeypatching.

Candidate discovery follows aioice's BSD-licensed gathering implementation,
with candidates recorded and emitted as each host, STUN, or TURN result becomes
available. The peer-connection override mirrors aiortc's DTLS transport factory
and is protected by exact dependency pins and regression tests.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable
from typing import Any

from aioice import turn
from aioice.ice import (
    Candidate,
    Connection,
    TransportPolicy,
    candidate_foundation,
    candidate_priority,
    get_host_addresses,
    server_reflexive_candidate,
)
from aiortc import RTCPeerConnection
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.rtcdtlstransport import RTCDtlsTransport
from aiortc.rtcicetransport import (
    RTCIceCandidate,
    RTCIceGatherer,
    RTCIceTransport,
    candidate_from_aioice,
    connection_kwargs,
)

from vox.webrtc.stun_compat import CompatStunProtocol, relayed_candidate

logger = logging.getLogger(__name__)

CandidateObserver = Callable[[Candidate], None]
DiscoveryFactory = Callable[[], Awaitable[tuple[Candidate, CompatStunProtocol | None]]]


class TrickleConnection(Connection):
    """An aioice connection that publishes candidates when discovered."""

    def __init__(self, *args: Any, on_candidate: CandidateObserver, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_candidate = on_candidate

    def _record_candidate(self, candidate: Candidate) -> None:
        self._local_candidates.append(candidate)
        self._on_candidate(candidate)

    async def _run_discovery(self, factory: DiscoveryFactory) -> None:
        candidate, protocol = await factory()
        if protocol is not None:
            self._protocols.append(protocol)
        self._record_candidate(candidate)

    async def get_component_candidates(
        self,
        component: int,
        addresses: list[str],
        timeout: int = 5,
    ) -> list[Candidate]:
        """Gather and emit one component's candidates in discovery order."""
        loop = asyncio.get_running_loop()
        host_protocols: list[CompatStunProtocol] = []
        discovered: list[Candidate] = []

        for address in addresses:
            try:
                transport, protocol = await loop.create_datagram_endpoint(
                    lambda: CompatStunProtocol(self),
                    local_addr=(address, 0),
                )
                sock = transport.get_extra_info("socket")
                if sock is not None:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, turn.UDP_SOCKET_BUFFER_SIZE)
            except OSError as exc:
                logger.info("Could not bind ICE candidate socket to %s: %s", address, exc)
                continue

            host_protocols.append(protocol)
            self._protocols.append(protocol)
            candidate_address = protocol.transport.get_extra_info("sockname")
            protocol.local_candidate = Candidate(
                foundation=candidate_foundation("host", "udp", candidate_address[0]),
                component=component,
                transport="udp",
                priority=candidate_priority(component, "host"),
                host=candidate_address[0],
                port=candidate_address[1],
                type="host",
            )
            if self._transport_policy == TransportPolicy.ALL:
                discovered.append(protocol.local_candidate)
                self._record_candidate(protocol.local_candidate)

        tasks: list[asyncio.Task[None]] = []

        if self.stun_server:
            for protocol in host_protocols:
                if ipaddress.ip_address(protocol.local_candidate.host).version == 4:
                    tasks.append(
                        asyncio.create_task(
                            self._run_discovery(
                                lambda protocol=protocol: server_reflexive_candidate(protocol, self.stun_server)
                            )
                        )
                    )

        if self.turn_server:
            tasks.append(
                asyncio.create_task(
                    self._run_discovery(
                        lambda: relayed_candidate(
                            component=component,
                            protocol_factory=lambda: CompatStunProtocol(self),
                            turn_server=self.turn_server,
                            turn_username=self.turn_username,
                            turn_password=self.turn_password,
                            turn_ssl=self.turn_ssl,
                            turn_transport=self.turn_transport,
                        )
                    )
                )
            )

        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            for task in done:
                try:
                    task.result()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("ICE candidate discovery failed: %s", exc)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return discovered

    async def gather_candidates(self) -> None:
        if self._local_candidates_start:
            return
        self._local_candidates_start = True
        addresses = get_host_addresses(use_ipv4=self._use_ipv4, use_ipv6=self._use_ipv6)
        await asyncio.gather(
            *(self.get_component_candidates(component=component, addresses=addresses) for component in self._components)
        )
        self._local_candidates_end = True


class TrickleIceGatherer(RTCIceGatherer):
    """RTCIceGatherer which emits ``localcandidate`` incrementally."""

    def __init__(
        self,
        iceServers: list[RTCIceServer] | None = None,
        local_username: str | None = None,
        local_password: str | None = None,
    ) -> None:
        super().__init__(
            iceServers=iceServers,
            local_username=local_username,
            local_password=local_password,
        )
        if iceServers is None:
            iceServers = self.getDefaultIceServers()
        ice_kwargs = connection_kwargs(iceServers)
        self._connection = TrickleConnection(
            ice_controlling=False,
            local_username=local_username,
            local_password=local_password,
            on_candidate=self._emit_local_candidate,
            **ice_kwargs,
        )

    def _emit_local_candidate(self, candidate: Candidate) -> None:
        self.emit("localcandidate", candidate_from_aioice(candidate))


class TrickleRTCPeerConnection(RTCPeerConnection):
    """aiortc peer connection with browser-compatible local ICE events."""

    def __init__(self, configuration: RTCConfiguration | None = None) -> None:
        super().__init__(configuration=configuration)
        self._end_of_candidates_emitted = False

        @self.on("icegatheringstatechange")
        def emit_end_of_candidates() -> None:
            if self.iceGatheringState == "completed" and not self._end_of_candidates_emitted:
                self._end_of_candidates_emitted = True
                self.emit("icecandidate", None)

    def _emit_local_candidate(self, gatherer: TrickleIceGatherer, candidate: RTCIceCandidate) -> None:
        for index, transceiver in enumerate(self.getTransceivers()):
            if transceiver.receiver.transport.transport.iceGatherer is gatherer:
                candidate.sdpMid = transceiver.mid
                candidate.sdpMLineIndex = index
                self.emit("icecandidate", candidate)
                return

        sctp = self.sctp
        if sctp is not None and sctp.transport.transport.iceGatherer is gatherer:
            candidate.sdpMid = sctp.mid
            candidate.sdpMLineIndex = len(self.getTransceivers())
            self.emit("icecandidate", candidate)

    def _RTCPeerConnection__createDtlsTransport(self) -> RTCDtlsTransport:
        """Create aiortc transports with Vox's incremental ICE gatherer."""
        transceivers = self._RTCPeerConnection__transceivers
        sctp = self._RTCPeerConnection__sctp
        if transceivers or sctp:
            if transceivers:
                parameters = transceivers[0].receiver.transport.transport.iceGatherer.getLocalParameters()
            else:
                parameters = sctp.transport.transport.iceGatherer.getLocalParameters()
            gatherer = TrickleIceGatherer(
                iceServers=self._RTCPeerConnection__configuration.iceServers,
                local_username=parameters.usernameFragment,
                local_password=parameters.password,
            )
        else:
            gatherer = TrickleIceGatherer(iceServers=self._RTCPeerConnection__configuration.iceServers)

        gatherer.on("localcandidate", lambda candidate: self._emit_local_candidate(gatherer, candidate))
        gatherer.on("statechange", self._RTCPeerConnection__updateIceGatheringState)
        ice_transport = RTCIceTransport(gatherer)
        ice_transport.on("statechange", self._RTCPeerConnection__updateIceConnectionState)
        ice_transport.on("statechange", self._RTCPeerConnection__updateConnectionState)
        self._RTCPeerConnection__iceTransports.add(ice_transport)

        dtls_transport = RTCDtlsTransport(ice_transport, self._RTCPeerConnection__certificates)
        dtls_transport.on("statechange", self._RTCPeerConnection__updateConnectionState)
        self._RTCPeerConnection__dtlsTransports.add(dtls_transport)

        self._RTCPeerConnection__updateIceGatheringState()
        self._RTCPeerConnection__updateIceConnectionState()
        self._RTCPeerConnection__updateConnectionState()
        return dtls_transport
