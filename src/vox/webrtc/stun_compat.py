"""Vox-owned STUN/TURN compatibility boundary for pinned aioice.

Some coturn deployments return non-UTF-8 bytes in the optional STUN
``ERROR-CODE`` reason. aioice 0.10.2 rejects the complete response while
decoding that text. These protocol classes use a local, tolerant parser
without mutating aioice's process-global attribute registry.
"""

from __future__ import annotations

import asyncio
import socket
import ssl as ssl_module
import struct
from collections.abc import Callable
from typing import Any, TypeVar, cast

from aioice import stun, turn
from aioice.ice import (
    Candidate,
    StunProtocol,
    candidate_foundation,
    candidate_priority,
)

ProtocolT = TypeVar("ProtocolT", bound=asyncio.BaseProtocol)


def parse_stun_message(data: bytes, integrity_key: bytes | None = None) -> stun.Message:
    """Parse STUN while decoding malformed error reasons defensively."""
    if len(data) < stun.HEADER_LENGTH:
        raise ValueError("STUN message length is less than 20 bytes")
    message_type, length, _cookie, transaction_id = struct.unpack(
        "!HHI12s",
        data[: stun.HEADER_LENGTH],
    )
    if len(data) != stun.HEADER_LENGTH + length:
        raise ValueError("STUN message length does not match")

    attributes: dict[str, Any] = {}
    pos = stun.HEADER_LENGTH
    while pos <= len(data) - 4:
        attr_type, attr_len = struct.unpack("!HH", data[pos : pos + 4])
        value = data[pos + 4 : pos + 4 + attr_len]
        if attr_type in stun.ATTRIBUTES_BY_TYPE:
            _, attr_name, _attr_pack, attr_unpack = stun.ATTRIBUTES_BY_TYPE[attr_type]
            if attr_name == "ERROR-CODE":
                attributes[attr_name] = unpack_error_code(value)
            elif attr_unpack == stun.unpack_xor_address:
                attributes[attr_name] = attr_unpack(value, transaction_id=transaction_id)
            else:
                attributes[attr_name] = attr_unpack(value)

            if attr_name == "FINGERPRINT":
                if attributes[attr_name] != stun.message_fingerprint(data[:pos]):
                    raise ValueError("STUN message fingerprint does not match")
            elif (
                attr_name == "MESSAGE-INTEGRITY"
                and integrity_key is not None
                and attributes[attr_name]
                != stun.message_integrity(
                    data[:pos],
                    integrity_key,
                )
            ):
                raise ValueError("STUN message integrity does not match")

        pos += 4 + attr_len + stun.padding_length(attr_len)

    return stun.Message(
        message_method=stun.Method(message_type & 0x3EEF),
        message_class=stun.Class(message_type & 0x0110),
        transaction_id=transaction_id,
        attributes=attributes,
    )


def unpack_error_code(data: bytes) -> tuple[int, str]:
    if len(data) < 4:
        raise ValueError("STUN error code is less than 4 bytes")
    _reserved, code_high, code_low = struct.unpack("!HBB", data[:4])
    return code_high * 100 + code_low, data[4:].decode("utf-8", errors="replace")


class CompatStunProtocol(StunProtocol):
    """aioice STUN protocol using Vox's local parser."""

    def datagram_received(self, data: bytes | str, addr: tuple) -> None:
        addr = (addr[0], addr[1])
        packet = cast(bytes, data)
        try:
            message = parse_stun_message(packet)
        except ValueError:
            self.receiver.data_received(packet, self.local_candidate.component)
            return

        if message.message_class in {stun.Class.RESPONSE, stun.Class.ERROR} and (
            message.transaction_id in self.transactions
        ):
            self.transactions[message.transaction_id].response_received(message, addr)
        elif message.message_class == stun.Class.REQUEST:
            self.receiver.request_received(message, addr, self, packet)


class _CompatTurnDatagramMixin:
    """TURN response dispatch using Vox's local parser."""

    def datagram_received(self, data: bytes | str, addr: tuple[str, int]) -> None:
        packet = cast(bytes, data)
        if len(packet) >= 4 and turn.is_channel_data(packet):
            channel, length = struct.unpack("!HH", packet[:4])
            if len(packet) >= length + 4 and self.receiver is not None:
                peer_address = self.channel_to_peer.get(channel)
                if peer_address:
                    self.receiver.datagram_received(packet[4 : 4 + length], peer_address)
            return

        try:
            message = parse_stun_message(packet)
        except ValueError:
            return

        if message.message_class in {stun.Class.RESPONSE, stun.Class.ERROR} and (
            message.transaction_id in self.transactions
        ):
            self.transactions[message.transaction_id].response_received(message, addr)


class CompatTurnClientUdpProtocol(_CompatTurnDatagramMixin, turn.TurnClientUdpProtocol):
    pass


class CompatTurnClientTcpProtocol(_CompatTurnDatagramMixin, turn.TurnClientTcpProtocol):
    pass


async def create_turn_endpoint(
    protocol_factory: Callable[[], ProtocolT],
    *,
    server_addr: tuple[str, int],
    username: str | None,
    password: str | None,
    lifetime: int = turn.DEFAULT_ALLOCATION_LIFETIME,
    channel_refresh_time: int = turn.DEFAULT_CHANNEL_REFRESH_TIME,
    ssl: bool | ssl_module.SSLContext | None = None,
    transport: str = "udp",
) -> tuple[turn.TurnTransport, ProtocolT]:
    """Create a TURN endpoint without aioice's global STUN parser."""
    loop = asyncio.get_running_loop()
    if transport == "tcp":
        inner_transport, inner_protocol = await loop.create_connection(
            lambda: CompatTurnClientTcpProtocol(
                server_addr,
                username=username,
                password=password,
                lifetime=lifetime,
                channel_refresh_time=channel_refresh_time,
            ),
            host=server_addr[0],
            port=server_addr[1],
            ssl=ssl,
        )
    else:
        inner_transport, inner_protocol = await loop.create_datagram_endpoint(
            lambda: CompatTurnClientUdpProtocol(
                server_addr,
                username=username,
                password=password,
                lifetime=lifetime,
                channel_refresh_time=channel_refresh_time,
            ),
            remote_addr=server_addr,
        )
        sock = inner_transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, turn.UDP_SOCKET_BUFFER_SIZE)

    try:
        protocol = protocol_factory()
        turn_transport = turn.TurnTransport(inner_protocol)
        await turn_transport._connect(protocol)
    except Exception:
        inner_transport.close()
        raise
    return turn_transport, protocol


async def relayed_candidate(
    *,
    component: int,
    protocol_factory: Callable[[], CompatStunProtocol],
    turn_server: tuple[str, int],
    turn_username: str | None,
    turn_password: str | None,
    turn_ssl: bool,
    turn_transport: str,
) -> tuple[Candidate, CompatStunProtocol]:
    """Acquire a relay candidate through Vox's compatible TURN protocol."""
    _, protocol = await create_turn_endpoint(
        protocol_factory,
        server_addr=turn_server,
        username=turn_username,
        password=turn_password,
        ssl=turn_ssl,
        transport=turn_transport,
    )
    candidate_address = protocol.transport.get_extra_info("sockname")
    related_address = protocol.transport.get_extra_info("related_address")
    protocol.local_candidate = Candidate(
        foundation=candidate_foundation("relay", "udp", candidate_address[0]),
        component=component,
        transport="udp",
        priority=candidate_priority(component, "relay"),
        host=candidate_address[0],
        port=candidate_address[1],
        type="relay",
        related_address=related_address[0],
        related_port=related_address[1],
    )
    return protocol.local_candidate, protocol
