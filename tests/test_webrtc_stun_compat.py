from __future__ import annotations

import pytest
from aioice import stun

from vox.webrtc.stun_compat import (
    CompatTurnClientUdpProtocol,
    parse_stun_message,
)


def _malformed_error_response() -> bytes:
    message = stun.Message(
        stun.Method.BINDING,
        stun.Class.ERROR,
        transaction_id=b"123456789012",
    )
    message.attributes["ERROR-CODE"] = (401, "xx")
    packet = bytearray(bytes(message))
    packet[-4:-2] = b"\xff\xff"
    return bytes(packet)


def test_local_parser_accepts_binary_error_reason_without_mutating_aioice():
    packet = _malformed_error_response()
    upstream_decoder = stun.ATTRIBUTES_BY_TYPE[0x0009][3]

    with pytest.raises(UnicodeDecodeError):
        stun.parse_message(packet)

    parsed = parse_stun_message(packet)

    assert parsed.attributes["ERROR-CODE"] == (401, "\ufffd\ufffd")
    assert stun.ATTRIBUTES_BY_TYPE[0x0009][3] is upstream_decoder


@pytest.mark.asyncio
async def test_compatible_turn_protocol_delivers_malformed_error_response():
    packet = _malformed_error_response()
    received = []

    class Transaction:
        def response_received(self, message, addr):
            received.append((message, addr))

    protocol = CompatTurnClientUdpProtocol(
        ("turn.example.test", 3478),
        username="user",
        password="password",
        lifetime=600,
        channel_refresh_time=500,
    )
    protocol.transactions[b"123456789012"] = Transaction()

    protocol.datagram_received(packet, ("192.0.2.10", 3478))

    assert len(received) == 1
    assert received[0][0].attributes["ERROR-CODE"] == (401, "\ufffd\ufffd")
    assert received[0][1] == ("192.0.2.10", 3478)
