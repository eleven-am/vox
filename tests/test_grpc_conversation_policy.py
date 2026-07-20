from __future__ import annotations

import pytest

from vox.conversation import TurnPolicy
from vox.grpc import vox_pb2
from vox.grpc.conversation_policy import (
    conversation_turn_policy_overrides,
    conversation_turn_policy_pb,
)
from vox.operations.conversation import TURN_POLICY_OVERRIDE_FIELDS


def test_conversation_turn_policy_overrides_preserve_only_explicit_fields():
    policy = vox_pb2.ConversationTurnPolicy(
        allow_interrupt_while_speaking=False,
        min_interrupt_duration_ms=101,
        self_echo_min_overlap=0.72,
    )

    assert conversation_turn_policy_overrides(policy) == {
        "allow_interrupt_while_speaking": False,
        "min_interrupt_duration_ms": 101,
        "self_echo_min_overlap": pytest.approx(0.72),
    }


def test_conversation_turn_policy_pb_populates_every_operation_owned_field():
    policy = TurnPolicy(
        allow_interrupt_while_speaking=False,
        min_interrupt_duration_ms=101,
        max_endpointing_delay_ms=102,
        false_interruption_timeout_ms=104,
        min_interrupt_words=105,
        partial_interrupts=True,
        dynamic_endpointing=False,
        min_endpointing_delay_ms=106,
        speaking_interrupt_min_duration_ms=107,
        speaking_interrupt_min_words=108,
        self_echo_min_words=109,
        self_echo_min_overlap=0.72,
        aec_warmup_ms=110,
        backchannel_end_cooldown_ms=111,
        vad_min_silence_ms=112,
    )

    pb = conversation_turn_policy_pb(policy)

    assert all(pb.HasField(field_name) for field_name in TURN_POLICY_OVERRIDE_FIELDS)
    assert conversation_turn_policy_overrides(pb) == {
        "allow_interrupt_while_speaking": False,
        "min_interrupt_duration_ms": 101,
        "max_endpointing_delay_ms": 102,
        "false_interruption_timeout_ms": 104,
        "min_interrupt_words": 105,
        "partial_interrupts": True,
        "dynamic_endpointing": False,
        "min_endpointing_delay_ms": 106,
        "speaking_interrupt_min_duration_ms": 107,
        "speaking_interrupt_min_words": 108,
        "self_echo_min_words": 109,
        "self_echo_min_overlap": pytest.approx(0.72),
        "aec_warmup_ms": 110,
        "backchannel_end_cooldown_ms": 111,
        "vad_min_silence_ms": 112,
    }
