"""gRPC mapping for conversation turn policy messages."""

from __future__ import annotations

from vox.conversation import TurnPolicy
from vox.grpc import vox_pb2
from vox.operations.conversation import TURN_POLICY_OVERRIDE_FIELDS


def conversation_turn_policy_overrides(
    policy: vox_pb2.ConversationTurnPolicy,
) -> dict[str, int | float | bool]:
    overrides: dict[str, int | float | bool] = {}
    for field_name in TURN_POLICY_OVERRIDE_FIELDS:
        if policy.HasField(field_name):
            overrides[field_name] = getattr(policy, field_name)
    return overrides


def conversation_turn_policy_pb(policy: TurnPolicy | None) -> vox_pb2.ConversationTurnPolicy:
    source = policy or TurnPolicy()
    pb = vox_pb2.ConversationTurnPolicy()
    for field_name in TURN_POLICY_OVERRIDE_FIELDS:
        setattr(pb, field_name, getattr(source, field_name))
    return pb
