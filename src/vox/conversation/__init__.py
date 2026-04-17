"""Voice-call orchestration: turn state machine + conversation session.

This package implements the barge-in / interruption logic that sits on top of
Vox's streaming primitives (VAD, EOU, STT, TTS). It's transport-agnostic: the
state machine is pure Python, and the session layer plugs into any async event
source.

See src/vox/conversation/state_machine.py for the core transition logic.
"""

from vox.conversation.interrupt import HeuristicInterruptClassifier, InterruptClassifier
from vox.conversation.state_machine import TurnStateMachine
from vox.conversation.types import (
    TimerKey,
    TurnAction,
    TurnActionType,
    TurnEvent,
    TurnEventType,
    TurnPolicy,
    TurnState,
)

__all__ = [
    "HeuristicInterruptClassifier",
    "InterruptClassifier",
    "TimerKey",
    "TurnAction",
    "TurnActionType",
    "TurnEvent",
    "TurnEventType",
    "TurnPolicy",
    "TurnState",
    "TurnStateMachine",
]
