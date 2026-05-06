from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TurnState(StrEnum):
    """Deterministic turn-taking states for a voice conversation.

    IDLE         -- no activity; waiting for either user or a programmatic response.
    LISTENING    -- VAD detected user voice; STT is producing transcripts.
    THINKING     -- user turn ended (EOU confirmed); waiting for agent reply.
    SPEAKING     -- TTS is actively streaming audio to the client.
    PAUSED       -- user started speaking mid-reply; waiting N ms to confirm it's
                    a real barge-in (not a cough / backchannel).
    INTERRUPTED  -- barge-in confirmed; TTS has been cancelled.
    """

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"


class TurnEventType(StrEnum):
    """Events the state machine consumes.

    Sources:
      - VAD:     SPEECH_STARTED, SPEECH_STOPPED
      - EOU/STT: USER_TRANSCRIPT_FINAL
      - TTS:     TTS_AUDIO_STARTED, TTS_COMPLETED
      - Client:  RESPONSE_STARTED (agent began generating), CLIENT_CANCEL
      - Timer:   TIMER_ELAPSED
    """

    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"
    USER_TRANSCRIPT_FINAL = "user_transcript_final"
    RESPONSE_STARTED = "response_started"
    TTS_AUDIO_STARTED = "tts_audio_started"
    TTS_COMPLETED = "tts_completed"
    TTS_FAILED = "tts_failed"
    CLIENT_CANCEL = "client_cancel"
    TIMER_ELAPSED = "timer_elapsed"


class TurnActionType(StrEnum):
    """Side-effects the state machine requests. The session layer executes them."""

    PAUSE_OUTPUT = "pause_output"
    RESUME_OUTPUT = "resume_output"
    FLUSH_OUTPUT = "flush_output"
    STOP_TTS = "stop_tts"
    CANCEL_RESPONSE = "cancel_response"
    START_TIMER = "start_timer"
    CANCEL_TIMER = "cancel_timer"


class TimerKey(StrEnum):
    """Identifiers for the state machine's timers."""

    ENDPOINTING = "endpointing"
    CONFIRM_INTERRUPT = "confirm_interrupt"


@dataclass
class TurnEvent:
    """A single input event for the state machine."""

    type: TurnEventType
    timestamp_ms: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnAction:
    """A side-effect the state machine asks the session layer to perform."""

    type: TurnActionType
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnPolicy:
    """Tunable thresholds for turn-taking decisions.

    Defaults are conservative and language-neutral. Callers should override via
    session config when they know their acoustic environment.
    """


    allow_interrupt_while_speaking: bool = True
    min_interrupt_duration_ms: int = 250
    max_endpointing_delay_ms: int = 3000
    stable_speaking_min_ms: int = 150







def ev(type_: TurnEventType, timestamp_ms: int = 0, **payload: Any) -> TurnEvent:
    return TurnEvent(type=type_, timestamp_ms=timestamp_ms, payload=payload)


def timer_event(key: TimerKey, timestamp_ms: int = 0) -> TurnEvent:
    return TurnEvent(type=TurnEventType.TIMER_ELAPSED, timestamp_ms=timestamp_ms, payload={"key": key.value})


def act(type_: TurnActionType, **payload: Any) -> TurnAction:
    return TurnAction(type=type_, payload=payload)


def start_timer(key: TimerKey, duration_ms: int) -> TurnAction:
    return act(TurnActionType.START_TIMER, key=key.value, duration_ms=duration_ms)


def cancel_timer(key: TimerKey) -> TurnAction:
    return act(TurnActionType.CANCEL_TIMER, key=key.value)
