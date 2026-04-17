from __future__ import annotations

import logging

from vox.conversation.types import (
    TimerKey,
    TurnAction,
    TurnActionType,
    TurnEvent,
    TurnEventType,
    TurnPolicy,
    TurnState,
    act,
    cancel_timer,
    start_timer,
)

logger = logging.getLogger(__name__)


class TurnStateMachine:
    """Pure turn-taking state machine. No IO, no asyncio.

    Drive it by pushing `TurnEvent`s via `handle()`; it returns a list of
    `TurnAction`s the caller should execute in order. The machine's current
    state is exposed via `state`.

    Concurrency: not thread-safe. Callers are expected to serialize events
    (e.g. via a single asyncio event loop draining an event queue).
    """

    def __init__(self, policy: TurnPolicy | None = None) -> None:
        self._policy = policy or TurnPolicy()
        self._state = TurnState.IDLE

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def policy(self) -> TurnPolicy:
        return self._policy

    def reset(self) -> None:
        self._state = TurnState.IDLE

    def handle(self, event: TurnEvent) -> list[TurnAction]:
        """Apply a single event. Returns the actions to execute.

        Unknown (state, event) pairs are ignored silently — the caller may emit
        stale or redundant events, and the machine should be resilient.
        """
        handler = _TRANSITIONS.get((self._state, event.type))
        if handler is None:
            return []

        new_state, actions = handler(self, event)
        if new_state is not None and new_state != self._state:
            logger.debug("turn state %s -> %s via %s", self._state.value, new_state.value, event.type.value)
            self._state = new_state
        return actions










def _on_speech_started_idle(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.LISTENING, []


def _on_speech_started_listening(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:

    return None, [cancel_timer(TimerKey.ENDPOINTING)]


def _on_speech_started_thinking(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:


    if not m.policy.allow_interrupt_while_speaking:
        return None, []
    return TurnState.LISTENING, [
        act(TurnActionType.STOP_TTS),
        act(TurnActionType.CANCEL_RESPONSE),
    ]


def _on_speech_started_speaking(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    if not m.policy.allow_interrupt_while_speaking:
        return None, []


    confirm_ms = int(e.payload.get("confirm_window_ms", m.policy.min_interrupt_duration_ms))
    return TurnState.PAUSED, [
        act(TurnActionType.PAUSE_OUTPUT),
        start_timer(TimerKey.CONFIRM_INTERRUPT, confirm_ms),
    ]


def _on_speech_started_paused(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:

    return None, []


def _on_speech_started_interrupted(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:

    return None, []


def _on_speech_stopped_listening(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:


    return None, [start_timer(TimerKey.ENDPOINTING, m.policy.max_endpointing_delay_ms)]


def _on_speech_stopped_paused(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:


    return TurnState.SPEAKING, [
        cancel_timer(TimerKey.CONFIRM_INTERRUPT),
        act(TurnActionType.RESUME_OUTPUT),
    ]


def _on_speech_stopped_interrupted(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:


    return TurnState.LISTENING, [start_timer(TimerKey.ENDPOINTING, m.policy.max_endpointing_delay_ms)]


def _on_user_transcript_final_listening(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.THINKING, [cancel_timer(TimerKey.ENDPOINTING)]


def _on_user_transcript_final_idle(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:

    return TurnState.THINKING, []


def _on_user_transcript_final_interrupted(
    m: TurnStateMachine,
    e: TurnEvent,
) -> tuple[TurnState | None, list[TurnAction]]:

    return TurnState.THINKING, [cancel_timer(TimerKey.ENDPOINTING)]


def _on_response_started(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:


    return None, []


def _on_tts_audio_started_from_quiet(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:


    return TurnState.SPEAKING, []


def _on_tts_completed_speaking(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, []


def _on_tts_completed_thinking(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:


    return TurnState.IDLE, []


def _on_tts_completed_paused(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:


    return TurnState.IDLE, [cancel_timer(TimerKey.CONFIRM_INTERRUPT)]


def _on_tts_failed_thinking(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, []


def _on_tts_failed_speaking(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, []


def _on_tts_failed_paused(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, [
        cancel_timer(TimerKey.CONFIRM_INTERRUPT),
        act(TurnActionType.FLUSH_OUTPUT),
    ]


def _on_tts_failed_interrupted(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, [act(TurnActionType.FLUSH_OUTPUT)]


def _on_timer_endpointing(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:

    if e.payload.get("key") != TimerKey.ENDPOINTING.value:
        return None, []
    return TurnState.THINKING, []


def _on_timer_confirm_interrupt(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    if e.payload.get("key") != TimerKey.CONFIRM_INTERRUPT.value:
        return None, []
    return TurnState.INTERRUPTED, [
        act(TurnActionType.STOP_TTS),
        act(TurnActionType.CANCEL_RESPONSE),
        act(TurnActionType.FLUSH_OUTPUT),
    ]


def _on_timer_elapsed_listening(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return _on_timer_endpointing(m, e)


def _on_timer_elapsed_paused(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return _on_timer_confirm_interrupt(m, e)





def _on_client_cancel_idle(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return None, []


def _on_client_cancel_listening(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, [cancel_timer(TimerKey.ENDPOINTING)]


def _on_client_cancel_thinking(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, [
        act(TurnActionType.STOP_TTS),
        act(TurnActionType.CANCEL_RESPONSE),
    ]


def _on_client_cancel_speaking(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, [
        act(TurnActionType.STOP_TTS),
        act(TurnActionType.CANCEL_RESPONSE),
        act(TurnActionType.FLUSH_OUTPUT),
    ]


def _on_client_cancel_paused(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:
    return TurnState.IDLE, [
        cancel_timer(TimerKey.CONFIRM_INTERRUPT),
        act(TurnActionType.STOP_TTS),
        act(TurnActionType.CANCEL_RESPONSE),
        act(TurnActionType.FLUSH_OUTPUT),
    ]


def _on_client_cancel_interrupted(m: TurnStateMachine, e: TurnEvent) -> tuple[TurnState | None, list[TurnAction]]:

    return TurnState.IDLE, []






_TRANSITIONS: dict[tuple[TurnState, TurnEventType], callable] = {

    (TurnState.IDLE, TurnEventType.SPEECH_STARTED): _on_speech_started_idle,
    (TurnState.LISTENING, TurnEventType.SPEECH_STARTED): _on_speech_started_listening,
    (TurnState.THINKING, TurnEventType.SPEECH_STARTED): _on_speech_started_thinking,
    (TurnState.SPEAKING, TurnEventType.SPEECH_STARTED): _on_speech_started_speaking,
    (TurnState.PAUSED, TurnEventType.SPEECH_STARTED): _on_speech_started_paused,
    (TurnState.INTERRUPTED, TurnEventType.SPEECH_STARTED): _on_speech_started_interrupted,


    (TurnState.LISTENING, TurnEventType.SPEECH_STOPPED): _on_speech_stopped_listening,
    (TurnState.PAUSED, TurnEventType.SPEECH_STOPPED): _on_speech_stopped_paused,
    (TurnState.INTERRUPTED, TurnEventType.SPEECH_STOPPED): _on_speech_stopped_interrupted,


    (TurnState.IDLE, TurnEventType.USER_TRANSCRIPT_FINAL): _on_user_transcript_final_idle,
    (TurnState.LISTENING, TurnEventType.USER_TRANSCRIPT_FINAL): _on_user_transcript_final_listening,
    (TurnState.INTERRUPTED, TurnEventType.USER_TRANSCRIPT_FINAL): _on_user_transcript_final_interrupted,


    (TurnState.THINKING, TurnEventType.RESPONSE_STARTED): _on_response_started,


    (TurnState.IDLE, TurnEventType.TTS_AUDIO_STARTED): _on_tts_audio_started_from_quiet,
    (TurnState.THINKING, TurnEventType.TTS_AUDIO_STARTED): _on_tts_audio_started_from_quiet,


    (TurnState.THINKING, TurnEventType.TTS_COMPLETED): _on_tts_completed_thinking,
    (TurnState.SPEAKING, TurnEventType.TTS_COMPLETED): _on_tts_completed_speaking,
    (TurnState.PAUSED, TurnEventType.TTS_COMPLETED): _on_tts_completed_paused,


    (TurnState.THINKING, TurnEventType.TTS_FAILED): _on_tts_failed_thinking,
    (TurnState.SPEAKING, TurnEventType.TTS_FAILED): _on_tts_failed_speaking,
    (TurnState.PAUSED, TurnEventType.TTS_FAILED): _on_tts_failed_paused,
    (TurnState.INTERRUPTED, TurnEventType.TTS_FAILED): _on_tts_failed_interrupted,


    (TurnState.LISTENING, TurnEventType.TIMER_ELAPSED): _on_timer_elapsed_listening,
    (TurnState.PAUSED, TurnEventType.TIMER_ELAPSED): _on_timer_elapsed_paused,


    (TurnState.IDLE, TurnEventType.CLIENT_CANCEL): _on_client_cancel_idle,
    (TurnState.LISTENING, TurnEventType.CLIENT_CANCEL): _on_client_cancel_listening,
    (TurnState.THINKING, TurnEventType.CLIENT_CANCEL): _on_client_cancel_thinking,
    (TurnState.SPEAKING, TurnEventType.CLIENT_CANCEL): _on_client_cancel_speaking,
    (TurnState.PAUSED, TurnEventType.CLIENT_CANCEL): _on_client_cancel_paused,
    (TurnState.INTERRUPTED, TurnEventType.CLIENT_CANCEL): _on_client_cancel_interrupted,
}
