"""Table-driven tests for TurnStateMachine.

The state machine is pure: no IO, no asyncio, no side effects beyond the returned
action list. These tests verify every (state, event) transition we care about.
"""

from __future__ import annotations

import pytest

from vox.conversation import (
    TimerKey,
    TurnAction,
    TurnActionType,
    TurnEvent,
    TurnEventType,
    TurnPolicy,
    TurnState,
    TurnStateMachine,
)
from vox.conversation.types import ev, timer_event







def _machine(state: TurnState, policy: TurnPolicy | None = None) -> TurnStateMachine:
    m = TurnStateMachine(policy=policy)
    m._state = state
    return m


def _action_types(actions: list[TurnAction]) -> list[TurnActionType]:
    return [a.type for a in actions]


def _action_with_payload(actions: list[TurnAction], type_: TurnActionType) -> TurnAction:
    matching = [a for a in actions if a.type == type_]
    assert len(matching) == 1, f"expected exactly one {type_.value} action, got {len(matching)}"
    return matching[0]







class TestHappyPath:
    def test_full_turn_idle_to_idle(self):
        m = TurnStateMachine()
        assert m.state == TurnState.IDLE

        m.handle(ev(TurnEventType.SPEECH_STARTED))
        assert m.state == TurnState.LISTENING

        m.handle(ev(TurnEventType.USER_TRANSCRIPT_FINAL))
        assert m.state == TurnState.THINKING

        m.handle(ev(TurnEventType.TTS_AUDIO_STARTED))
        assert m.state == TurnState.SPEAKING

        m.handle(ev(TurnEventType.TTS_COMPLETED))
        assert m.state == TurnState.IDLE

    def test_listening_emits_no_actions_on_speech_start(self):
        m = _machine(TurnState.IDLE)
        actions = m.handle(ev(TurnEventType.SPEECH_STARTED))
        assert actions == []

    def test_speech_stopped_arms_endpointing_timer(self):
        m = _machine(TurnState.LISTENING)
        actions = m.handle(ev(TurnEventType.SPEECH_STOPPED))
        assert _action_types(actions) == [TurnActionType.START_TIMER]
        timer = _action_with_payload(actions, TurnActionType.START_TIMER)
        assert timer.payload["key"] == TimerKey.ENDPOINTING.value
        assert timer.payload["duration_ms"] == 3000

    def test_user_transcript_final_cancels_endpointing(self):
        m = _machine(TurnState.LISTENING)
        actions = m.handle(ev(TurnEventType.USER_TRANSCRIPT_FINAL))
        assert m.state == TurnState.THINKING
        assert _action_types(actions) == [TurnActionType.CANCEL_TIMER]
        assert actions[0].payload["key"] == TimerKey.ENDPOINTING.value

    def test_user_transcript_final_can_defer_commit(self):
        m = _machine(TurnState.LISTENING)
        actions = m.handle(ev(
            TurnEventType.USER_TRANSCRIPT_FINAL,
            defer_commit=True,
            commit_delay_ms=900,
        ))
        assert m.state == TurnState.LISTENING
        assert _action_types(actions) == [TurnActionType.CANCEL_TIMER, TurnActionType.START_TIMER]
        assert actions[0].payload["key"] == TimerKey.ENDPOINTING.value
        assert actions[1].payload["key"] == TimerKey.ENDPOINTING.value
        assert actions[1].payload["duration_ms"] == 900

    def test_endpointing_timer_forces_turn_end(self):
        m = _machine(TurnState.LISTENING)
        actions = m.handle(timer_event(TimerKey.ENDPOINTING))
        assert m.state == TurnState.THINKING
        assert actions == []

    def test_response_started_is_informational(self):
        m = _machine(TurnState.THINKING)
        actions = m.handle(ev(TurnEventType.RESPONSE_STARTED))
        assert m.state == TurnState.THINKING
        assert actions == []







class TestBargeIn:
    def test_speech_during_speaking_enters_paused(self):
        m = _machine(TurnState.SPEAKING)
        actions = m.handle(ev(TurnEventType.SPEECH_STARTED))
        assert m.state == TurnState.PAUSED
        assert _action_types(actions) == [TurnActionType.PAUSE_OUTPUT, TurnActionType.START_TIMER]
        timer = _action_with_payload(actions, TurnActionType.START_TIMER)
        assert timer.payload["key"] == TimerKey.CONFIRM_INTERRUPT.value
        assert timer.payload["duration_ms"] == 800

    def test_confirm_timer_fires_interrupt(self):
        m = _machine(TurnState.PAUSED)
        actions = m.handle(timer_event(TimerKey.CONFIRM_INTERRUPT))
        assert m.state == TurnState.INTERRUPTED
        assert _action_types(actions) == [
            TurnActionType.STOP_TTS,
            TurnActionType.CANCEL_RESPONSE,
            TurnActionType.FLUSH_OUTPUT,
        ]

    def test_brief_cough_resumes_speaking(self):
        """PAUSED + SPEECH_STOPPED before the confirm timer fires → back to SPEAKING."""
        m = _machine(TurnState.SPEAKING)
        m.handle(ev(TurnEventType.SPEECH_STARTED))
        assert m.state == TurnState.PAUSED

        actions = m.handle(ev(TurnEventType.SPEECH_STOPPED))
        assert m.state == TurnState.SPEAKING
        assert _action_types(actions) == [TurnActionType.CANCEL_TIMER, TurnActionType.RESUME_OUTPUT]
        timer_action = _action_with_payload(actions, TurnActionType.CANCEL_TIMER)
        assert timer_action.payload["key"] == TimerKey.CONFIRM_INTERRUPT.value

    def test_interrupted_then_user_stops_returns_to_listening(self):
        m = _machine(TurnState.INTERRUPTED)
        actions = m.handle(ev(TurnEventType.SPEECH_STOPPED))
        assert m.state == TurnState.LISTENING
        assert _action_types(actions) == [TurnActionType.START_TIMER]

    def test_interrupted_then_transcript_goes_to_thinking(self):
        m = _machine(TurnState.INTERRUPTED)
        actions = m.handle(ev(TurnEventType.USER_TRANSCRIPT_FINAL))
        assert m.state == TurnState.THINKING
        assert _action_types(actions) == [TurnActionType.CANCEL_TIMER]

    def test_tts_completes_naturally_during_pause(self):
        """Race: TTS stream finished while we were paused. Settle to IDLE."""
        m = _machine(TurnState.PAUSED)
        actions = m.handle(ev(TurnEventType.TTS_COMPLETED))
        assert m.state == TurnState.IDLE
        assert _action_types(actions) == [TurnActionType.CANCEL_TIMER]

    def test_policy_disables_interrupt_during_speaking(self):
        policy = TurnPolicy(allow_interrupt_while_speaking=False)
        m = _machine(TurnState.SPEAKING, policy=policy)
        actions = m.handle(ev(TurnEventType.SPEECH_STARTED))
        assert m.state == TurnState.SPEAKING
        assert actions == []

    def test_speech_during_thinking_cancels_response(self):
        """User starts speaking while LLM is generating — treat as interrupt.

        Also stops the TTS worker because streaming responses can have a TTS
        task reading from the delta queue even before audio has started.
        """
        m = _machine(TurnState.THINKING)
        actions = m.handle(ev(TurnEventType.SPEECH_STARTED))
        assert m.state == TurnState.LISTENING
        assert _action_types(actions) == [
            TurnActionType.STOP_TTS,
            TurnActionType.CANCEL_RESPONSE,
        ]

    def test_policy_disables_interrupt_during_thinking(self):
        policy = TurnPolicy(allow_interrupt_while_speaking=False)
        m = _machine(TurnState.THINKING, policy=policy)
        actions = m.handle(ev(TurnEventType.SPEECH_STARTED))
        assert m.state == TurnState.THINKING
        assert actions == []







class TestClientCancel:
    def test_cancel_in_idle_is_noop(self):
        m = _machine(TurnState.IDLE)
        actions = m.handle(ev(TurnEventType.CLIENT_CANCEL))
        assert m.state == TurnState.IDLE
        assert actions == []

    def test_cancel_in_listening_drops_timer(self):
        m = _machine(TurnState.LISTENING)
        actions = m.handle(ev(TurnEventType.CLIENT_CANCEL))
        assert m.state == TurnState.IDLE
        assert _action_types(actions) == [TurnActionType.CANCEL_TIMER]

    def test_cancel_in_thinking_stops_tts_and_cancels_response(self):


        m = _machine(TurnState.THINKING)
        actions = m.handle(ev(TurnEventType.CLIENT_CANCEL))
        assert m.state == TurnState.IDLE
        assert _action_types(actions) == [
            TurnActionType.STOP_TTS,
            TurnActionType.CANCEL_RESPONSE,
        ]

    def test_cancel_in_speaking_stops_tts_and_response(self):
        m = _machine(TurnState.SPEAKING)
        actions = m.handle(ev(TurnEventType.CLIENT_CANCEL))
        assert m.state == TurnState.IDLE
        assert _action_types(actions) == [
            TurnActionType.STOP_TTS,
            TurnActionType.CANCEL_RESPONSE,
            TurnActionType.FLUSH_OUTPUT,
        ]

    def test_cancel_in_paused_stops_everything(self):
        m = _machine(TurnState.PAUSED)
        actions = m.handle(ev(TurnEventType.CLIENT_CANCEL))
        assert m.state == TurnState.IDLE
        assert _action_types(actions) == [
            TurnActionType.CANCEL_TIMER,
            TurnActionType.STOP_TTS,
            TurnActionType.CANCEL_RESPONSE,
            TurnActionType.FLUSH_OUTPUT,
        ]

    def test_cancel_in_interrupted_settles_to_idle(self):
        m = _machine(TurnState.INTERRUPTED)
        actions = m.handle(ev(TurnEventType.CLIENT_CANCEL))
        assert m.state == TurnState.IDLE
        assert actions == []







class TestUnhandledEvents:
    @pytest.mark.parametrize("state", [
        TurnState.IDLE,
        TurnState.LISTENING,
        TurnState.THINKING,
        TurnState.SPEAKING,
        TurnState.PAUSED,
        TurnState.INTERRUPTED,
    ])
    @pytest.mark.parametrize("event_type", list(TurnEventType))
    def test_any_state_event_pair_does_not_raise(self, state, event_type):
        """Smoke: every (state, event) combination returns a list without raising."""
        m = _machine(state)
        actions = m.handle(TurnEvent(type=event_type, payload={"key": TimerKey.ENDPOINTING.value}))
        assert isinstance(actions, list)

    def test_tts_completed_in_idle_is_noop(self):
        """TTS completed arriving after we already settled — ignore."""
        m = _machine(TurnState.IDLE)
        actions = m.handle(ev(TurnEventType.TTS_COMPLETED))
        assert m.state == TurnState.IDLE
        assert actions == []

    def test_speech_stopped_in_idle_is_noop(self):
        m = _machine(TurnState.IDLE)
        actions = m.handle(ev(TurnEventType.SPEECH_STOPPED))
        assert m.state == TurnState.IDLE
        assert actions == []

    def test_tts_audio_started_in_idle_promotes_to_speaking(self):
        """Supports agent-initiated greetings: TTS may start without a user turn first."""
        m = _machine(TurnState.IDLE)
        actions = m.handle(ev(TurnEventType.TTS_AUDIO_STARTED))
        assert m.state == TurnState.SPEAKING
        assert actions == []

    def test_timer_elapsed_with_wrong_key_in_paused_is_noop(self):
        """PAUSED only cares about CONFIRM_INTERRUPT, not ENDPOINTING."""
        m = _machine(TurnState.PAUSED)
        actions = m.handle(timer_event(TimerKey.ENDPOINTING))
        assert m.state == TurnState.PAUSED
        assert actions == []

    def test_timer_elapsed_with_wrong_key_in_listening_is_noop(self):
        """LISTENING only cares about ENDPOINTING."""
        m = _machine(TurnState.LISTENING)
        actions = m.handle(timer_event(TimerKey.CONFIRM_INTERRUPT))
        assert m.state == TurnState.LISTENING
        assert actions == []







class TestPolicy:
    def test_custom_confirm_duration(self):
        policy = TurnPolicy(min_interrupt_duration_ms=150)
        m = _machine(TurnState.SPEAKING, policy=policy)
        actions = m.handle(ev(TurnEventType.SPEECH_STARTED))
        timer = _action_with_payload(actions, TurnActionType.START_TIMER)
        assert timer.payload["duration_ms"] == 150

    def test_custom_endpointing_delay(self):
        policy = TurnPolicy(max_endpointing_delay_ms=5000)
        m = _machine(TurnState.LISTENING, policy=policy)
        actions = m.handle(ev(TurnEventType.SPEECH_STOPPED))
        timer = _action_with_payload(actions, TurnActionType.START_TIMER)
        assert timer.payload["duration_ms"] == 5000







class TestResetAndReentry:
    def test_reset_returns_to_idle(self):
        m = _machine(TurnState.SPEAKING)
        m.reset()
        assert m.state == TurnState.IDLE

    def test_second_turn_works_after_first(self):
        m = TurnStateMachine()

        m.handle(ev(TurnEventType.SPEECH_STARTED))
        m.handle(ev(TurnEventType.USER_TRANSCRIPT_FINAL))
        m.handle(ev(TurnEventType.TTS_AUDIO_STARTED))
        m.handle(ev(TurnEventType.TTS_COMPLETED))
        assert m.state == TurnState.IDLE


        m.handle(ev(TurnEventType.SPEECH_STARTED))
        assert m.state == TurnState.LISTENING







class TestActionPayloads:
    def test_start_timer_has_key_and_duration(self):
        m = _machine(TurnState.SPEAKING)
        actions = m.handle(ev(TurnEventType.SPEECH_STARTED))
        timer = _action_with_payload(actions, TurnActionType.START_TIMER)
        assert "key" in timer.payload
        assert "duration_ms" in timer.payload

    def test_cancel_timer_has_key(self):
        m = _machine(TurnState.LISTENING)
        actions = m.handle(ev(TurnEventType.USER_TRANSCRIPT_FINAL))
        cancel = _action_with_payload(actions, TurnActionType.CANCEL_TIMER)
        assert cancel.payload["key"] == TimerKey.ENDPOINTING.value

    def test_side_effect_actions_have_empty_payload(self):
        m = _machine(TurnState.SPEAKING)
        actions = m.handle(ev(TurnEventType.CLIENT_CANCEL))
        for action in actions:
            if action.type in (TurnActionType.STOP_TTS, TurnActionType.CANCEL_RESPONSE, TurnActionType.FLUSH_OUTPUT):
                assert action.payload == {}
