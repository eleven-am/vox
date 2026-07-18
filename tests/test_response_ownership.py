"""Adversarial tests for single-owner response state (stabilization F2/F3).

Covers:
  * control-plane start/append/commit racing VAD-driven interruption
  * TTS completion racing an explicit client cancel
  * recovery when a lifecycle-critical state-machine action fails
  * orchestrator generation aliveness always matching the session lifecycle
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from tests.fakes import FakeScheduler
from tests.test_conversation_session import (
    EventCollector,
    HangingReleaseScheduler,
    ScriptedTTSAdapter,
    _AcceptAllClassifier,
    _build_session,
    _drain_events,
)
from vox.conversation import TurnEvent, TurnEventType, TurnPolicy, TurnState
from vox.conversation.response_lifecycle import ResponsePhase
from vox.conversation.session import (
    WIRE_AUDIO_CLEAR,
    WIRE_AUDIO_DELTA,
    WIRE_ERROR,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_DONE,
    ConversationConfig,
    ConversationSession,
)
from vox.core.types import SynthesizeChunk
from vox.operations.conversation import ConversationOrchestrator, parse_session_update
from vox.operations.errors import InvalidConfigError
from vox.streaming.types import SpeechStarted, StreamTranscript


class GatedTTS(ScriptedTTSAdapter):
    def __init__(self) -> None:
        super().__init__(chunks=1)
        self.started = threading.Event()
        self.finish_gate = threading.Event()

    async def synthesize(self, text: str, **_kwargs):
        self.last_text = text
        self.texts.append(text)
        yield SynthesizeChunk(
            audio=np.full(256, 0.01, dtype=np.float32).tobytes(),
            sample_rate=24_000,
            is_final=False,
        )
        self.started.set()
        while not self.finish_gate.is_set():
            await asyncio.sleep(0.002)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class BrokenTTS(ScriptedTTSAdapter):
    async def synthesize(self, text: str, **_kwargs):
        raise RuntimeError("tts exploded")
        yield  # pragma: no cover


async def _wait_for(predicate, *, timeout_s: float = 2.0, interval_s: float = 0.005) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


def _session_config():
    return parse_session_update(
        {"session": {"stt_model": "x:1", "tts_model": "y:1", "voice": "default"}}
    )


class TestControlPlaneVsInterruption:
    @pytest.mark.asyncio
    async def test_append_and_commit_after_confirmed_interrupt_are_rejected(self):
        tts = ScriptedTTSAdapter(chunks=60, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        response_id = await session.start_response_stream()
        assert response_id is not None
        assert await session.append_response_text(
            "A long reply that keeps going on.",
            expected_response_id=response_id,
        )
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)

        results: list[bool] = []

        async def spam_appends() -> None:
            for index in range(40):
                results.append(
                    await session.append_response_text(
                        f" chunk {index}.",
                        expected_response_id=response_id,
                    )
                )
                await asyncio.sleep(0.004)

        spam = asyncio.create_task(spam_appends())
        await asyncio.sleep(0.02)
        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000, utterance_id=1))
        await session._forward_stream_event(
            StreamTranscript(
                text="I need",
                is_partial=True,
                start_ms=1000,
                end_ms=1500,
                audio_duration_ms=500,
                utterance_id=1,
            )
        )
        assert await _wait_for(lambda: session.state == TurnState.INTERRUPTED)
        await spam

        assert session.response_phase == ResponsePhase.CANCELLED
        assert not session.response_active
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)

        if False in results:
            first_rejected = results.index(False)
            assert all(result is False for result in results[first_rejected:])

        assert not await session.append_response_text(
            "into the void",
            expected_response_id=response_id,
        )
        assert session.response_phase == ResponsePhase.CANCELLED
        assert not await session.commit_response_stream(expected_response_id=response_id)
        assert not collector.by_type(WIRE_RESPONSE_DONE)

        await session.close()

    @pytest.mark.asyncio
    async def test_start_racing_vad_speech_is_serialized_with_turn_events(self):
        session, collector, adapter = _build_session()
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        response_id = await session.start_response_stream()
        await _drain_events(session)

        assert response_id is None
        assert session.state == TurnState.LISTENING
        assert not session.response_active
        assert adapter.texts == []
        assert collector.by_type(WIRE_ERROR)

        await session.close()


class TestCompletionVsCancelRace:
    async def _run_completion_cancel_race(self, delay_s: float) -> None:
        tts = GatedTTS()
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("racy reply.")
        assert await _wait_for(tts.started.is_set)
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)

        async def cancel_later() -> None:
            if delay_s:
                await asyncio.sleep(delay_s)
            await session.cancel_response()

        tts.finish_gate.set()
        await asyncio.gather(cancel_later())
        await session.wait_until_settled()

        done_events = collector.by_type(WIRE_RESPONSE_DONE)
        cancelled_events = collector.by_type(WIRE_RESPONSE_CANCELLED)
        terminal_count = len(done_events) + len(cancelled_events)
        assert terminal_count == 1, (
            f"delay={delay_s}: expected one terminal event, got done={done_events} "
            f"cancelled={cancelled_events}"
        )
        assert session.state == TurnState.IDLE
        assert not session.response_active

        next_tts_gate_open = GatedTTS()
        next_tts_gate_open.finish_gate.set()
        session._scheduler = FakeScheduler(next_tts_gate_open)  # type: ignore[attr-defined]
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        before_done = len(collector.by_type(WIRE_RESPONSE_DONE))
        await session.submit_response_text("follow-up reply.")
        assert await _wait_for(lambda: len(collector.by_type(WIRE_RESPONSE_DONE)) == before_done + 1)
        assert session.state == TurnState.IDLE

        await session.close()

    @pytest.mark.asyncio
    async def test_completion_racing_cancel_yields_single_terminal_event(self):
        for delay_s in (0.0, 0.002, 0.005, 0.01, 0.02):
            await self._run_completion_cancel_race(delay_s)


class TestActionFailureRecovery:
    @pytest.mark.asyncio
    async def test_flush_failure_during_cancel_recovers_to_idle(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply.")
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)
        response_id = session.active_response_id
        assert response_id is not None

        original_flush = session._audio_output.flush

        def broken_flush() -> None:
            raise RuntimeError("flush exploded")

        session._audio_output.flush = broken_flush  # type: ignore[method-assign]
        await session.cancel_response()

        assert session.state == TurnState.IDLE
        assert not session.response_active
        assert session.response_phase == ResponsePhase.CANCELLED
        assert session._tts_task is None or session._tts_task.done()
        assert session.pending_audio_count == 0

        errors = collector.by_type(WIRE_ERROR)
        assert any("flush_output" in event["message"] for event in errors)
        clears = collector.by_type(WIRE_AUDIO_CLEAR)
        cancelled = collector.by_type(WIRE_RESPONSE_CANCELLED)
        assert clears and clears[-1]["response_id"] == response_id
        assert cancelled and cancelled[-1]["response_id"] == response_id

        session._audio_output.flush = original_flush  # type: ignore[method-assign]
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("next turn works.")
        assert await _wait_for(lambda: bool(collector.by_type(WIRE_RESPONSE_DONE)))
        assert session.state == TurnState.IDLE

        await session.close()

    @pytest.mark.asyncio
    async def test_pause_failure_during_barge_in_recovers_to_idle(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply.")
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)

        original_pause = session._audio_output.pause

        def broken_pause() -> None:
            raise RuntimeError("pause exploded")

        session._audio_output.pause = broken_pause  # type: ignore[method-assign]
        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        assert await _wait_for(lambda: session.state == TurnState.IDLE)

        assert not session.response_active
        errors = collector.by_type(WIRE_ERROR)
        assert any("pause_output" in event["message"] for event in errors)
        assert collector.by_type(WIRE_AUDIO_CLEAR)
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)

        session._audio_output.pause = original_pause  # type: ignore[method-assign]
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("next turn works.")
        assert await _wait_for(lambda: bool(collector.by_type(WIRE_RESPONSE_DONE)))
        assert session.state == TurnState.IDLE

        await session.close()

    @pytest.mark.asyncio
    async def test_stop_tts_failure_recovers_without_duplicate_terminal_events(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply.")
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)

        detector = session._interrupt_detector
        original_reset = detector.reset

        def broken_reset() -> None:
            raise RuntimeError("reset exploded")

        detector.reset = broken_reset  # type: ignore[method-assign]
        await session.cancel_response()
        detector.reset = original_reset  # type: ignore[method-assign]

        assert session.state == TurnState.IDLE
        assert not session.response_active
        errors = collector.by_type(WIRE_ERROR)
        assert any("stop_tts" in event["message"] for event in errors)
        assert len(collector.by_type(WIRE_RESPONSE_CANCELLED)) == 1

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("next turn works.")
        assert await _wait_for(lambda: bool(collector.by_type(WIRE_RESPONSE_DONE)))
        assert session.state == TurnState.IDLE

        await session.close()


class TestRunnerReentrantCancel:
    @pytest.mark.asyncio
    async def test_cancel_response_from_command_on_runner_task_completes(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply.")
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)

        async def cancel_on_runner() -> bool:
            assert asyncio.current_task() is session._runner
            await session.cancel_response()
            return True

        result = await asyncio.wait_for(session._submit_command(cancel_on_runner), timeout=2.0)

        assert result is True
        assert session.state == TurnState.IDLE
        assert not session.response_active
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)

        await session.close()

    @pytest.mark.asyncio
    async def test_cancel_response_from_on_event_callback_completes(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        original_emit = session._on_event
        cancelled_from_callback = asyncio.Event()

        async def cancelling_emit(event: dict) -> None:
            await original_emit(event)
            is_speaking_change = event.get("type") == "turn.state_changed" and event.get("state") == "speaking"
            if is_speaking_change and not cancelled_from_callback.is_set():
                cancelled_from_callback.set()
                await asyncio.wait_for(session.cancel_response(), timeout=2.0)

        session._on_event = cancelling_emit

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply.")

        assert await _wait_for(cancelled_from_callback.is_set)
        assert await _wait_for(lambda: session.state == TurnState.IDLE)
        assert not session.response_active
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)

        await session.close()


class TestAppendIntoDeadStream:
    @pytest.mark.asyncio
    async def test_append_after_teardown_between_admission_and_put_returns_false(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, _, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        response_id = await session.start_response_stream()
        assert response_id is not None
        stream = session._response_stream
        assert stream is not None

        original_append = stream.append_text

        async def cancel_then_append(text: str) -> bool:
            stream.append_text = original_append
            await session.cancel_response()
            assert stream.closed
            return await stream.append_text(text)

        stream.append_text = cancel_then_append

        assert not await session.append_response_text("lost delta", expected_response_id=response_id)
        assert stream.queue.empty()
        assert session.response_phase == ResponsePhase.CANCELLED

        await session.close()

    @pytest.mark.asyncio
    async def test_commit_response_stream_returns_false_after_close(self):
        session, _, _ = _build_session()
        await session.start()
        await session.close()

        assert not await session.commit_response_stream()


class TestSlowTeardownDoesNotStallLoop:
    @pytest.mark.asyncio
    async def test_hanging_tts_teardown_keeps_loop_responsive_and_gates_stale_audio(self):
        tts = ScriptedTTSAdapter(chunks=50, inter_chunk_delay=0.01)
        scheduler = HangingReleaseScheduler(tts)
        collector = EventCollector()
        config = ConversationConfig(
            stt_model="fake-stt:latest",
            tts_model="fake-tts:latest",
            voice="default",
            language="en",
            policy=TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200),
            interrupt_classifier=_AcceptAllClassifier(),
        )
        session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply")
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)
        old_task = session._tts_task
        old_response_id = session.active_response_id
        assert old_task is not None
        assert old_response_id is not None

        started = time.monotonic()
        await asyncio.wait_for(session.cancel_response(), timeout=1.0)
        assert session._tts_task is None
        assert not old_task.done()

        new_response_id = await asyncio.wait_for(session.start_response_stream(), timeout=1.0)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert new_response_id is not None
        assert new_response_id != old_response_id
        assert not old_task.done()

        await asyncio.sleep(0.05)
        clear_indices = [
            index for index, event in enumerate(collector.events) if event.get("type") == WIRE_AUDIO_CLEAR
        ]
        assert clear_indices
        stale_deltas = [
            event
            for event in collector.events[clear_indices[-1] + 1 :]
            if event.get("type") == WIRE_AUDIO_DELTA and event.get("response_id") == old_response_id
        ]
        assert stale_deltas == []

        scheduler.release_gate.set()
        assert await _wait_for(old_task.done)

        await session.close()


class TestRecoveryIdempotence:
    @pytest.mark.asyncio
    async def test_early_stop_tts_failure_does_not_double_emit_terminal_events(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply.")
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)
        response_id = session.active_response_id
        assert response_id is not None

        original_mark_ended = session._mark_agent_speech_ended

        def broken_mark_ended() -> None:
            raise RuntimeError("speech-end bookkeeping exploded")

        session._mark_agent_speech_ended = broken_mark_ended  # type: ignore[method-assign]
        await session.cancel_response()
        session._mark_agent_speech_ended = original_mark_ended  # type: ignore[method-assign]

        assert session.state == TurnState.IDLE
        assert not session.response_active
        errors = collector.by_type(WIRE_ERROR)
        assert any("stop_tts" in event["message"] for event in errors)
        cancelled = collector.by_type(WIRE_RESPONSE_CANCELLED)
        clears = collector.by_type(WIRE_AUDIO_CLEAR)
        assert len(cancelled) == 1
        assert len(clears) == 1
        assert cancelled[0]["response_id"] == response_id
        assert clears[0]["response_id"] == response_id

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("next turn works.")
        assert await _wait_for(lambda: bool(collector.by_type(WIRE_RESPONSE_DONE)))
        assert session.state == TurnState.IDLE

        await session.close()


class TestOrchestratorSessionAgreement:
    @pytest.mark.asyncio
    async def test_orchestrator_aliveness_matches_session_after_tts_failure(self):
        orchestrator = ConversationOrchestrator(scheduler=FakeScheduler(BrokenTTS()))
        await orchestrator.start_session(_session_config())
        session = orchestrator._session
        assert session is not None

        await orchestrator.start_response(generation_id="gen-1")
        assert orchestrator.response_active == session.response_active
        assert orchestrator.active_generation_id == "gen-1"

        await orchestrator.append_response_text("boom", generation_id="gen-1")
        await orchestrator.commit_response(generation_id="gen-1")

        assert await _wait_for(lambda: not session.response_active)
        assert not orchestrator.response_active
        assert orchestrator.active_generation_id is None

        await orchestrator.start_response(generation_id="gen-2")
        assert orchestrator.response_active == session.response_active
        assert orchestrator.active_generation_id == "gen-2"

        await orchestrator.close()

    @pytest.mark.asyncio
    async def test_orchestrator_rejects_commands_for_cancelled_generation(self):
        adapter = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        orchestrator = ConversationOrchestrator(scheduler=FakeScheduler(adapter))
        await orchestrator.start_session(_session_config())
        session = orchestrator._session
        assert session is not None

        await orchestrator.start_response(generation_id="gen-1")
        await orchestrator.append_response_text("A very long reply.", generation_id="gen-1")
        await orchestrator.cancel_response(generation_id="gen-1")

        assert not session.response_active
        assert session.response_phase == ResponsePhase.CANCELLED
        assert not orchestrator.response_active
        assert orchestrator.active_generation_id is None

        with pytest.raises(InvalidConfigError):
            await orchestrator.append_response_text("late delta", generation_id="gen-1")
        with pytest.raises(InvalidConfigError):
            await orchestrator.commit_response(generation_id="gen-1")

        assert not session.response_active

        await orchestrator.close()

    @pytest.mark.asyncio
    async def test_live_generation_still_reports_active_on_both_sides(self):
        adapter = GatedTTS()
        orchestrator = ConversationOrchestrator(scheduler=FakeScheduler(adapter))
        await orchestrator.start_session(_session_config())
        session = orchestrator._session
        assert session is not None

        await orchestrator.start_response(generation_id="gen-live")
        await orchestrator.append_response_text("Streaming sentence one.", generation_id="gen-live")
        assert await _wait_for(adapter.started.is_set)

        assert orchestrator.response_active
        assert session.response_active
        with pytest.raises(InvalidConfigError, match="response already active"):
            await orchestrator.start_response(generation_id="gen-other")

        adapter.finish_gate.set()
        await orchestrator.commit_response(generation_id="gen-live")
        assert await _wait_for(lambda: not session.response_active)
        assert not orchestrator.response_active

        await orchestrator.close()
