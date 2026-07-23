"""Adversarial tests for single-owner response state (stabilization F2/F3).

Covers:
  * control-plane start/append/commit racing VAD-driven interruption
  * TTS completion racing an explicit client cancel
  * recovery when a lifecycle-critical state-machine action fails
  * orchestrator generation aliveness always matching the session lifecycle
"""

from __future__ import annotations

import asyncio
import inspect
import re
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
from vox.conversation import session as session_module
from vox.conversation.response_lifecycle import TerminalRecord
from vox.conversation.session import (
    ERROR_CODE_RESPONSE_STALE_GENERATION,
    WIRE_AUDIO_CLEAR,
    WIRE_AUDIO_DELTA,
    WIRE_ERROR,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_COMMITTED,
    WIRE_RESPONSE_DONE,
    AppendResult,
    ConversationConfig,
    ConversationSession,
)
from vox.core.types import SynthesizeChunk
from vox.operations.conversation import (
    ConversationOrchestrator,
    ConvResponseDoneEvent,
    parse_session_update,
)
from vox.operations.errors import ConversationCommandError, InvalidConfigError
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
        response_id = (await session.start_response_stream()).response_id
        assert response_id is not None
        assert (
            await session.append_response_text(
                "A long reply that keeps going on.",
                expected_response_id=response_id,
            )
            is AppendResult.ACCEPTED
        )
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)

        results: list[AppendResult] = []

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

        record = session.terminal_record
        assert record is not None
        assert record.reason == "cancelled"
        assert record.response_id == response_id
        assert not session.response_active
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)

        rejections = [result for result in results if result is not AppendResult.ACCEPTED]
        if rejections:
            first_rejected = results.index(rejections[0])
            assert all(result is not AppendResult.ACCEPTED for result in results[first_rejected:])
            assert set(rejections) <= {AppendResult.NO_ACTIVE_RESPONSE, AppendResult.STREAM_ENDED}

        assert (
            await session.append_response_text(
                "into the void",
                expected_response_id=response_id,
            )
            is AppendResult.NO_ACTIVE_RESPONSE
        )
        assert session.terminal_record is record
        assert (
            await session.commit_response_stream(expected_response_id=response_id)
            is AppendResult.NO_ACTIVE_RESPONSE
        )
        assert not collector.by_type(WIRE_RESPONSE_DONE)

        await session.close()

    @pytest.mark.asyncio
    async def test_start_racing_vad_speech_is_serialized_with_turn_events(self):
        session, collector, adapter = _build_session()
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        result = await session.start_response_stream()
        await _drain_events(session)

        assert result.response_id is None
        assert result.rejection is not None
        assert session.state == TurnState.LISTENING
        assert not session.response_active
        assert adapter.texts == []
        assert collector.by_type(WIRE_ERROR) == []

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
        record = session.terminal_record
        assert record is not None
        assert record.reason == "cancelled"
        assert record.response_id == response_id
        assert session._tts_task is None or session._tts_task.done()
        assert session.pending_audio_count == 0

        errors = collector.by_type(WIRE_ERROR)
        assert any("flush_output" in event["message"] for event in errors)
        recovery_errors = [e for e in errors if "response pipeline reset" in e["message"]]
        assert recovery_errors
        assert recovery_errors[0]["code"] == "session_failed"
        assert recovery_errors[0]["recoverable"] is False
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
        response_id = (await session.start_response_stream()).response_id
        assert response_id is not None
        stream = session._response_stream
        assert stream is not None

        original_append = stream.append_text

        async def cancel_then_append(text: str) -> AppendResult:
            stream.append_text = original_append
            await session.cancel_response()
            assert stream.closed
            return await stream.append_text(text)

        stream.append_text = cancel_then_append

        assert (
            await session.append_response_text("lost delta", expected_response_id=response_id)
            is AppendResult.STREAM_ENDED
        )
        assert stream.queue.empty()
        record = session.terminal_record
        assert record is not None
        assert record.reason == "cancelled"
        assert record.response_id == response_id

        await session.close()

    @pytest.mark.asyncio
    async def test_commit_response_stream_reports_session_closed_after_close(self):
        session, _, _ = _build_session()
        await session.start()
        await session.close()

        assert await session.commit_response_stream() is AppendResult.SESSION_CLOSED


class TestCloseRacesQueuedCommit:
    @pytest.mark.asyncio
    async def test_queued_commit_drained_during_close_reports_session_closed(self):
        tts = GatedTTS()
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        response_id = (await session.start_response_stream()).response_id
        assert response_id is not None
        assert (
            await session.append_response_text(
                "reply awaiting commit.",
                expected_response_id=response_id,
            )
            is AppendResult.ACCEPTED
        )
        assert await _wait_for(tts.started.is_set)
        assert session._tts_task is not None and not session._tts_task.done()
        assert await _wait_for(session._event_queue.empty)

        original_admit = session._admit_response_commit
        admissions: list[object] = []

        async def recording_admit(**kwargs):
            result = await original_admit(**kwargs)
            admissions.append(result)
            return result

        session._admit_response_commit = recording_admit

        commit_task = asyncio.create_task(
            session.commit_response_stream(expected_response_id=response_id)
        )
        await asyncio.sleep(0)
        assert not session._event_queue.empty()
        assert not session._closed

        await session.close()

        assert await commit_task is AppendResult.SESSION_CLOSED
        assert admissions == [AppendResult.SESSION_CLOSED]
        assert collector.by_type(WIRE_RESPONSE_COMMITTED) == []


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

        new_response_id = (await asyncio.wait_for(session.start_response_stream(), timeout=1.0)).response_id
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
        record = session.terminal_record
        assert record is not None
        assert record.reason == "cancelled"
        assert record.generation_id == "gen-1"
        assert not orchestrator.response_active
        assert orchestrator.active_generation_id is None

        with pytest.raises(InvalidConfigError):
            await orchestrator.append_response_text("late delta", generation_id="gen-1")
        with pytest.raises(InvalidConfigError):
            await orchestrator.commit_response(generation_id="gen-1")

        assert not session.response_active

        await orchestrator.close()

    @pytest.mark.asyncio
    async def test_rejected_delta_after_commit_keeps_live_generation_correlated(self):
        adapter = GatedTTS()
        orchestrator = ConversationOrchestrator(scheduler=FakeScheduler(adapter))
        await orchestrator.start_session(_session_config())

        await orchestrator.start_response(generation_id="gen-live")
        await orchestrator.append_response_text("Streaming sentence one.", generation_id="gen-live")
        assert await _wait_for(adapter.started.is_set)
        await orchestrator.commit_response(generation_id="gen-live")

        with pytest.raises(ConversationCommandError, match="already committed") as exc_info:
            await orchestrator.append_response_text("late tail", generation_id="gen-live")
        assert exc_info.value.code == ERROR_CODE_RESPONSE_STALE_GENERATION
        assert exc_info.value.generation_id == "gen-live"
        assert orchestrator.response_active
        assert orchestrator.active_generation_id == "gen-live"

        adapter.finish_gate.set()
        await orchestrator.end_of_stream()
        events = []
        async for event in orchestrator.events():
            events.append(event)
        done_events = [event for event in events if isinstance(event, ConvResponseDoneEvent)]
        assert done_events
        assert done_events[0].generation_id == "gen-live"

        await orchestrator.close()

    @pytest.mark.asyncio
    async def test_stale_generation_mismatch_leaves_live_generation_untouched(self):
        adapter = GatedTTS()
        orchestrator = ConversationOrchestrator(scheduler=FakeScheduler(adapter))
        await orchestrator.start_session(_session_config())
        session = orchestrator._session
        assert session is not None

        await orchestrator.start_response(generation_id="gen-a")
        stale_response_id = session.active_response_id
        assert stale_response_id is not None

        original_append = session.append_response_text
        swap_completed = asyncio.Event()

        async def swap_then_append(text: str, **kwargs):
            session.append_response_text = original_append
            await orchestrator.cancel_response(generation_id="gen-a")
            await orchestrator.start_response(generation_id="gen-b")
            swap_completed.set()
            return await original_append(text, **kwargs)

        session.append_response_text = swap_then_append

        with pytest.raises(
            ConversationCommandError,
            match="response id does not match the active response",
        ) as exc_info:
            await orchestrator.append_response_text("late for gen-a", generation_id="gen-a")

        assert swap_completed.is_set()
        assert exc_info.value.code == ERROR_CODE_RESPONSE_STALE_GENERATION
        assert exc_info.value.generation_id == "gen-a"

        live_response_id = session.active_response_id
        assert live_response_id is not None
        assert live_response_id != stale_response_id
        assert orchestrator.response_active
        assert orchestrator.active_generation_id == "gen-b"
        assert session.active_response_id == live_response_id

        assert (
            await session.append_response_text(
                "stale delta",
                expected_response_id=stale_response_id,
            )
            is AppendResult.RESPONSE_MISMATCH
        )
        assert (
            await session.commit_response_stream(expected_response_id=stale_response_id)
            is AppendResult.RESPONSE_MISMATCH
        )
        with pytest.raises(ConversationCommandError) as commit_exc:
            await orchestrator.commit_response(generation_id="gen-a")
        assert commit_exc.value.code == ERROR_CODE_RESPONSE_STALE_GENERATION
        assert orchestrator.response_active
        assert orchestrator.active_generation_id == "gen-b"

        assert (
            await session.append_response_text(
                "Sentence for gen-b.",
                expected_response_id=live_response_id,
            )
            is AppendResult.ACCEPTED
        )
        assert await _wait_for(adapter.started.is_set)
        adapter.finish_gate.set()
        await orchestrator.commit_response(generation_id="gen-b")
        await orchestrator.end_of_stream()
        events = []
        async for event in orchestrator.events():
            events.append(event)
        done_events = [event for event in events if isinstance(event, ConvResponseDoneEvent)]
        assert done_events
        assert done_events[-1].generation_id == "gen-b"
        assert done_events[-1].response_id == live_response_id

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


def _assert_orchestrator_session_agreement(orchestrator, session) -> None:
    assert orchestrator.response_active == session.response_active
    assert orchestrator.active_generation_id == session.active_generation_id
    if session._response_stream is None:
        assert orchestrator.active_generation_id is None
        assert session.active_response_id is None


class TestOwnershipInvariantI2:
    @pytest.mark.asyncio
    async def test_orchestrator_view_matches_session_across_lifecycle_sequences(self):
        sequences = [
            ("start", "append", "commit", "settle"),
            ("start", "append", "cancel"),
            ("start", "cancel", "start", "append", "commit", "settle"),
            ("start", "append", "cancel", "start", "append", "commit", "settle"),
            ("cancel", "start", "append", "commit", "settle"),
            ("start", "append", "commit", "settle", "cancel", "start", "append", "cancel"),
        ]
        for sequence in sequences:
            adapter = ScriptedTTSAdapter(chunks=2, inter_chunk_delay=0.005)
            orchestrator = ConversationOrchestrator(scheduler=FakeScheduler(adapter))
            await orchestrator.start_session(_session_config())
            session = orchestrator._session
            assert session is not None
            generation = 0
            generation_id = None
            for op in sequence:
                if op == "start":
                    generation += 1
                    generation_id = f"gen-{generation}"
                    await orchestrator.start_response(generation_id=generation_id)
                elif op == "append":
                    await orchestrator.append_response_text(
                        "A reply sentence.", generation_id=generation_id
                    )
                elif op == "commit":
                    await orchestrator.commit_response(generation_id=generation_id)
                elif op == "cancel":
                    await orchestrator.cancel_response()
                elif op == "settle":
                    await session.wait_until_settled()
                _assert_orchestrator_session_agreement(orchestrator, session)
            await orchestrator.close()

    @pytest.mark.asyncio
    async def test_agreement_holds_through_completion_cancel_races(self):
        for delay_s in (0.0, 0.002, 0.005, 0.01):
            tts = GatedTTS()
            orchestrator = ConversationOrchestrator(scheduler=FakeScheduler(tts))
            await orchestrator.start_session(_session_config())
            session = orchestrator._session
            assert session is not None

            await orchestrator.start_response(generation_id="gen-race")
            await orchestrator.append_response_text("racy reply.", generation_id="gen-race")
            await orchestrator.commit_response(generation_id="gen-race")
            assert await _wait_for(tts.started.is_set)
            _assert_orchestrator_session_agreement(orchestrator, session)

            tts.finish_gate.set()
            if delay_s:
                await asyncio.sleep(delay_s)
            await orchestrator.cancel_response()
            _assert_orchestrator_session_agreement(orchestrator, session)

            await session.wait_until_settled()
            _assert_orchestrator_session_agreement(orchestrator, session)
            assert not orchestrator.response_active
            record = session.terminal_record
            assert record is not None
            assert record.reason in {"done", "cancelled"}
            assert record.generation_id == "gen-race"

            with pytest.raises(ConversationCommandError):
                await orchestrator.append_response_text("after death", generation_id="gen-race")
            _assert_orchestrator_session_agreement(orchestrator, session)

            await orchestrator.close()


class TestTerminalizeIdentity:
    @pytest.mark.asyncio
    async def test_stale_stream_outcome_cannot_terminalize_or_emit(self):
        session, collector, _ = _build_session()
        stale = session._response_lifecycle.start_stream(
            output=session._default_response_output,
            generation_id="gen-old",
        )
        first_record = session._response_lifecycle.terminalize(stale, "cancelled")
        assert first_record is not None
        live = session._response_lifecycle.start_stream(
            output=session._default_response_output,
            generation_id="gen-new",
        )

        events_before = list(collector.events)
        await session._apply_tts_stream_outcome(
            TurnEvent(type=TurnEventType.TTS_COMPLETED),
            stale,
            True,
        )

        assert collector.events == events_before
        assert session._response_lifecycle.stream is live
        assert not live.closed
        assert session.terminal_record is None

    @pytest.mark.asyncio
    async def test_paused_pending_done_stream_survives_unaccepted_tts_completion(self):
        session, _, _ = _build_session()
        stream = session._response_lifecycle.start_stream(
            output=session._default_response_output,
        )
        stream.pending_done = True

        await session._apply_tts_stream_outcome(
            TurnEvent(type=TurnEventType.TTS_COMPLETED),
            stream,
            False,
        )

        assert session._response_lifecycle.stream is stream
        assert not stream.closed
        assert session.terminal_record is None


class TestEmissionIdempotence:
    @pytest.mark.asyncio
    async def test_cancel_after_teardown_emits_no_second_terminal_events(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply.")
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)
        response_id = session.active_response_id
        assert response_id is not None

        await session.cancel_response()
        assert len(collector.by_type(WIRE_RESPONSE_CANCELLED)) == 1
        assert len(collector.by_type(WIRE_AUDIO_CLEAR)) == 1

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        assert session.state == TurnState.THINKING
        await session.cancel_response()

        assert session.state == TurnState.IDLE
        cancelled = collector.by_type(WIRE_RESPONSE_CANCELLED)
        clears = collector.by_type(WIRE_AUDIO_CLEAR)
        assert len(cancelled) == 1
        assert len(clears) == 1
        assert cancelled[0]["response_id"] == response_id
        assert clears[0]["response_id"] == response_id

        await session.close()


class TestTerminalEventUnrepresentability:
    def test_no_terminal_emit_call_site_passes_raw_ids(self):
        source = inspect.getsource(session_module)
        assert not re.search(r"_emit_response_(?:created|committed|done|cancelled)\(\s*[\"']", source)
        assert not re.search(r"_emit_audio_clear\(\s*[\"']", source)
        assert not re.search(r"_emit_response_event\(\s*WIRE_[A-Z_]+\s*,\s*[\"']", source)

        hints = inspect.get_annotations(session_module.ConversationSession._emit_response_done)
        assert hints["record"] == "TerminalRecord"
        hints = inspect.get_annotations(session_module.ConversationSession._emit_response_cancelled)
        assert hints["record"] == "TerminalRecord"

    @pytest.mark.asyncio
    async def test_teardown_events_carry_the_records_exact_identity_pair(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        result = await session.start_response_stream(generation_id="gen-pair")
        assert result.response_id is not None
        assert (
            await session.append_response_text(
                "A long reply that keeps going on.",
                expected_response_id=result.response_id,
            )
            is AppendResult.ACCEPTED
        )
        assert await _wait_for(lambda: session.state == TurnState.SPEAKING)

        await session.cancel_response()

        record = session.terminal_record
        assert record == TerminalRecord(
            response_id=result.response_id,
            generation_id="gen-pair",
            reason="cancelled",
        )
        cancelled = collector.by_type(WIRE_RESPONSE_CANCELLED)
        clears = collector.by_type(WIRE_AUDIO_CLEAR)
        assert cancelled and clears
        for event in (cancelled[-1], clears[-1]):
            assert event["response_id"] == record.response_id
            assert event["generation_id"] == record.generation_id

        await session.close()
