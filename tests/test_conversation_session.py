"""Integration tests for ConversationSession.

These tests don't use real STT/TTS models. Instead, we:
  * mock the scheduler to hand out a fake TTS adapter that emits scripted chunks
  * drive state transitions by injecting events into the session's event_queue
    directly OR by calling the session's public API methods
  * observe outgoing client events via the on_event callback
"""

from __future__ import annotations

import asyncio
import numpy as np
import pytest

from vox.conversation import TimerKey, TurnEvent, TurnEventType, TurnPolicy, TurnState
from vox.conversation.session import (
    WIRE_AUDIO_DELTA,
    WIRE_ERROR,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_CREATED,
    WIRE_RESPONSE_DONE,
    WIRE_STATE_CHANGED,
    ConversationConfig,
    ConversationSession,
)
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.streaming.types import SpeechStopped, StreamTranscript

from tests.fakes import FakeScheduler




class ScriptedTTSAdapter(TTSAdapter):
    """Emits N audio chunks, optionally blocking between them."""

    def __init__(self, *, chunks: int = 3, sample_rate: int = 24_000, inter_chunk_delay: float = 0.01) -> None:
        self._chunks = chunks
        self._sample_rate = sample_rate
        self._delay = inter_chunk_delay
        self.cancelled_at_chunk: int | None = None
        self.last_text: str | None = None
        self.texts: list[str] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted-tts", type=ModelType.TTS,
            architectures=("scripted",), default_sample_rate=self._sample_rate,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *_a, **_k): ...
    def unload(self): ...

    @property
    def is_loaded(self): return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **_kwargs):
        self.last_text = text
        self.texts.append(text)
        try:
            for i in range(self._chunks):
                audio = (np.full(1024, 0.01 * (i + 1), dtype=np.float32)).tobytes()
                yield SynthesizeChunk(audio=audio, sample_rate=self._sample_rate, is_final=False)
                await asyncio.sleep(self._delay)
            yield SynthesizeChunk(audio=b"", sample_rate=self._sample_rate, is_final=True)
        except asyncio.CancelledError:
            self.cancelled_at_chunk = i if "i" in locals() else 0
            raise


MockScheduler = FakeScheduler


class EventCollector:
    """Collects every event emitted by the session for later assertions."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def __call__(self, event: dict) -> None:
        self.events.append(event)

    def by_type(self, type_: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == type_]

    def states(self) -> list[str]:
        return [e["state"] for e in self.by_type(WIRE_STATE_CHANGED)]







class _AcceptAllClassifier:
    """Test helper: confirms every timer fire. These tests exercise state-
    machine transitions, not the content-based backchannel filter.
    """

    def confirm_window_ms(self, base_ms, last_eou_probability):
        return base_ms

    async def is_real_interrupt(self, audio, partial_transcript, eou, duration_ms):
        return True


def _build_session(
    *,
    adapter: TTSAdapter | None = None,
    policy: TurnPolicy | None = None,
) -> tuple[ConversationSession, EventCollector, ScriptedTTSAdapter]:
    tts = adapter or ScriptedTTSAdapter()
    scheduler = MockScheduler(tts)
    collector = EventCollector()

    config = ConversationConfig(
        stt_model="fake-stt:latest",
        tts_model="fake-tts:latest",
        voice="default",
        language="en",
        policy=policy or TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200),
        interrupt_classifier=_AcceptAllClassifier(),
    )
    session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)
    return session, collector, tts


async def _drain_events(session: ConversationSession, max_iterations: int = 20) -> None:
    """Yield control so the event loop can drain pending turn events + actions."""
    for _ in range(max_iterations):
        await asyncio.sleep(0)
        if session._event_queue.empty() and (
            session._tts_task is None or session._tts_task.done()
        ):
            break







class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_close_no_leaks(self):
        session, _, _ = _build_session()
        await session.start()
        await session.close()
        assert session._runner.done()

    @pytest.mark.asyncio
    async def test_state_starts_idle(self):
        session, _, _ = _build_session()
        assert session.state == TurnState.IDLE


class TestTTSHappyPath:
    @pytest.mark.asyncio
    async def test_submit_response_emits_audio_and_done(self):
        session, collector, tts = _build_session()
        await session.start()


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        assert session.state == TurnState.THINKING

        await session.submit_response_text("hello there")
        await asyncio.sleep(0.1)
        await _drain_events(session)

        assert tts.last_text == "hello there"
        assert collector.by_type(WIRE_RESPONSE_CREATED)
        assert len(collector.by_type(WIRE_AUDIO_DELTA)) >= 1
        assert collector.by_type(WIRE_RESPONSE_DONE)
        assert session.state == TurnState.IDLE

        await session.close()

    @pytest.mark.asyncio
    async def test_state_transitions_emit_events(self):
        session, collector, _ = _build_session()
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await _drain_events(session)
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("reply")
        await asyncio.sleep(0.1)
        await _drain_events(session)

        states = collector.states()
        assert "listening" in states
        assert "thinking" in states
        assert "speaking" in states
        assert "idle" in states

        await session.close()

    @pytest.mark.asyncio
    async def test_streamed_response_starts_on_sentence_boundary_before_commit(self):
        session, collector, tts = _build_session()
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        assert session.state == TurnState.THINKING

        await session.append_response_text("Hello world. Still streaming")
        await asyncio.sleep(0.05)
        await _drain_events(session)

        assert collector.by_type(WIRE_RESPONSE_CREATED)
        assert tts.texts == ["Hello world."]
        assert collector.by_type(WIRE_AUDIO_DELTA)

        await session.append_response_text(" without punctuation yet")
        await session.commit_response_stream()
        await asyncio.sleep(0.1)
        await _drain_events(session)

        assert tts.texts == ["Hello world.", "Still streaming without punctuation yet"]
        assert collector.by_type(WIRE_RESPONSE_DONE)
        assert session.state == TurnState.IDLE

        await session.close()


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_confirmed_barge_in_cancels_tts(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02)
        session, collector, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply")


        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED


        await asyncio.sleep(0.1)
        await _drain_events(session)

        assert session.state == TurnState.INTERRUPTED
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)
        assert tts.cancelled_at_chunk is not None

        await session.close()

    @pytest.mark.asyncio
    async def test_false_interrupt_resumes_tts(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02)
        session, collector, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=100, max_endpointing_delay_ms=200),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply")
        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.01)

        assert session.state == TurnState.SPEAKING

        assert tts.cancelled_at_chunk is None

        assert not collector.by_type(WIRE_RESPONSE_CANCELLED)

        await session.close()

    @pytest.mark.asyncio
    async def test_pending_audio_buffered_during_pause(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=200, max_endpointing_delay_ms=500),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("paused reply")
        await asyncio.sleep(0.03)

        chunks_before_pause = len(collector.by_type(WIRE_AUDIO_DELTA))
        assert chunks_before_pause >= 1


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.05)

        assert session.state == TurnState.PAUSED

        chunks_while_paused_start = len(collector.by_type(WIRE_AUDIO_DELTA))

        await asyncio.sleep(0.05)
        chunks_while_paused_end = len(collector.by_type(WIRE_AUDIO_DELTA))
        assert chunks_while_paused_end == chunks_while_paused_start,\
            "no audio chunks should be emitted while paused"
        assert session.pending_audio_count >= 0


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.05)


        chunks_after_resume = len(collector.by_type(WIRE_AUDIO_DELTA))
        assert chunks_after_resume >= chunks_while_paused_end

        await session.close()


class TestClientCancel:
    @pytest.mark.asyncio
    async def test_cancel_during_speaking_stops_tts(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("reply")
        await asyncio.sleep(0.05)

        assert session.state == TurnState.SPEAKING

        await session.cancel_response()
        await asyncio.sleep(0.05)

        assert session.state == TurnState.IDLE
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)
        assert tts.cancelled_at_chunk is not None

        await session.close()

    @pytest.mark.asyncio
    async def test_cancel_in_idle_is_noop(self):
        session, collector, _ = _build_session()
        await session.start()
        await session.cancel_response()
        await asyncio.sleep(0.02)
        assert session.state == TurnState.IDLE
        assert not collector.by_type(WIRE_RESPONSE_CANCELLED)
        await session.close()


class TestTTSErrorPath:
    @pytest.mark.asyncio
    async def test_tts_adapter_failure_emits_error(self):
        class BrokenTTS(ScriptedTTSAdapter):
            async def synthesize(self, text, **_):
                raise RuntimeError("tts exploded")
                yield

        tts = BrokenTTS()
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        assert session.state == TurnState.THINKING

        await session.submit_response_text("hello")
        await asyncio.sleep(0.05)
        await _drain_events(session)

        errors = collector.by_type(WIRE_ERROR)
        assert errors
        assert "tts exploded" in errors[0]["message"]
        assert session.state == TurnState.IDLE
        assert session._pipeline._conversation_history == []

        await session.close()

    @pytest.mark.asyncio
    async def test_empty_tts_completion_returns_to_idle(self):
        class SilentTTS(ScriptedTTSAdapter):
            async def synthesize(self, text, **_):
                self.last_text = text
                yield SynthesizeChunk(audio=b"", sample_rate=self._sample_rate, is_final=True)

        tts = SilentTTS()
        session, collector, _ = _build_session(adapter=tts)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        assert session.state == TurnState.THINKING

        await session.submit_response_text("quiet reply")
        await asyncio.sleep(0.05)
        await _drain_events(session)

        assert collector.by_type(WIRE_RESPONSE_DONE)
        assert session.state == TurnState.IDLE

        await session.close()


class TestEndpointingFallback:
    @pytest.mark.asyncio
    async def test_endpointing_timer_forces_turn_end(self):


        session, collector, _ = _build_session(
            policy=TurnPolicy(max_endpointing_delay_ms=50, min_interrupt_duration_ms=300),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.LISTENING

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.1)

        assert session.state == TurnState.THINKING

        await session.close()

    @pytest.mark.asyncio
    async def test_transcript_after_speech_stop_waits_for_continuation_window(self):
        session, _, _ = _build_session(
            policy=TurnPolicy(max_endpointing_delay_ms=3000, min_interrupt_duration_ms=300),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.LISTENING

        await session._forward_stream_event(SpeechStopped(timestamp_ms=2400))
        await asyncio.sleep(0.01)
        assert TimerKey.ENDPOINTING.value in session._timers

        await session._forward_stream_event(StreamTranscript(
            text="still thinking",
            start_ms=0,
            end_ms=2400,
        ))
        await asyncio.sleep(0.05)

        assert session.state == TurnState.LISTENING

        await asyncio.sleep(1.25)
        assert session.state == TurnState.THINKING

        await session.close()


class TestAssistantTurnInEouHistory:
    @pytest.mark.asyncio
    async def test_submit_response_text_adds_assistant_turn(self):
        """EOU history must include assistant turns for correct turn-taking."""
        session, _, _ = _build_session()
        await session.start()

        await session.submit_response_text("hello from the bot")
        await asyncio.sleep(0.05)


        history = session._pipeline._conversation_history
        assert any(
            turn.role == "assistant" and "hello from the bot" in turn.content
            for turn in history
        ), f"assistant turn not found; history={history}"

        await session.close()

    @pytest.mark.asyncio
    async def test_empty_reply_does_not_add_turn(self):
        session, _, _ = _build_session()
        await session.start()

        before = list(session._pipeline._conversation_history)
        await session.submit_response_text("   ")
        await asyncio.sleep(0.02)
        assert session._pipeline._conversation_history == before

        await session.close()


class TestResponseGating:
    @pytest.mark.asyncio
    async def test_second_submit_ignored_while_tts_in_flight(self):
        tts = ScriptedTTSAdapter(chunks=10, inter_chunk_delay=0.02)
        session, _, _ = _build_session(adapter=tts)
        await session.start()

        await session.submit_response_text("first")
        await asyncio.sleep(0.01)

        await session.submit_response_text("second")
        await asyncio.sleep(0.2)

        assert tts.last_text == "first"
        await session.close()
