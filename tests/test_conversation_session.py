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

from tests.fakes import FakeScheduler
from vox.conversation import (
    HeuristicInterruptClassifier,
    TimerKey,
    TurnEvent,
    TurnEventType,
    TurnPolicy,
    TurnState,
)
from vox.conversation.session import (
    WIRE_AUDIO_CLEAR,
    WIRE_AUDIO_DELTA,
    WIRE_ERROR,
    WIRE_INTERRUPTION_DETECTED,
    WIRE_INTERRUPTION_FALSE_POSITIVE,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_CREATED,
    WIRE_RESPONSE_DONE,
    WIRE_STATE_CHANGED,
    WIRE_TRANSCRIPT_DONE,
    WIRE_TURN_EOU_PREDICTED,
    ConversationConfig,
    ConversationSession,
)
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.streaming.types import SpeechStarted, SpeechStopped, StreamTranscript


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
            name="scripted-tts",
            type=ModelType.TTS,
            architectures=("scripted",),
            default_sample_rate=self._sample_rate,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *_a, **_k): ...
    def unload(self): ...

    @property
    def is_loaded(self):
        return True

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

    def wants_short_circuit(self):
        return False

    def should_short_circuit(self, partial_transcript):
        return False

    async def is_real_interrupt(self, audio, partial_transcript, eou, duration_ms, sample_rate):
        return True


def _build_session(
    *,
    adapter: TTSAdapter | None = None,
    policy: TurnPolicy | None = None,
    audio_preprocessor=None,
    pace_response_done_to_audio: bool = False,
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
        audio_preprocessor=audio_preprocessor,
        pace_response_done_to_audio=pace_response_done_to_audio,
    )
    session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)
    return session, collector, tts


async def _drain_events(session: ConversationSession, max_iterations: int = 20) -> None:
    """Yield control so the event loop can drain pending turn events + actions."""
    for _ in range(max_iterations):
        await asyncio.sleep(0)
        if session._event_queue.empty() and (session._tts_task is None or session._tts_task.done()):
            break


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_close_no_leaks(self):
        session, _, _ = _build_session()
        await session.start()
        await session.close()

    @pytest.mark.asyncio
    async def test_final_transcript_emits_eou_prediction_event(self):
        session, collector, _ = _build_session()
        await session.start()

        await session._forward_stream_event(
            StreamTranscript(
                text="that is all",
                eou_probability=0.9,
                start_ms=100,
                end_ms=600,
            )
        )
        await _drain_events(session)

        events = collector.by_type(WIRE_TURN_EOU_PREDICTED)
        assert events
        assert events[-1]["probability"] == pytest.approx(0.9)
        assert events[-1]["threshold"] == pytest.approx(0.5)
        assert events[-1]["decision"] == "complete"
        assert events[-1]["action"] == "commit"
        assert events[-1]["turn_detector"] == "livekit"
        assert events[-1]["start_ms"] == 100
        assert events[-1]["end_ms"] == 600

        await session.close()
        assert session._runner.done()

    @pytest.mark.asyncio
    async def test_final_transcripts_coalesce_during_endpointing_window(self):
        session, collector, _ = _build_session()
        await session.start()

        await session._forward_stream_event(SpeechStarted(timestamp_ms=0))
        await _drain_events(session)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=1000))
        await _drain_events(session)

        await session._forward_stream_event(
            StreamTranscript(
                text="first part",
                eou_probability=0.9,
                start_ms=0,
                end_ms=1000,
            )
        )
        await _drain_events(session)

        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert collector.by_type(WIRE_TURN_EOU_PREDICTED)[-1]["action"] == "wait"
        assert session.state == TurnState.LISTENING

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1300))
        await _drain_events(session)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=2200))
        await _drain_events(session)
        await session._forward_stream_event(
            StreamTranscript(
                text="second part",
                eou_probability=0.9,
                start_ms=1300,
                end_ms=2200,
            )
        )
        await _drain_events(session)

        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

        await session._cancel_timer(TimerKey.ENDPOINTING.value)
        await session._event_queue.put(
            TurnEvent(
                type=TurnEventType.TIMER_ELAPSED,
                payload={"key": TimerKey.ENDPOINTING.value},
            )
        )
        await _drain_events(session)

        completed = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(completed) == 1
        assert completed[0]["transcript"] == "first part second part"
        assert completed[0]["start_ms"] == 0
        assert completed[0]["end_ms"] == 2200
        assert session.state == TurnState.THINKING

        await session.close()

    @pytest.mark.asyncio
    async def test_final_transcript_is_not_rewritten_by_partials(self):
        session, collector, _ = _build_session()
        await session.start()

        assert session._speech_session is not None
        session._speech_session.start_speech()
        session._speech_session.update_partial(
            8700,
            [
                "Um,",
                "I",
                "am",
                "taking",
                "long",
                "pauses,",
                "just",
                "to",
                "see",
                "if",
                "you",
                "know",
                "the",
                "system",
                "works",
                "with",
                "everything",
                "I'm",
                "saying.",
            ],
        )
        session._speech_session.stop_speech()

        await session._forward_stream_event(
            StreamTranscript(
                text="I am See If you know the system works. Same.",
                eou_probability=0.9,
                start_ms=2356,
                end_ms=11100,
                audio_duration_ms=8744,
            )
        )
        await _drain_events(session)

        completed = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(completed) == 1
        assert completed[0]["transcript"] == "I am See If you know the system works. Same."

        await session.close()

    @pytest.mark.asyncio
    async def test_final_transcript_kept_when_partials_are_not_materially_richer(self):
        session, collector, _ = _build_session()
        await session.start()

        assert session._speech_session is not None
        session._speech_session.start_speech()
        session._speech_session.update_partial(500, ["Yeah."])
        session._speech_session.stop_speech()

        await session._forward_stream_event(
            StreamTranscript(
                text="Yeah.",
                eou_probability=0.9,
                start_ms=0,
                end_ms=500,
                audio_duration_ms=500,
            )
        )
        await _drain_events(session)

        completed = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(completed) == 1
        assert completed[0]["transcript"] == "Yeah."

        await session.close()

    @pytest.mark.asyncio
    async def test_state_starts_idle(self):
        session, _, _ = _build_session()
        assert session.state == TurnState.IDLE

    @pytest.mark.asyncio
    async def test_ingest_audio_applies_preprocessor_before_pipeline(self):
        seen: dict[str, object] = {}

        def preprocessor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
            seen["pre_sample_rate"] = sample_rate
            seen["pre_audio"] = audio.copy()
            return np.zeros_like(audio)

        session, _, _ = _build_session(audio_preprocessor=preprocessor)

        async def fake_process_audio(audio: np.ndarray):
            seen["pipeline_audio"] = audio.copy()
            if False:
                yield None

        session._pipeline.process_audio = fake_process_audio  # type: ignore[method-assign]

        await session.start()
        pcm = (np.ones(160, dtype=np.int16) * 1000).tobytes()
        await session.ingest_audio(pcm, 16_000)

        assert seen["pre_sample_rate"] == 16_000
        assert np.any(seen["pre_audio"])
        assert np.all(seen["pipeline_audio"] == 0)

        await session.close()

    @pytest.mark.asyncio
    async def test_speech_start_chunk_is_kept_for_partials(self):
        session, _, _ = _build_session()

        async def fake_process_audio(audio: np.ndarray):
            yield SpeechStarted(timestamp_ms=0)

        session._pipeline.process_audio = fake_process_audio  # type: ignore[method-assign]

        await session.start()
        pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
        await session.ingest_audio(pcm, 16_000)

        assert session._speech_session is not None
        assert session._speech_session.get_buffer_length() == 1600

        await session.close()


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
    async def test_audio_deltas_include_response_id_and_sequence(self):
        session, collector, _ = _build_session()
        await session.start()

        await session.submit_response_text("hello there")
        await asyncio.sleep(0.1)
        await _drain_events(session)

        created = collector.by_type(WIRE_RESPONSE_CREATED)
        deltas = collector.by_type(WIRE_AUDIO_DELTA)
        assert created
        assert deltas
        response_id = created[0]["response_id"]
        assert response_id
        assert {d["response_id"] for d in deltas} == {response_id}
        assert [d["sequence"] for d in deltas] == sorted(d["sequence"] for d in deltas)

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
    async def test_paced_response_done_stays_speaking_until_audio_playout(self):
        class LongChunkTTS(ScriptedTTSAdapter):
            async def synthesize(self, text: str, **_kwargs):
                self.last_text = text
                self.texts.append(text)
                audio = np.full(4_800, 0.01, dtype=np.float32).tobytes()
                yield SynthesizeChunk(audio=audio, sample_rate=24_000, is_final=False)
                yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)

        session, collector, _ = _build_session(
            adapter=LongChunkTTS(),
            pace_response_done_to_audio=True,
        )
        await session.start()

        await session.submit_response_text("paced reply")
        await asyncio.sleep(0.05)
        await _drain_events(session)

        assert collector.by_type(WIRE_AUDIO_DELTA)
        assert not collector.by_type(WIRE_RESPONSE_DONE)
        assert session.state == TurnState.SPEAKING

        await asyncio.sleep(0.2)
        await _drain_events(session)

        assert collector.by_type(WIRE_RESPONSE_DONE)
        assert session.state == TurnState.IDLE

        await session.close()

    @pytest.mark.asyncio
    async def test_response_done_waits_for_output_playout_callback(self):
        released = asyncio.Event()
        waiter_entered = asyncio.Event()

        class OneChunkTTS(ScriptedTTSAdapter):
            async def synthesize(self, text: str, **_kwargs):
                self.last_text = text
                self.texts.append(text)
                audio = np.full(1_024, 0.01, dtype=np.float32).tobytes()
                yield SynthesizeChunk(audio=audio, sample_rate=24_000, is_final=False)
                yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)

        async def wait_for_output_playout() -> None:
            waiter_entered.set()
            await released.wait()

        session, collector, _ = _build_session(adapter=OneChunkTTS())
        session._config.wait_for_output_playout = wait_for_output_playout
        await session.start()

        await session.submit_response_text("wait for output")
        await asyncio.wait_for(waiter_entered.wait(), timeout=1.0)
        await _drain_events(session)

        assert collector.by_type(WIRE_AUDIO_DELTA)
        assert not collector.by_type(WIRE_RESPONSE_DONE)
        assert session.state == TurnState.SPEAKING

        released.set()
        await _drain_events(session)

        assert collector.by_type(WIRE_RESPONSE_DONE)
        assert session.state == TurnState.IDLE

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

        session._latest_partial = StreamTranscript(text="I need", is_partial=True)
        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED

        await asyncio.sleep(0.1)
        await _drain_events(session)

        assert session.state == TurnState.INTERRUPTED
        assert collector.by_type(WIRE_INTERRUPTION_DETECTED)
        assert collector.by_type(WIRE_AUDIO_CLEAR)
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

        assert collector.by_type(WIRE_INTERRUPTION_FALSE_POSITIVE) == []
        assert collector.by_type(WIRE_AUDIO_CLEAR)
        assert not collector.by_type(WIRE_RESPONSE_CANCELLED)

        await session.close()

    @pytest.mark.asyncio
    async def test_keyword_partial_short_circuits_before_timer(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02)
        scheduler = MockScheduler(tts)
        collector = EventCollector()
        config = ConversationConfig(
            stt_model="fake-stt:latest",
            tts_model="fake-tts:latest",
            voice="default",
            language="en",
            policy=TurnPolicy(min_interrupt_duration_ms=500, max_endpointing_delay_ms=200),
            interrupt_classifier=HeuristicInterruptClassifier(
                interrupt_keywords=frozenset({"stop"}),
            ),
        )
        session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)

        await session.start()
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply")
        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING

        await session._event_queue.put(
            TurnEvent(
                type=TurnEventType.SPEECH_STARTED,
                payload={"confirm_window_ms": 500},
            )
        )
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED

        await session._forward_stream_event(
            StreamTranscript(
                text="please stop talking",
                is_partial=True,
            )
        )

        await asyncio.sleep(0.01)
        await _drain_events(session)

        assert session.state == TurnState.INTERRUPTED
        assert tts.cancelled_at_chunk is not None
        assert collector.by_type(WIRE_INTERRUPTION_DETECTED)
        assert collector.by_type(WIRE_AUDIO_CLEAR)
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)

        await session.close()

    @pytest.mark.asyncio
    async def test_interrupted_response_only_adds_heard_text_to_eou_history(self):
        tts = ScriptedTTSAdapter(chunks=1, inter_chunk_delay=0.01)
        session, _, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200),
        )
        await session.start()

        await session.append_response_text("Hello world. Second sentence is not heard yet")
        for _ in range(50):
            await asyncio.sleep(0.01)
            if tts.texts == ["Hello world."]:
                break
        assert tts.texts == ["Hello world."]

        session._latest_partial = StreamTranscript(text="I need", is_partial=True)
        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.1)
        await _drain_events(session)

        history = session._pipeline._conversation_history
        assistant_turns = [turn.content for turn in history if turn.role == "assistant"]
        assert assistant_turns == ["Hello world."]

        await session.close()

    @pytest.mark.asyncio
    async def test_keyword_partial_outside_paused_does_not_short_circuit(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02)
        scheduler = MockScheduler(tts)
        collector = EventCollector()
        config = ConversationConfig(
            stt_model="fake-stt:latest",
            tts_model="fake-tts:latest",
            voice="default",
            language="en",
            policy=TurnPolicy(min_interrupt_duration_ms=500, max_endpointing_delay_ms=200),
            interrupt_classifier=HeuristicInterruptClassifier(
                interrupt_keywords=frozenset({"stop"}),
            ),
        )
        session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)
        await session.start()

        await session._forward_stream_event(
            StreamTranscript(
                text="please stop talking",
                is_partial=True,
            )
        )

        assert session._latest_partial is not None
        assert session._latest_partial.text == "please stop talking"
        assert session.state == TurnState.IDLE
        assert not collector.by_type(WIRE_RESPONSE_CANCELLED)

        await session.close()

    @pytest.mark.asyncio
    async def test_partial_interrupt_policy_enables_partials_without_keywords(self):
        tts = ScriptedTTSAdapter()
        scheduler = MockScheduler(tts)
        collector = EventCollector()
        config = ConversationConfig(
            stt_model="fake-stt:latest",
            tts_model="fake-tts:latest",
            voice="default",
            language="en",
            interrupt_classifier=HeuristicInterruptClassifier(),
        )
        session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)
        assert session._wants_partials is True
        assert session._partial_service is not None
        assert session._speech_session is not None

    @pytest.mark.asyncio
    async def test_partial_interrupt_policy_can_disable_partials_without_keywords(self):
        tts = ScriptedTTSAdapter()
        scheduler = MockScheduler(tts)
        collector = EventCollector()
        config = ConversationConfig(
            stt_model="fake-stt:latest",
            tts_model="fake-tts:latest",
            voice="default",
            language="en",
            policy=TurnPolicy(partial_interrupts=False),
            interrupt_classifier=HeuristicInterruptClassifier(),
        )
        session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)
        assert session._wants_partials is False
        assert session._partial_service is None
        assert session._speech_session is None

    @pytest.mark.asyncio
    async def test_audio_generated_during_pause_is_buffered_until_resume(self):
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
        assert chunks_while_paused_end == chunks_while_paused_start, "no audio chunks should be emitted while paused"
        assert session.pending_audio_count > 0

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.05)

        chunks_after_resume = len(collector.by_type(WIRE_AUDIO_DELTA))
        assert chunks_after_resume > chunks_while_paused_end
        assert session.pending_audio_count == 0

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
        session, collector, _ = _build_session(
            policy=TurnPolicy(max_endpointing_delay_ms=3000, min_interrupt_duration_ms=300),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.LISTENING

        await session._forward_stream_event(SpeechStopped(timestamp_ms=2400))
        await asyncio.sleep(0.01)
        assert TimerKey.ENDPOINTING.value in session._timers

        await session._forward_stream_event(
            StreamTranscript(
                text="still thinking",
                start_ms=0,
                end_ms=2400,
            )
        )
        await asyncio.sleep(0.05)

        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

        await asyncio.sleep(1.25)
        await _drain_events(session)
        assert session.state == TurnState.THINKING
        events = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(events) == 1
        assert events[0]["transcript"] == "still thinking"

        await session.close()

    @pytest.mark.asyncio
    async def test_deferred_transcripts_coalesce_before_commit(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=100,
                min_interrupt_duration_ms=300,
                dynamic_endpointing=False,
            ),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=1200))
        await asyncio.sleep(0.01)

        await session._forward_stream_event(
            StreamTranscript(
                text="Can you tell me?",
                eou_probability=0.1,
                start_ms=0,
                end_ms=1200,
            )
        )
        await asyncio.sleep(0.01)
        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

        await session._forward_stream_event(
            StreamTranscript(
                text="What can you tell me?",
                eou_probability=0.1,
                start_ms=0,
                end_ms=1800,
            )
        )
        await asyncio.sleep(0.01)
        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

        await asyncio.sleep(0.12)
        await _drain_events(session)
        events = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(events) == 1
        assert events[0]["transcript"] == "What can you tell me?"

        await session.close()

    @pytest.mark.asyncio
    async def test_dynamic_endpointing_uses_recent_pause_history(self):
        session, _, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=3000,
                min_endpointing_delay_ms=400,
                dynamic_endpointing=True,
            ),
        )
        session._recent_endpoint_pauses_ms = [800, 1000, 1200]
        assert session._transcript_commit_delay_ms() == 1250

        session._recent_endpoint_pauses_ms = [100]
        assert session._transcript_commit_delay_ms() == 1200


class TestAssistantTurnInEouHistory:
    @pytest.mark.asyncio
    async def test_submit_response_text_adds_assistant_turn(self):
        """EOU history must include assistant turns for correct turn-taking."""
        session, _, _ = _build_session()
        await session.start()

        await session.submit_response_text("hello from the bot")
        await asyncio.sleep(0.05)

        history = session._pipeline._conversation_history
        assert any(turn.role == "assistant" and "hello from the bot" in turn.content for turn in history), (
            f"assistant turn not found; history={history}"
        )

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
