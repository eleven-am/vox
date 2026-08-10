"""Integration tests for ConversationSession.

These tests don't use real STT/TTS models. Instead, we:
  * mock the scheduler to hand out a fake TTS adapter that emits scripted chunks
  * drive state transitions by injecting events into the session's event_queue
    directly OR by calling the session's public API methods
  * observe outgoing client events via the on_event callback
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

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
    ERROR_CODE_COMMAND_INVALID,
    ERROR_CODE_RESPONSE_REJECTED_TURN_STATE,
    ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH,
    WIRE_AUDIO_CLEAR,
    WIRE_AUDIO_DELTA,
    WIRE_AUDIO_RESUME,
    WIRE_AUDIO_SUSPEND,
    WIRE_ERROR,
    WIRE_INTERRUPTION_DETECTED,
    WIRE_INTERRUPTION_FALSE_POSITIVE,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_COMMITTED,
    WIRE_RESPONSE_CREATED,
    WIRE_RESPONSE_DONE,
    WIRE_STATE_CHANGED,
    WIRE_TRANSCRIPT_DELTA,
    WIRE_TRANSCRIPT_DONE,
    WIRE_TURN_EOU_PREDICTED,
    AppendResult,
    ConversationConfig,
    ConversationSession,
)
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.speech_context.types import SpeechContext
from vox.streaming.types import SpeechStarted, SpeechStopped, StreamTranscript
from vox.streaming.vad import SpeechSegment


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


class HangingReleaseScheduler(FakeScheduler):
    def __init__(self, adapter: TTSAdapter) -> None:
        super().__init__(adapter)
        self.release_started = asyncio.Event()
        self.release_gate = asyncio.Event()

    @asynccontextmanager
    async def acquire(self, name: str):
        try:
            yield self._default_adapter
        finally:
            self.release_started.set()
            while not self.release_gate.is_set():
                try:
                    await self.release_gate.wait()
                except asyncio.CancelledError:
                    continue


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


class _RejectAllClassifier(_AcceptAllClassifier):
    async def is_real_interrupt(self, audio, partial_transcript, eou, duration_ms, sample_rate):
        return False


def _build_session(
    *,
    adapter: TTSAdapter | None = None,
    policy: TurnPolicy | None = None,
    audio_preprocessor=None,
    pace_response_done_to_audio: bool = False,
    interrupt_classifier=None,
    speech_context: bool = False,
    speech_context_service=None,
) -> tuple[ConversationSession, EventCollector, ScriptedTTSAdapter]:
    tts = adapter or ScriptedTTSAdapter()
    scheduler = MockScheduler(tts)
    collector = EventCollector()

    config = ConversationConfig(
        stt_model="fake-stt:latest",
        tts_model="fake-tts:latest",
        voice="default",
        language="en",
        policy=policy or TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200, aec_warmup_ms=0),
        interrupt_classifier=interrupt_classifier or _AcceptAllClassifier(),
        audio_preprocessor=audio_preprocessor,
        pace_response_done_to_audio=pace_response_done_to_audio,
        speech_context=speech_context,
    )
    session = ConversationSession(
        scheduler=scheduler,
        config=config,
        on_event=collector,
        speech_context_service=speech_context_service,
    )
    return session, collector, tts


async def _drain_events(session: ConversationSession, max_iterations: int = 20) -> None:
    """Yield control so the event loop can drain pending turn events + actions."""
    for _ in range(max_iterations):
        await asyncio.sleep(0)
        if session._event_queue.empty() and (session._tts_task is None or session._tts_task.done()):
            break


@pytest.mark.asyncio
async def test_pending_continuation_context_is_reanalyzed_as_one_timeline(caplog):
    class RecordingContextService:
        def __init__(self) -> None:
            self.chunks = ()
            self.timeline_offset_ms = None

        async def analyze_chunks(self, chunks, *, timeline_offset_ms=0):
            self.chunks = tuple(chunks)
            self.timeline_offset_ms = timeline_offset_ms
            return SpeechContext(status="failed", unavailable=("speaker", "sounds"))

    context_service = RecordingContextService()
    collector = EventCollector()
    session = ConversationSession(
        scheduler=MockScheduler(ScriptedTTSAdapter()),
        config=ConversationConfig(
            stt_model="fake-stt:latest",
            tts_model="fake-tts:latest",
            voice="default",
            language="en",
            speech_context=True,
            policy=TurnPolicy(aec_warmup_ms=0),
            interrupt_classifier=_AcceptAllClassifier(),
        ),
        on_event=collector,
        speech_context_service=context_service,
    )
    original_context = SpeechContext(
        status="failed",
        unavailable=("speaker", "sounds"),
    )
    session._transcript_finalizer.remember(
        StreamTranscript(
            text="first phrase",
            start_ms=100,
            end_ms=200,
            audio=np.full(1_600, 0.1, dtype=np.float32),
            speech_context=original_context,
        )
    )
    session._transcript_finalizer.remember(
        StreamTranscript(
            text="second phrase",
            start_ms=600,
            end_ms=800,
            audio=np.full(3_200, 0.2, dtype=np.float32),
            speech_context=original_context,
        )
    )

    with caplog.at_level(logging.INFO, logger="vox.conversation.session"):
        await session._emit_pending_transcript_done()

    assert [chunk.offset_ms for chunk in context_service.chunks] == [100, 600]
    assert context_service.timeline_offset_ms == 100
    assert (
        "conversation speech context emitted chunks=2 audio_ms=300 status=failed "
        "emotions=0 vocal=0 sounds=0 unavailable=2"
    ) in caplog.text
    assert collector.by_type(WIRE_TRANSCRIPT_DONE) == [
        {
            "type": WIRE_TRANSCRIPT_DONE,
            "transcript": "first phrase second phrase",
            "language": "en",
            "start_ms": 100,
            "end_ms": 800,
            "speech_context": {
                "schema_version": 2,
                "status": "failed",
                "unavailable": ["speaker", "sounds"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_speech_context_timeline_starts_with_context_only_vad_audio():
    class RecordingContextService:
        def __init__(self) -> None:
            self.chunks = ()
            self.timeline_offset_ms = None

        async def analyze_chunks(self, chunks, *, timeline_offset_ms=0):
            self.chunks = tuple(chunks)
            self.timeline_offset_ms = timeline_offset_ms
            return SpeechContext(status="failed", unavailable=("speaker", "sounds"))

    context_service = RecordingContextService()
    session = ConversationSession(
        scheduler=MockScheduler(ScriptedTTSAdapter()),
        config=ConversationConfig(
            stt_model="fake-stt:latest",
            tts_model="fake-tts:latest",
            voice="default",
            language="en",
            speech_context=True,
            policy=TurnPolicy(aec_warmup_ms=0),
            interrupt_classifier=_AcceptAllClassifier(),
        ),
        on_event=EventCollector(),
        speech_context_service=context_service,
    )
    session._transcript_finalizer.remember_turn_audio(
        utterance_id=1,
        audio=np.full(4_800, 0.3, dtype=np.float32),
        start_ms=100,
    )
    session._transcript_finalizer.remember_turn_audio(
        utterance_id=2,
        audio=np.full(3_200, 0.2, dtype=np.float32),
        start_ms=600,
    )
    session._transcript_finalizer.remember(
        StreamTranscript(
            text="actual words",
            start_ms=600,
            end_ms=800,
            audio=np.full(3_200, 0.2, dtype=np.float32),
            utterance_id=2,
        )
    )

    await session._emit_pending_transcript_done()

    assert [chunk.offset_ms for chunk in context_service.chunks] == [100, 600]
    assert context_service.timeline_offset_ms == 100


@pytest.mark.asyncio
async def test_single_segment_speech_context_is_analyzed_at_turn_finalization():
    class RecordingContextService:
        def __init__(self) -> None:
            self.chunks = ()

        async def analyze_chunks(self, chunks, *, timeline_offset_ms=0):
            self.chunks = tuple(chunks)
            return SpeechContext(status="failed", unavailable=("speaker", "sounds"))

    context_service = RecordingContextService()
    session, _, _ = _build_session(
        speech_context=True,
        speech_context_service=context_service,
    )
    session._transcript_finalizer.remember_turn_audio(
        utterance_id=1,
        audio=np.full(3_200, 0.2, dtype=np.float32),
        start_ms=100,
    )
    session._transcript_finalizer.remember(
        StreamTranscript(
            text="actual words",
            start_ms=100,
            end_ms=300,
            utterance_id=1,
        )
    )

    await session._emit_pending_transcript_done()

    assert [(chunk.offset_ms, chunk.duration_ms) for chunk in context_service.chunks] == [(100, 200)]


@pytest.mark.asyncio
async def test_single_segment_reuses_context_computed_concurrently_with_stt():
    class UnexpectedContextService:
        async def analyze_chunks(self, chunks, *, timeline_offset_ms=0):
            raise AssertionError("single-segment context must not be analyzed twice")

    session, collector, _ = _build_session(
        speech_context=True,
        speech_context_service=UnexpectedContextService(),
    )
    context = SpeechContext(status="failed", unavailable=("speaker", "sounds"))
    session._transcript_finalizer.remember_turn_audio(
        utterance_id=1,
        audio=np.full(3_200, 0.2, dtype=np.float32),
        start_ms=100,
    )
    session._transcript_finalizer.remember(
        StreamTranscript(
            text="actual words",
            start_ms=100,
            end_ms=300,
            utterance_id=1,
            speech_context=context,
        )
    )

    await session._emit_pending_transcript_done()

    assert collector.by_type(WIRE_TRANSCRIPT_DONE)[0]["speech_context"]["status"] == "failed"


class TestInterruptionEventContracts:
    @pytest.mark.asyncio
    async def test_response_lifecycle_event_shapes_are_owned_by_session_helpers(self):
        session, collector, _ = _build_session()
        stream = session._response_lifecycle.start_stream(
            output=session._default_response_output,
            generation_id="gen-1",
        )

        await session._emit_response_created(stream)
        await session._emit_response_committed(stream)
        record = session._response_lifecycle.terminalize(stream, "done")
        assert record is not None
        await session._emit_response_done(record)
        await session._emit_response_cancelled(record)
        await session._emit_error("boom", code=ERROR_CODE_COMMAND_INVALID)

        assert collector.events == [
            {
                "type": WIRE_RESPONSE_CREATED,
                "response_id": "resp_1",
                "generation_id": "gen-1",
                "output": session._default_response_output.to_payload(),
            },
            {"type": WIRE_RESPONSE_COMMITTED, "response_id": "resp_1", "generation_id": "gen-1"},
            {"type": WIRE_RESPONSE_DONE, "response_id": "resp_1", "generation_id": "gen-1"},
            {"type": WIRE_RESPONSE_CANCELLED, "response_id": "resp_1", "generation_id": "gen-1"},
            {
                "type": WIRE_ERROR,
                "message": "boom",
                "code": ERROR_CODE_COMMAND_INVALID,
                "recoverable": True,
            },
        ]

    @pytest.mark.asyncio
    async def test_response_lifecycle_events_omit_generation_when_unstamped(self):
        session, collector, _ = _build_session()
        stream = session._response_lifecycle.start_stream(
            output=session._default_response_output,
        )

        await session._emit_response_created(stream)
        record = session._response_lifecycle.terminalize(stream, "done")
        assert record is not None
        await session._emit_response_done(record)

        assert collector.events == [
            {
                "type": WIRE_RESPONSE_CREATED,
                "response_id": "resp_1",
                "output": session._default_response_output.to_payload(),
            },
            {"type": WIRE_RESPONSE_DONE, "response_id": "resp_1"},
        ]

    @pytest.mark.asyncio
    async def test_detected_interrupt_event_shape_is_owned_by_session_helper(self, caplog):
        session, collector, _ = _build_session()
        session._response_lifecycle.start_stream(output=session._default_response_output)
        caplog.set_level(logging.INFO, logger="vox.conversation.session")

        await session._emit_interruption_detected(
            vad_active_ms=420,
            partial_transcript="private interruption text",
            reason="partial_keyword",
        )

        assert collector.events == [
            {
                "type": WIRE_INTERRUPTION_DETECTED,
                "response_id": "resp_1",
                "vad_active_ms": 420,
                "partial_transcript": "private interruption text",
                "reason": "partial_keyword",
            }
        ]
        assert "private interruption text" not in caplog.text
        assert "transcript_chars=25" in caplog.text

    @pytest.mark.asyncio
    async def test_false_positive_interrupt_event_shape_is_owned_by_session_helper(self):
        session, collector, _ = _build_session()
        session._response_lifecycle.start_stream(output=session._default_response_output)

        await session._emit_interruption_false_positive(
            vad_active_ms=120,
            partial_transcript="mhmm",
            reason="backchannel",
        )

        assert collector.events == [
            {
                "type": WIRE_INTERRUPTION_FALSE_POSITIVE,
                "response_id": "resp_1",
                "vad_active_ms": 120,
                "partial_transcript": "mhmm",
                "reason": "backchannel",
            }
        ]


class TestResponseAdmission:
    @pytest.mark.asyncio
    async def test_late_response_while_listening_never_synthesizes_audio(self):
        session, collector, adapter = _build_session()
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await _drain_events(session)
        assert session.state == TurnState.LISTENING

        await session.start_response_stream()
        await session.append_response_text("stale response")
        await session.commit_response_stream()
        await _drain_events(session)

        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_RESPONSE_CREATED)
        assert not collector.by_type(WIRE_AUDIO_DELTA)
        assert not collector.by_type(WIRE_RESPONSE_DONE)
        assert collector.by_type(WIRE_ERROR) == [
            {
                "type": WIRE_ERROR,
                "message": "response rejected: turn state is listening",
                "code": ERROR_CODE_RESPONSE_REJECTED_TURN_STATE,
                "recoverable": True,
            }
        ]
        assert adapter.texts == []

        await session.close()

    @pytest.mark.asyncio
    async def test_tts_audio_start_must_be_accepted_before_audio_can_emit(self):
        session, collector, _ = _build_session()
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await _drain_events(session)
        assert session.state == TurnState.LISTENING

        with pytest.raises(asyncio.CancelledError):
            await session._notify_tts_audio_started()

        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_AUDIO_DELTA)
        await session.close()

    @pytest.mark.asyncio
    async def test_raw_vad_flag_does_not_drop_stream_after_first_audio(self):
        session, _, _ = _build_session(adapter=ScriptedTTSAdapter(chunks=1))
        await session.start()
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        assert session.state == TurnState.THINKING

        response_id = (await session.start_response_stream()).response_id
        assert response_id is not None
        assert (
            await session.append_response_text(
                "The first phrase.",
                expected_response_id=response_id,
            )
            is AppendResult.ACCEPTED
        )

        # A raw VAD edge can race with the first playout frame when the
        # assistant's own audio reaches the microphone. The turn-state event,
        # not this early flag, owns whether the response is interrupted.
        session._input_speech_active = True
        await asyncio.sleep(0.05)

        assert session.state == TurnState.SPEAKING
        assert (
            await session.append_response_text(
                " The response continues.",
                expected_response_id=response_id,
            )
            is AppendResult.ACCEPTED
        )

        session._input_speech_active = False
        assert await session.commit_response_stream(expected_response_id=response_id) is AppendResult.ACCEPTED
        await session.close()

    @pytest.mark.asyncio
    async def test_response_is_rejected_during_vad_state_transition(self):
        session, collector, adapter = _build_session()
        await session.start()
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        assert session.state == TurnState.THINKING
        session._input_speech_active = True

        await session.submit_response_text("raced with speech start")
        await _drain_events(session)

        assert session.state == TurnState.THINKING
        assert not collector.by_type(WIRE_RESPONSE_CREATED)
        assert not collector.by_type(WIRE_AUDIO_DELTA)
        assert collector.by_type(WIRE_ERROR) == [
            {
                "type": WIRE_ERROR,
                "message": "response rejected: user speech is active",
                "code": ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH,
                "recoverable": True,
            }
        ]
        assert adapter.texts == []

        await session.close()

    @pytest.mark.asyncio
    async def test_response_retry_remains_blocked_after_provisional_evidence_timeout(self):
        session, collector, _ = _build_session(interrupt_classifier=_RejectAllClassifier())
        await session.start()
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        assert session.state == TurnState.THINKING

        await session._forward_stream_event(SpeechStarted(timestamp_ms=2_000, utterance_id=7))
        await _drain_events(session)
        assert session._input_speech_active is True

        rejected = await session.start_response_stream()
        assert rejected.response_id is None
        assert rejected.rejection is not None
        assert rejected.rejection.code == ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH
        await session._evaluate_interrupt_candidate()
        await _drain_events(session)

        assert collector.by_type(WIRE_INTERRUPTION_FALSE_POSITIVE) == []
        candidate = session._interrupt_detector.current()
        assert candidate is not None
        assert candidate.provisional_rejection_reason == "no_transcript_timeout"
        assert session._input_speech_active is True

        retry = await session.start_response_stream()
        assert retry.response_id is None
        assert retry.rejection is not None
        assert retry.rejection.code == ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH
        assert collector.by_type(WIRE_ERROR) == []

        await session.close()

    @pytest.mark.asyncio
    async def test_rejected_short_candidate_cannot_admit_response_for_unfinished_turn(self):
        session, collector, _ = _build_session(interrupt_classifier=_RejectAllClassifier())
        await session.start()
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        session._last_eou_probability = 0.000028

        await session._forward_stream_event(SpeechStarted(timestamp_ms=2_000, utterance_id=7))
        await _drain_events(session)
        await session._evaluate_interrupt_candidate()
        await _drain_events(session)

        assert collector.by_type(WIRE_INTERRUPTION_FALSE_POSITIVE) == []
        candidate = session._interrupt_detector.current()
        assert candidate is not None
        assert candidate.provisional_rejection_reason == "no_transcript_timeout"
        assert session._input_speech_active is True

        result = await session.start_response_stream()
        assert result.response_id is None
        assert result.rejection is not None
        assert result.rejection.code == ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH
        assert not collector.by_type(WIRE_RESPONSE_CREATED)

        await session.close()

    @pytest.mark.asyncio
    async def test_provisional_timeout_keeps_audio_until_empty_final_rejects(self):
        session, _, _ = _build_session(
            interrupt_classifier=_RejectAllClassifier(),
            speech_context=True,
        )
        await session.start()
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)

        await session._forward_stream_event(SpeechStarted(timestamp_ms=2_000, utterance_id=7))
        await session._evaluate_interrupt_candidate()
        await session._forward_stream_event(
            SpeechStopped(
                timestamp_ms=2_300,
                expects_transcript=True,
                utterance_id=7,
                start_ms=2_000,
                end_ms=2_300,
                audio=np.full(4_800, 0.2, dtype=np.float32),
            )
        )

        assert 7 in session._transcript_finalizer.pending_turn_audio
        await session._forward_stream_event(
            StreamTranscript(
                text="",
                start_ms=2_000,
                end_ms=2_300,
                audio_duration_ms=300,
                utterance_id=7,
            )
        )
        assert session._transcript_finalizer.pending_turn_audio == {}
        await session.close()

    @pytest.mark.asyncio
    async def test_rejected_final_remains_authoritative_until_vad_stops(self):
        session, collector, _ = _build_session(interrupt_classifier=_RejectAllClassifier())
        await session.start()
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)

        await session._forward_stream_event(SpeechStarted(timestamp_ms=2_000, utterance_id=7))
        await _drain_events(session)
        await session._forward_stream_event(
            StreamTranscript(
                text="",
                is_partial=False,
                start_ms=2_000,
                end_ms=2_220,
                audio_duration_ms=220,
                utterance_id=7,
            )
        )
        await _drain_events(session)

        false_positives = collector.by_type(WIRE_INTERRUPTION_FALSE_POSITIVE)
        assert false_positives[-1]["reason"] == "empty_final"
        assert session._input_speech_active is True

        result = await session.start_response_stream()
        assert result.response_id is not None
        assert result.context.candidate_status.value == "rejected"
        candidate = session._interrupt_detector.current()
        assert candidate is not None
        assert candidate.status.value == "rejected"

        await session._forward_stream_event(SpeechStopped(timestamp_ms=2_300, expects_transcript=False, utterance_id=7))
        await _drain_events(session)
        assert session._interrupt_detector.current() is None

        await session.close()

    @pytest.mark.asyncio
    async def test_rejected_candidate_does_not_admit_a_later_utterance(self):
        session, _, _ = _build_session(interrupt_classifier=_RejectAllClassifier())
        await session.start()
        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)

        await session._forward_stream_event(SpeechStarted(timestamp_ms=2_000, utterance_id=7))
        await _drain_events(session)
        await session._evaluate_interrupt_candidate()
        await _drain_events(session)

        await session._forward_stream_event(SpeechStopped(timestamp_ms=2_300, expects_transcript=False, utterance_id=7))
        await _drain_events(session)
        await session._forward_stream_event(SpeechStarted(timestamp_ms=3_000, utterance_id=8))
        await _drain_events(session)

        result = await session.start_response_stream()
        assert result.response_id is None
        assert result.rejection is not None
        assert result.rejection.code == ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH
        assert result.context.candidate_id is not None
        assert result.context.candidate_status.value == "pending"

        await session.close()


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
    async def test_partial_transcript_emits_delta_event(self):
        session, collector, _ = _build_session()
        await session.start()

        await session._forward_stream_event(
            StreamTranscript(
                text="hello there",
                is_partial=True,
                start_ms=0,
                end_ms=700,
            )
        )
        await _drain_events(session)

        deltas = collector.by_type(WIRE_TRANSCRIPT_DELTA)
        assert len(deltas) == 1
        assert deltas[0]["delta"] == "hello there"
        assert deltas[0]["start_ms"] == 0
        assert deltas[0]["end_ms"] == 700

        await session.close()

    @pytest.mark.asyncio
    async def test_empty_partial_transcript_not_emitted(self):
        session, collector, _ = _build_session()
        await session.start()

        await session._forward_stream_event(StreamTranscript(text="  ", is_partial=True, start_ms=0, end_ms=700))
        await _drain_events(session)

        assert not collector.by_type(WIRE_TRANSCRIPT_DELTA)
        await session.close()

    @pytest.mark.asyncio
    async def test_final_transcript_logs_metadata_without_client_text(self, caplog):
        session, collector, _ = _build_session()
        await session.start()

        caplog.set_level(logging.INFO, logger="vox.conversation.session")

        await session._forward_stream_event(
            StreamTranscript(
                text="client facing final text",
                eou_probability=0.9,
                start_ms=100,
                end_ms=900,
            )
        )
        await _drain_events(session)

        completed = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(completed) == 1
        assert completed[0]["transcript"] == "client facing final text"
        assert any(
            record.name == "vox.conversation.session"
            and "conversation final transcript emitted" in record.message
            and "chars=24" in record.message
            and "start_ms=100" in record.message
            and "end_ms=900" in record.message
            for record in caplog.records
        )
        assert "client facing final text" not in caplog.text

        await session.close()

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
    async def test_slow_transcription_is_not_recorded_as_a_thinking_pause(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=3000,
                min_endpointing_delay_ms=350,
                dynamic_endpointing=True,
                aec_warmup_ms=0,
            ),
        )
        await session.start()

        try:
            await session._forward_stream_event(SpeechStarted(timestamp_ms=0, utterance_id=1))
            await _drain_events(session)
            await session._forward_stream_event(SpeechStopped(timestamp_ms=700, utterance_id=1))
            await _drain_events(session)
            session._last_speech_stopped_at = time.monotonic() - 2.0
            await session._forward_stream_event(
                StreamTranscript(
                    text="I have not finished",
                    eou_probability=0.0,
                    start_ms=0,
                    end_ms=700,
                    utterance_id=1,
                )
            )
            await _drain_events(session)

            predicted = collector.by_type(WIRE_TURN_EOU_PREDICTED)
            assert predicted[-1]["delay_ms"] == 1200
            assert session._endpoint_pause_history.values() == ()

            session._last_speech_stopped_at = time.monotonic() - 0.9
            await session._forward_stream_event(SpeechStarted(timestamp_ms=1600, utterance_id=2))
            await _drain_events(session)

            observed_pause_ms = session._endpoint_pause_history.values()
            assert len(observed_pause_ms) == 1
            assert 850 <= observed_pause_ms[0] <= 1000
        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_incomplete_clause_survives_endpoint_expiry_racing_resumed_speech(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=3000,
                min_endpointing_delay_ms=400,
                dynamic_endpointing=False,
                aec_warmup_ms=0,
            ),
        )
        await session.start()

        await session._forward_stream_event(SpeechStarted(timestamp_ms=0, utterance_id=1))
        await _drain_events(session)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=1_800, utterance_id=1))
        await _drain_events(session)
        await session._forward_stream_event(
            StreamTranscript(
                text="I am calling out because I wanna",
                eou_probability=0.000028,
                start_ms=0,
                end_ms=1_800,
                utterance_id=1,
            )
        )
        await _drain_events(session)

        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

        # VAD has already observed the continuation, but the speech-start event
        # is still queued behind an endpoint expiry from the previous pause.
        await session._forward_stream_event(SpeechStarted(timestamp_ms=4_790, utterance_id=2))
        await session._process_turn_event(
            TurnEvent(
                type=TurnEventType.TIMER_ELAPSED,
                payload={"key": TimerKey.ENDPOINTING.value},
            )
        )
        await _drain_events(session)

        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

        await session._forward_stream_event(SpeechStopped(timestamp_ms=5_700, utterance_id=2))
        await _drain_events(session)
        await session._forward_stream_event(
            StreamTranscript(
                text="finish what I was saying",
                eou_probability=0.95,
                start_ms=4_790,
                end_ms=5_700,
                utterance_id=2,
            )
        )
        await _drain_events(session)

        await session._cancel_timer(TimerKey.ENDPOINTING.value)
        await session._process_turn_event(
            TurnEvent(
                type=TurnEventType.TIMER_ELAPSED,
                payload={"key": TimerKey.ENDPOINTING.value},
            )
        )

        completed = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(completed) == 1
        assert completed[0]["transcript"] == ("I am calling out because I wanna finish what I was saying")
        assert session.state == TurnState.THINKING

        await session.close()

    @pytest.mark.asyncio
    async def test_multiple_thinking_pauses_emit_one_turn_and_offer_one_response(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=3000,
                min_endpointing_delay_ms=400,
                dynamic_endpointing=False,
                aec_warmup_ms=0,
            ),
        )
        await session.start()

        clauses = (
            ("I have been thinking", 0.01),
            ("about how this should", 0.04),
            ("work when I pause", 0.95),
        )
        timestamp_ms = 0
        for index, (text, eou_probability) in enumerate(clauses, start=1):
            await session._forward_stream_event(SpeechStarted(timestamp_ms=timestamp_ms, utterance_id=index))
            await _drain_events(session)
            timestamp_ms += 700
            await session._forward_stream_event(SpeechStopped(timestamp_ms=timestamp_ms, utterance_id=index))
            await _drain_events(session)
            await session._forward_stream_event(
                StreamTranscript(
                    text=text,
                    eou_probability=eou_probability,
                    start_ms=timestamp_ms - 700,
                    end_ms=timestamp_ms,
                    utterance_id=index,
                )
            )
            await _drain_events(session)
            assert not collector.by_type(WIRE_TRANSCRIPT_DONE)
            timestamp_ms += 500

        await session._cancel_timer(TimerKey.ENDPOINTING.value)
        await session._process_turn_event(
            TurnEvent(
                type=TurnEventType.TIMER_ELAPSED,
                payload={"key": TimerKey.ENDPOINTING.value},
            )
        )

        completed = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(completed) == 1
        assert completed[0]["transcript"] == ("I have been thinking about how this should work when I pause")

        first = await session.start_response_stream(generation_id="generation-1")
        second = await session.start_response_stream(generation_id="generation-1")

        assert first.response_id is not None
        assert second.response_id == first.response_id
        assert len(collector.by_type(WIRE_RESPONSE_CREATED)) == 1

        await session.close()

    @pytest.mark.asyncio
    async def test_replaced_endpoint_timer_cannot_commit_pending_continuation(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=3000,
                min_endpointing_delay_ms=400,
                dynamic_endpointing=False,
                aec_warmup_ms=0,
            ),
        )
        await session.start()

        await session._forward_stream_event(SpeechStarted(timestamp_ms=0, utterance_id=1))
        await _drain_events(session)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=700, utterance_id=1))
        await _drain_events(session)
        await session._forward_stream_event(
            StreamTranscript(
                text="I am still",
                eou_probability=0.01,
                start_ms=0,
                end_ms=700,
                utterance_id=1,
            )
        )
        await _drain_events(session)

        stale = await session._timer_registry.start(TimerKey.ENDPOINTING.value, 10_000)
        replacement = await session._timer_registry.start(TimerKey.ENDPOINTING.value, 10_000)
        assert replacement is not stale

        await session._process_turn_event(
            TurnEvent(
                type=TurnEventType.TIMER_ELAPSED,
                payload={
                    "key": TimerKey.ENDPOINTING.value,
                    "_timer_lease": stale,
                },
            )
        )

        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert session._timer_registry.has_active(TimerKey.ENDPOINTING.value)

        await session.close()

    @pytest.mark.asyncio
    async def test_close_discards_held_transcript_without_late_emission(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=3000,
                min_endpointing_delay_ms=400,
                dynamic_endpointing=False,
                aec_warmup_ms=0,
            ),
        )
        await session.start()

        await session._forward_stream_event(SpeechStarted(timestamp_ms=0, utterance_id=1))
        await _drain_events(session)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=700, utterance_id=1))
        await _drain_events(session)
        await session._forward_stream_event(
            StreamTranscript(
                text="I have not finished",
                eou_probability=0.01,
                start_ms=0,
                end_ms=700,
                utterance_id=1,
            )
        )
        await _drain_events(session)
        assert session._transcript_finalizer.pending is not None

        await session.close()
        await asyncio.sleep(0)

        assert session._transcript_finalizer.pending is None
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

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
    async def test_response_done_is_emitted_after_idle_state_transition(self):
        session, collector, _ = _build_session(adapter=ScriptedTTSAdapter(chunks=1))
        await session.start()

        await session.submit_response_text("ordered response")
        await asyncio.sleep(0.05)
        await _drain_events(session)

        idle_index = next(
            index
            for index, event in enumerate(collector.events)
            if event.get("type") == WIRE_STATE_CHANGED and event.get("state") == TurnState.IDLE.value
        )
        done_index = next(
            index for index, event in enumerate(collector.events) if event.get("type") == WIRE_RESPONSE_DONE
        )
        assert idle_index < done_index

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
    async def test_empty_stt_from_real_pipeline_suspends_then_resumes_without_cancelling(self):
        from vox.core.adapter import STTAdapter
        from vox.core.types import TranscribeResult

        class EmptySTTAdapter(STTAdapter):
            def info(self) -> AdapterInfo:
                return AdapterInfo(
                    name="empty-stt",
                    type=ModelType.STT,
                    architectures=("test",),
                    default_sample_rate=16_000,
                    supported_formats=(ModelFormat.ONNX,),
                )

            def load(self, *_args, **_kwargs) -> None: ...
            def unload(self) -> None: ...

            @property
            def is_loaded(self) -> bool:
                return True

            def transcribe(self, audio, **_kwargs) -> TranscribeResult:
                return TranscribeResult(
                    text="",
                    language="en",
                    duration_ms=int(len(audio) * 1000 / 16_000),
                )

        class ModelScheduler:
            def __init__(self, stt, tts) -> None:
                self.stt = stt
                self.tts = tts

            @asynccontextmanager
            async def acquire(self, name: str):
                yield self.stt if name == "fake-stt:latest" else self.tts

        class StartThenEmptyFinalVad:
            def __init__(self) -> None:
                self.calls = 0

            def append(self, audio: np.ndarray):
                self.calls += 1
                if self.calls == 1:
                    return SpeechStarted(timestamp_ms=100, utterance_id=1), None
                return (
                    SpeechStopped(timestamp_ms=300, utterance_id=1),
                    SpeechSegment(
                        audio=audio,
                        start_ms=100,
                        end_ms=300,
                        utterance_id=1,
                    ),
                )

            def reset(self) -> None:
                self.calls = 0

        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.02)
        collector = EventCollector()
        session = ConversationSession(
            scheduler=ModelScheduler(EmptySTTAdapter(), tts),
            config=ConversationConfig(
                stt_model="fake-stt:latest",
                tts_model="fake-tts:latest",
                voice="default",
                language="en",
                policy=TurnPolicy(aec_warmup_ms=0),
            ),
            on_event=collector,
        )
        session._pipeline._vad = StartThenEmptyFinalVad()
        await session.start()
        await session.submit_response_text("long reply")
        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING

        pcm = np.full(3200, 1200, dtype=np.int16).tobytes()
        await session.ingest_audio(pcm, sample_rate=16_000)
        await asyncio.sleep(0.01)
        assert session.state == TurnState.SPEAKING
        assert session._audio_output.paused is True
        assert collector.by_type(WIRE_AUDIO_SUSPEND)

        await session.ingest_audio(pcm, sample_rate=16_000)
        await asyncio.sleep(0.02)
        await _drain_events(session)

        rejected = collector.by_type(WIRE_INTERRUPTION_FALSE_POSITIVE)
        assert rejected and rejected[-1]["reason"] == "empty_final"
        assert session.state == TurnState.SPEAKING
        assert session._audio_output.paused is False
        assert collector.by_type(WIRE_AUDIO_RESUME)
        assert not collector.by_type(WIRE_AUDIO_CLEAR)
        assert not collector.by_type(WIRE_RESPONSE_CANCELLED)
        assert tts.cancelled_at_chunk is None

        await session.close()

    @pytest.mark.asyncio
    async def test_empty_listening_final_recovers_serially_and_emits_idle_state(self):
        session, collector, _ = _build_session()
        await session.start()

        await session._forward_stream_event(SpeechStarted(timestamp_ms=100, utterance_id=1))
        await session._forward_stream_event(
            SpeechStopped(
                timestamp_ms=300,
                expects_transcript=True,
                utterance_id=1,
                start_ms=100,
                end_ms=300,
            )
        )
        await session._forward_stream_event(
            StreamTranscript(
                text="",
                start_ms=100,
                end_ms=300,
                audio_duration_ms=200,
                utterance_id=1,
            )
        )
        await asyncio.sleep(0.02)
        await _drain_events(session)

        assert session.state == TurnState.IDLE
        assert not session._has_active_timer(TimerKey.ENDPOINTING.value)
        assert collector.states()[-2:] == [TurnState.LISTENING.value, TurnState.IDLE.value]
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert not collector.by_type(WIRE_RESPONSE_CREATED)

        await session.close()

    @pytest.mark.asyncio
    async def test_stale_empty_final_without_candidate_does_not_end_active_response(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.02)
        session, collector, _ = _build_session(adapter=tts)
        await session.start()
        await session.submit_response_text("long reply")
        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING

        await session._forward_stream_event(
            StreamTranscript(
                text="",
                start_ms=100,
                end_ms=300,
                audio_duration_ms=200,
                utterance_id=99,
            )
        )
        await asyncio.sleep(0.02)
        await _drain_events(session)

        assert session.state == TurnState.SPEAKING
        assert session.active_response_id is not None
        assert tts.cancelled_at_chunk is None
        assert not collector.by_type(WIRE_AUDIO_CLEAR)
        assert not collector.by_type(WIRE_RESPONSE_CANCELLED)

        await session.close()

    @pytest.mark.asyncio
    async def test_speech_stop_keeps_playout_suspended_until_empty_final_resumes_it(self):
        tts = ScriptedTTSAdapter(chunks=40, inter_chunk_delay=0.02)
        session, collector, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=500, max_endpointing_delay_ms=200, aec_warmup_ms=0),
        )
        await session.start()
        await session.submit_response_text("long reply")
        await asyncio.sleep(0.05)

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000, utterance_id=1))
        await asyncio.sleep(0.01)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=1250, expects_transcript=True, utterance_id=1))
        await asyncio.sleep(0.01)

        assert session.state == TurnState.SPEAKING
        assert session._audio_output.paused is True
        assert collector.by_type(WIRE_AUDIO_SUSPEND)
        assert not collector.by_type(WIRE_AUDIO_CLEAR)

        await session._forward_stream_event(StreamTranscript(text="", start_ms=1000, end_ms=1250, utterance_id=1))
        await asyncio.sleep(0.01)
        await _drain_events(session)

        assert session.state == TurnState.SPEAKING
        assert session._audio_output.paused is False
        assert collector.by_type(WIRE_INTERRUPTION_FALSE_POSITIVE)[-1]["reason"] == "empty_final"
        assert collector.by_type(WIRE_AUDIO_RESUME)
        assert not collector.by_type(WIRE_AUDIO_CLEAR)
        await session.close()

    @pytest.mark.asyncio
    async def test_confirmed_barge_in_cancels_tts(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02)
        session, collector, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200, aec_warmup_ms=0),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply")

        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000, utterance_id=1))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.SPEAKING
        assert not collector.by_type(WIRE_AUDIO_CLEAR)

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
        await asyncio.sleep(0.01)
        await _drain_events(session)

        assert session.state == TurnState.INTERRUPTED
        assert collector.by_type(WIRE_INTERRUPTION_DETECTED)
        audio_clear = collector.by_type(WIRE_AUDIO_CLEAR)
        assert audio_clear
        assert audio_clear[-1]["response_id"] == collector.by_type(WIRE_RESPONSE_CREATED)[-1]["response_id"]
        assert collector.by_type(WIRE_RESPONSE_CANCELLED)
        assert tts.cancelled_at_chunk is not None

        await session.close()

    @pytest.mark.asyncio
    async def test_confirmed_interrupt_emits_clear_and_cancelled_before_slow_tts_teardown(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02)
        scheduler = HangingReleaseScheduler(tts)
        collector = EventCollector()
        config = ConversationConfig(
            stt_model="fake-stt:latest",
            tts_model="fake-tts:latest",
            voice="default",
            language="en",
            policy=TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200, aec_warmup_ms=0),
            interrupt_classifier=_AcceptAllClassifier(),
        )
        session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply")
        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING
        tts_task = session._tts_task
        assert tts_task is not None

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000, utterance_id=1))
        await asyncio.sleep(0.01)
        started = time.monotonic()
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
        while time.monotonic() - started < 2.0:
            if collector.by_type(WIRE_AUDIO_CLEAR) and collector.by_type(WIRE_RESPONSE_CANCELLED):
                break
            await asyncio.sleep(0.005)
        elapsed = time.monotonic() - started

        assert elapsed < 2.0
        response_id = collector.by_type(WIRE_RESPONSE_CREATED)[-1]["response_id"]
        audio_clear = collector.by_type(WIRE_AUDIO_CLEAR)
        cancelled = collector.by_type(WIRE_RESPONSE_CANCELLED)
        assert audio_clear
        assert cancelled
        assert audio_clear[-1]["response_id"] == response_id
        assert cancelled[-1]["response_id"] == response_id
        assert session.state == TurnState.INTERRUPTED
        assert not tts_task.done()

        await asyncio.wait_for(scheduler.release_started.wait(), timeout=2.0)
        assert not tts_task.done()

        scheduler.release_gate.set()
        await asyncio.sleep(0.05)
        await _drain_events(session)
        await session.close()

    @pytest.mark.asyncio
    async def test_false_interrupt_leaves_tts_running(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02)
        session, collector, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=100, max_endpointing_delay_ms=200, aec_warmup_ms=0),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("long reply")
        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.SPEAKING

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.01)

        assert session.state == TurnState.SPEAKING

        assert tts.cancelled_at_chunk is None

        assert collector.by_type(WIRE_INTERRUPTION_FALSE_POSITIVE) == []
        audio_clear = collector.by_type(WIRE_AUDIO_CLEAR)
        assert audio_clear == []
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
            policy=TurnPolicy(min_interrupt_duration_ms=500, max_endpointing_delay_ms=200, aec_warmup_ms=0),
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

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000, utterance_id=1))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.SPEAKING

        await session._forward_stream_event(
            StreamTranscript(
                text="please stop talking",
                is_partial=True,
                utterance_id=1,
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
            policy=TurnPolicy(min_interrupt_duration_ms=50, max_endpointing_delay_ms=200, aec_warmup_ms=0),
        )
        await session.start()

        await session.append_response_text("Hello world. Second sentence is not heard yet")
        for _ in range(50):
            await asyncio.sleep(0.01)
            stream = session._response_stream
            if tts.texts == ["Hello world."] and stream is not None and stream.heard_parts:
                break
        assert tts.texts == ["Hello world."]

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
            policy=TurnPolicy(min_interrupt_duration_ms=500, max_endpointing_delay_ms=200, aec_warmup_ms=0),
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

        assert collector.by_type(WIRE_TRANSCRIPT_DELTA)
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
            policy=TurnPolicy(partial_interrupts=False, aec_warmup_ms=0),
            interrupt_classifier=HeuristicInterruptClassifier(),
        )
        session = ConversationSession(scheduler=scheduler, config=config, on_event=collector)
        assert session._wants_partials is False
        assert session._partial_service is None
        assert session._speech_session is None

    @pytest.mark.asyncio
    async def test_audio_generated_during_candidate_is_buffered_and_resumed_in_order(self):
        tts = ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.01)
        session, collector, _ = _build_session(
            adapter=tts,
            policy=TurnPolicy(min_interrupt_duration_ms=200, max_endpointing_delay_ms=500, aec_warmup_ms=0),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(session)
        await session.submit_response_text("paused reply")
        await asyncio.sleep(0.03)

        chunks_before_candidate = len(collector.by_type(WIRE_AUDIO_DELTA))
        assert chunks_before_candidate >= 1

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000, utterance_id=1))
        await asyncio.sleep(0.05)

        assert session.state == TurnState.SPEAKING
        assert session._audio_output.paused is True
        assert collector.by_type(WIRE_AUDIO_SUSPEND)

        chunks_during_candidate_start = len(collector.by_type(WIRE_AUDIO_DELTA))

        await asyncio.sleep(0.05)
        chunks_during_candidate_end = len(collector.by_type(WIRE_AUDIO_DELTA))
        assert chunks_during_candidate_end == chunks_during_candidate_start
        assert session.pending_audio_count > 0

        await session._forward_stream_event(
            SpeechStopped(timestamp_ms=1150, expects_transcript=False, utterance_id=1)
        )
        await asyncio.sleep(0.05)

        chunks_after_stop = len(collector.by_type(WIRE_AUDIO_DELTA))
        assert chunks_after_stop > chunks_during_candidate_end
        assert session.pending_audio_count == 0
        assert session._audio_output.paused is False
        assert collector.by_type(WIRE_AUDIO_RESUME)

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
    async def test_cancel_during_thinking_flushes_output_like_speaking_cancel(self):
        thinking_session, thinking_events, _ = _build_session()
        await thinking_session.start()
        await thinking_session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(thinking_session)
        assert thinking_session.state == TurnState.THINKING

        thinking_response_id = (await thinking_session.start_response_stream()).response_id
        assert thinking_response_id is not None
        thinking_before = len(thinking_events.events)
        await thinking_session.cancel_response()
        thinking_tail = [e["type"] for e in thinking_events.events[thinking_before:]]

        speaking_session, speaking_events, _ = _build_session(
            adapter=ScriptedTTSAdapter(chunks=20, inter_chunk_delay=0.02),
        )
        await speaking_session.start()
        await speaking_session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await _drain_events(speaking_session)
        await speaking_session.submit_response_text("reply")
        await asyncio.sleep(0.05)
        assert speaking_session.state == TurnState.SPEAKING
        speaking_before = len(speaking_events.events)
        await speaking_session.cancel_response()
        speaking_tail = [e["type"] for e in speaking_events.events[speaking_before:] if e["type"] != WIRE_AUDIO_DELTA]

        assert thinking_tail == [WIRE_AUDIO_CLEAR, WIRE_RESPONSE_CANCELLED, WIRE_STATE_CHANGED]
        assert speaking_tail == thinking_tail
        assert thinking_session.state == TurnState.IDLE
        assert thinking_session.pending_audio_count == 0
        assert thinking_events.by_type(WIRE_AUDIO_CLEAR)[-1]["response_id"] == thinking_response_id
        assert thinking_events.by_type(WIRE_RESPONSE_CANCELLED)[-1]["response_id"] == thinking_response_id

        await thinking_session.close()
        await speaking_session.close()

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
            policy=TurnPolicy(max_endpointing_delay_ms=50, min_interrupt_duration_ms=300, aec_warmup_ms=0),
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
            policy=TurnPolicy(max_endpointing_delay_ms=3000, min_interrupt_duration_ms=300, aec_warmup_ms=0),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.LISTENING

        await session._forward_stream_event(SpeechStopped(timestamp_ms=2400))
        await asyncio.sleep(0.01)
        assert session._timer_registry.has_active(TimerKey.ENDPOINTING.value)

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
    async def test_endpointing_waits_for_pending_final_transcript(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=50,
                min_interrupt_duration_ms=300,
                dynamic_endpointing=False,
                aec_warmup_ms=0,
            ),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=1200, expects_transcript=True))

        await asyncio.sleep(0.08)
        await _drain_events(session)

        assert session.state == TurnState.LISTENING
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

        await session._forward_stream_event(
            StreamTranscript(
                text="still working on the transcript",
                eou_probability=0.9,
                start_ms=0,
                end_ms=1200,
            )
        )
        await asyncio.sleep(0.08)
        await _drain_events(session)

        events = collector.by_type(WIRE_TRANSCRIPT_DONE)
        assert len(events) == 1
        assert events[0]["transcript"] == "still working on the transcript"
        assert session.state == TurnState.THINKING

        await session.close()

    @pytest.mark.asyncio
    async def test_endpointing_does_not_wait_when_no_transcript_is_expected(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(max_endpointing_delay_ms=50, min_interrupt_duration_ms=300, aec_warmup_ms=0),
        )
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        await session._forward_stream_event(SpeechStopped(timestamp_ms=1200, expects_transcript=False))
        await asyncio.sleep(0.08)
        await _drain_events(session)

        assert session.state == TurnState.THINKING
        assert not collector.by_type(WIRE_TRANSCRIPT_DONE)

        await session.close()

    @pytest.mark.asyncio
    async def test_deferred_transcripts_coalesce_before_commit(self):
        session, collector, _ = _build_session(
            policy=TurnPolicy(
                max_endpointing_delay_ms=100,
                min_interrupt_duration_ms=300,
                dynamic_endpointing=False,
                aec_warmup_ms=0,
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
    async def test_policy_vad_min_silence_reaches_vad_config(self):
        session, _, _ = _build_session(
            policy=TurnPolicy(vad_min_silence_ms=550, aec_warmup_ms=0),
        )
        assert session._pipeline._vad.config.min_silence_duration_ms == 550


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
