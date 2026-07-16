"""Async orchestrator for a single voice conversation.

Wires together:
  * the pure `TurnStateMachine`
  * Vox's existing streaming pipeline (VAD + STT + EOU)
  * a TTS adapter acquired from the scheduler
  * a timer registry (asyncio tasks)
  * pause/clear handling during barge-in confirmation
  * an event emitter for client-facing notifications

Concurrency model
-----------------
One `asyncio.Task` drives the state machine (`_run_loop`); it is the **only**
mutator of output pause state and `_tts_task`. The audio ingest path and the
TTS task push work back through the main loop. When output is paused for an
unconfirmed barge-in, newly generated TTS audio is held until the candidate is
confirmed or rejected. Acoustic echo is evidence owned by the interruption
detector; it never bypasses candidate creation or vetoes later transcript
evidence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np

from vox.conversation import response_stream as response_streams
from vox.conversation import transcripts as transcript_finalization
from vox.conversation.audio_history import ConversationAudioHistory
from vox.conversation.audio_output import ResponseAudioOutput
from vox.conversation.interrupt import (
    HeuristicInterruptClassifier,
    InterruptClassifier,
    looks_like_self_echo,
)
from vox.conversation.interruption_detector import (
    EvidenceBasedInterruptDetector,
    InterruptDetector,
    InterruptionDecision,
    InterruptionDecisionAction,
)
from vox.conversation.profiles import DEFAULT_TURN_PROFILE, resolve_turn_profile
from vox.conversation.response_lifecycle import ConversationResponseLifecycle
from vox.conversation.response_synthesis import synthesize_response_stream
from vox.conversation.speech_guard import AssistantSpeechGuard
from vox.conversation.state_machine import TurnStateMachine
from vox.conversation.timers import ConversationTimerRegistry
from vox.conversation.types import (
    TimerKey,
    TurnAction,
    TurnActionType,
    TurnEvent,
    TurnEventType,
    TurnPolicy,
    TurnState,
)
from vox.core.adapter import TTSAdapter
from vox.core.adapter_acquisition import AdapterTypeMismatchError, acquire_typed_adapter
from vox.core.scheduler import Scheduler
from vox.core.tasks import reap_task
from vox.streaming.annotation import enrich_transcript
from vox.streaming.codecs import StreamResampler, float32_to_pcm16, pcm16_to_float32, resample_audio
from vox.streaming.eou import DEFAULT_TURN_DETECTOR, EOUConfig
from vox.streaming.partials import PartialTranscriptService
from vox.streaming.pipeline import StreamPipeline, StreamPipelineConfig
from vox.streaming.session import SpeechSession
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    SpeechStarted,
    SpeechStopped,
    StreamSessionConfig,
    StreamTranscript,
)
from vox.streaming.vad import VADConfig

logger = logging.getLogger(__name__)
ResponseStream = response_streams.ResponseStream


WIRE_SPEECH_STARTED = "input_audio_buffer.speech_started"
WIRE_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
WIRE_TRANSCRIPT_DONE = transcript_finalization.WIRE_TRANSCRIPT_DONE
WIRE_TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
WIRE_RESPONSE_CREATED = "response.created"
WIRE_AUDIO_DELTA = "response.audio.delta"
WIRE_RESPONSE_DONE = "response.done"
WIRE_RESPONSE_CANCELLED = "response.cancelled"
WIRE_RESPONSE_COMMITTED = "response.committed"
WIRE_AUDIO_CLEAR = "response.audio.clear"
WIRE_INTERRUPTION_DETECTED = "interruption.detected"
WIRE_INTERRUPTION_FALSE_POSITIVE = "interruption.false_positive"
WIRE_TURN_EOU_PREDICTED = transcript_finalization.WIRE_TURN_EOU_PREDICTED
WIRE_STATE_CHANGED = "turn.state_changed"
WIRE_ERROR = "error"
RESPONSE_STREAM_QUEUE_MAX = response_streams.RESPONSE_STREAM_QUEUE_MAX


EventEmitter = Callable[[dict], Awaitable[None]]
AudioPreprocessor = Callable[[np.ndarray, int], np.ndarray | Awaitable[np.ndarray]]


@dataclass
class ConversationConfig:
    stt_model: str
    tts_model: str
    voice: str | None = None
    language: str = "en"
    sample_rate: int = TARGET_SAMPLE_RATE
    policy: TurnPolicy = None  # type: ignore[assignment]
    turn_profile: str = DEFAULT_TURN_PROFILE
    vad_backend: str = "silero"
    turn_detector: str = DEFAULT_TURN_DETECTOR
    include_word_timestamps: bool = False

    interrupt_classifier: InterruptClassifier | None = None
    interrupt_detector: InterruptDetector | None = None
    audio_preprocessor: AudioPreprocessor | None = None
    pace_response_done_to_audio: bool = False
    wait_for_output_playout: Callable[[], Awaitable[None]] | None = None
    output_playout_observed: bool = False

    def __post_init__(self) -> None:
        self.turn_profile, profile_policy = resolve_turn_profile(self.turn_profile)
        if self.policy is None:
            self.policy = profile_policy
        if self.interrupt_classifier is None:
            self.interrupt_classifier = HeuristicInterruptClassifier(
                min_interrupt_words=self.policy.min_interrupt_words,
            )
        if self.interrupt_detector is None:
            self.interrupt_detector = EvidenceBasedInterruptDetector(
                policy=self.policy,
                classifier=self.interrupt_classifier,
            )


TRANSCRIPT_PENDING_STT_RECHECK_MS = 100
_RESPONSE_START_STATES = frozenset({TurnState.IDLE, TurnState.THINKING})


class ConversationSession:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        config: ConversationConfig,
        on_event: EventEmitter,
    ) -> None:
        self._scheduler = scheduler
        self._config = config
        self._on_event = on_event

        self._sm = TurnStateMachine(policy=config.policy)
        if config.interrupt_detector is None:
            raise RuntimeError("conversation interruption detector was not configured")
        self._interrupt_detector = config.interrupt_detector

        self._wants_partials = self._interrupt_detector.wants_partials()

        self._pipeline = StreamPipeline(
            scheduler=scheduler,
            config=StreamPipelineConfig(
                vad_config=VADConfig(
                    backend=config.vad_backend,
                    min_silence_duration_ms=max(0, int(config.policy.vad_min_silence_ms)),
                ),
                eou_config=EOUConfig(model=config.turn_detector),
            ),
        )
        self._stream_session_config = StreamSessionConfig(
            language=config.language,
            sample_rate=TARGET_SAMPLE_RATE,
            model=config.stt_model,
            partials=self._wants_partials,
            include_word_timestamps=config.include_word_timestamps,
        )
        self._pipeline.configure(self._stream_session_config)

        self._speech_session: SpeechSession | None = None
        self._partial_service: PartialTranscriptService | None = None
        if self._wants_partials:
            self._speech_session = SpeechSession()
            self._partial_service = PartialTranscriptService(
                transcribe_async_fn=self._pipeline.transcribe_async,
            )

        self._event_queue: asyncio.Queue[TurnEvent] = asyncio.Queue()
        self._timer_registry = ConversationTimerRegistry(self._on_timer_expired)
        self._interrupt_timer_candidate_id: int | None = None
        self._tts_task: asyncio.Task | None = None
        self._runner: asyncio.Task | None = None
        self._audio_output = ResponseAudioOutput(pace_to_playout=config.pace_response_done_to_audio)
        self._response_lifecycle = ConversationResponseLifecycle()
        self._response_rejection_reported = False
        self._closed: bool = False
        self._client_sample_rate: int = config.sample_rate
        self._input_resampler = StreamResampler(TARGET_SAMPLE_RATE)

        self._speech_guard = AssistantSpeechGuard()
        self._input_speech_active = False
        self._output_playout_started = False

        self._last_eou_probability: float | None = None

        self._last_speech_stopped_at: float | None = None
        self._awaiting_final_transcript: bool = False
        self._awaiting_final_transcript_started_at: float = 0.0
        self._endpoint_pause_history = transcript_finalization.EndpointPauseHistory()
        self._transcript_finalizer = transcript_finalization.PendingTranscriptFinalizer(language=config.language)
        self._endpoint_commit_delay = transcript_finalization.EndpointCommitDelayPolicy.from_turn_policy(config.policy)
        self._audio_history = ConversationAudioHistory()

    async def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(self._run_loop())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._timer_registry.cancel_all()

        await reap_task(self._tts_task)
        await reap_task(self._runner)

        self._pipeline.shutdown()
        self._interrupt_detector.reset()

    async def wait_until_settled(self, *, poll_interval_s: float = 0.01) -> None:
        """Drain timers, queued actions, and any in-flight TTS after client EOF.

        The runner task stays alive for the session lifetime, so "settled" here
        means there is no pending work left that can emit additional events.
        """
        while True:
            timers_active = self._timer_registry.has_any_active()
            tts_active = self._tts_task is not None and not self._tts_task.done()
            queue_busy = not self._event_queue.empty()
            if not timers_active and not tts_active and not queue_busy:
                await asyncio.sleep(0)
                timers_active = self._timer_registry.has_any_active()
                tts_active = self._tts_task is not None and not self._tts_task.done()
                queue_busy = not self._event_queue.empty()
                if not timers_active and not tts_active and not queue_busy:
                    return
            await asyncio.sleep(poll_interval_s)

    async def ingest_audio(self, pcm16: bytes, sample_rate: int | None = None) -> None:
        """Feed a raw PCM16 audio chunk from the client."""
        if self._closed or not pcm16:
            return

        source_rate = sample_rate or self._config.sample_rate
        self._client_sample_rate = source_rate
        audio = pcm16_to_float32(pcm16)
        if source_rate != TARGET_SAMPLE_RATE:
            audio = self._input_resampler.process(audio, source_rate)

        if self._config.audio_preprocessor is not None and audio.size:
            try:
                processed = self._config.audio_preprocessor(audio, TARGET_SAMPLE_RATE)
                if asyncio.iscoroutine(processed):
                    processed = await processed
                audio = np.asarray(processed, dtype=np.float32)
            except Exception as exc:
                logger.exception("audio preprocessor raised")
                await self._emit_error(f"audio preprocessor failed: {exc}")
                return

        if self._is_response_uninterruptible():
            return

        if audio.size:
            self._audio_history.append_mic(audio)

        try:
            async for stream_event in self._pipeline.process_audio(audio):
                await self._forward_stream_event(stream_event)
        except Exception as exc:
            logger.exception("pipeline.process_audio raised")
            self._awaiting_final_transcript = False
            self._awaiting_final_transcript_started_at = 0.0
            await self._emit_error(str(exc))
            return

        if audio.size and self._speech_session is not None:
            self._speech_session.append_audio(audio)

        if (
            self._wants_partials
            and self._partial_service is not None
            and self._speech_session is not None
            and self._speech_session.is_active()
        ):
            try:
                partial = await self._partial_service.generate_partial_async(
                    self._speech_session,
                    self._stream_session_config,
                )
            except Exception:
                logger.exception("partial transcript generation raised")
                partial = None
            if partial is not None:
                await self._forward_stream_event(partial)

    async def submit_response_text(self, text: str, *, allow_interruptions: bool = True) -> None:
        """Agent delivers the reply text; session kicks off TTS."""
        if self._closed:
            return
        await self.append_response_text(text, allow_interruptions=allow_interruptions)
        await self.commit_response_stream()

    async def replace_response_text(
        self,
        text: str,
        *,
        allow_interruptions: bool = True,
    ) -> None:
        """Cancel any active response before starting a fresh text response."""
        if not str(text).strip():
            return
        await self.cancel_response()
        await self.submit_response_text(text, allow_interruptions=allow_interruptions)

    async def start_response_stream(self, *, allow_interruptions: bool = True) -> str | None:
        if self._closed:
            return None
        stream = await self._ensure_response_stream(allow_interruptions=allow_interruptions)
        return stream.response_id if stream is not None else None

    async def append_response_text(
        self,
        text: str,
        *,
        allow_interruptions: bool = True,
        expected_response_id: str | None = None,
    ) -> bool:
        if self._closed or not text or not text.strip():
            return False
        if expected_response_id is None:
            stream = await self._ensure_response_stream(allow_interruptions=allow_interruptions)
        else:
            stream = self._response_stream
            if stream is None or stream.committed or stream.response_id != expected_response_id:
                return False
        if stream is None:
            return False
        await stream.append_text(text)
        return True

    async def commit_response_stream(self, *, expected_response_id: str | None = None) -> bool:
        stream = self._response_stream
        if expected_response_id is not None and (stream is None or stream.response_id != expected_response_id):
            return False
        if stream is None or not stream.mark_committed():
            return False

        await self._emit_response_committed(stream.response_id)
        await stream.enqueue_end()
        return True

    async def cancel_response(self) -> None:
        """Explicit client cancel — orthogonal to barge-in."""
        if self._closed or self._runner is None or self._runner.done():
            return
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        has_active_response = self._response_stream is not None or (
            self._tts_task is not None and not self._tts_task.done()
        )
        await self._event_queue.put(
            TurnEvent(
                type=TurnEventType.CLIENT_CANCEL,
                payload={
                    "has_active_response": has_active_response,
                    "_done": done,
                },
            )
        )
        runner = self._runner
        if runner is not None and runner is not asyncio.current_task():
            completed, _ = await asyncio.wait(
                {done, runner},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if done not in completed:
                return
        await done

    async def _run_loop(self) -> None:
        while not self._closed:
            try:
                event = await self._event_queue.get()
            except asyncio.CancelledError:
                break

            prev_state = self._sm.state
            done = event.payload.get("_done") if isinstance(event.payload, dict) else None
            if self._should_wait_for_final_transcript(event):
                await self._start_timer(
                    TimerKey.ENDPOINTING.value,
                    TRANSCRIPT_PENDING_STT_RECHECK_MS,
                )
                self._resolve_event_future(done)
                continue
            try:
                accepted = self._accepts_turn_event(event)
                actions = self._sm.handle(event) if accepted else []
            except Exception as exc:
                logger.exception("state machine raised on event %s", event)
                self._resolve_event_future(done, exc)
                continue

            for action in actions:
                try:
                    await self._execute(action)
                except Exception:
                    logger.exception("action %s raised", action.type.value)

            committed_user_turn = (
                self._sm.state == TurnState.THINKING
                and prev_state != TurnState.THINKING
                and event.type
                in {
                    TurnEventType.USER_TRANSCRIPT_FINAL,
                    TurnEventType.TIMER_ELAPSED,
                }
            )
            if committed_user_turn:
                await self._emit_pending_transcript_done()
            elif event.type == TurnEventType.CLIENT_CANCEL and self._sm.state == TurnState.IDLE:
                self._transcript_finalizer.clear()

            if self._sm.state != prev_state:
                await self._emit(
                    {
                        "type": WIRE_STATE_CHANGED,
                        "state": self._sm.state.value,
                        "previous_state": prev_state.value,
                    }
                )

            response_stream = event.payload.get("_response_stream")
            if accepted and event.type == TurnEventType.TTS_COMPLETED and isinstance(response_stream, ResponseStream):
                await self._finalize_response_stream(response_stream)

            self._resolve_event_future(done, result=accepted)

    async def _forward_stream_event(self, stream_event) -> None:
        if self._is_response_uninterruptible():
            return
        if isinstance(stream_event, SpeechStarted):
            self._input_speech_active = True
            if self._speech_session is not None:
                self._speech_session.start_speech(stream_event.utterance_id)
            await self._emit(
                {
                    "type": WIRE_SPEECH_STARTED,
                    "timestamp_ms": stream_event.timestamp_ms,
                }
            )

            if self._sm.state in {TurnState.SPEAKING, TurnState.PAUSED} and stream_event.utterance_id <= 0:
                logger.warning("ignoring interruption candidate without utterance identity")
                await self._emit_interruption_false_positive(
                    vad_active_ms=0,
                    partial_transcript=None,
                    reason="missing_candidate_identity",
                )
                return

            if self._sm.state in {TurnState.SPEAKING, TurnState.PAUSED}:
                self._interrupt_detector.begin(
                    utterance_id=stream_event.utterance_id,
                    response_id=self._active_response_id,
                    started_at=time.monotonic(),
                    speech_started_ms=stream_event.timestamp_ms,
                    assistant_text=self._active_assistant_text(),
                )

            if self._sm.state == TurnState.SPEAKING and self._speech_guard.flutter_cooldown_active():
                logger.debug("flutter cooldown active; deferring SPEECH_STARTED state transition")
                return
            confirm_ms = self._interrupt_detector.confirm_window_ms(
                self._config.policy.min_interrupt_duration_ms,
                self._last_eou_probability,
            )
            if self._sm.state == TurnState.SPEAKING:
                confirm_ms = max(
                    confirm_ms,
                    self._config.policy.speaking_interrupt_min_duration_ms,
                )
            await self._event_queue.put(
                TurnEvent(
                    type=TurnEventType.SPEECH_STARTED,
                    timestamp_ms=stream_event.timestamp_ms,
                    payload={
                        "confirm_window_ms": confirm_ms,
                        "defer_output_clear": self._sm.state == TurnState.SPEAKING,
                    },
                )
            )
        elif isinstance(stream_event, SpeechStopped):
            self._input_speech_active = False
            if self._speech_session is not None:
                self._speech_session.stop_speech()
            if stream_event.expects_transcript:
                self._awaiting_final_transcript = True
                self._awaiting_final_transcript_started_at = time.monotonic()
            self._last_speech_stopped_at = time.monotonic()
            interrupt_decision = self._interrupt_detector.mark_speech_stopped(
                utterance_id=stream_event.utterance_id,
                stopped_at=self._last_speech_stopped_at,
                speech_stopped_ms=stream_event.timestamp_ms,
                expects_transcript=stream_event.expects_transcript,
            )
            await self._apply_interrupt_decision(interrupt_decision, resume_on_reject=False)
            await self._emit(
                {
                    "type": WIRE_SPEECH_STOPPED,
                    "timestamp_ms": stream_event.timestamp_ms,
                }
            )
            await self._event_queue.put(
                TurnEvent(
                    type=TurnEventType.SPEECH_STOPPED,
                    timestamp_ms=stream_event.timestamp_ms,
                    payload={"await_final_transcript": stream_event.expects_transcript},
                )
            )
            if not stream_event.expects_transcript:
                self._interrupt_detector.finish(stream_event.utterance_id)
        elif isinstance(stream_event, StreamTranscript) and stream_event.is_partial:
            if stream_event.text and stream_event.text.strip() and not self._is_in_self_echo_window():
                await self._emit(
                    {
                        "type": WIRE_TRANSCRIPT_DELTA,
                        "delta": stream_event.text,
                        "start_ms": stream_event.start_ms,
                        "end_ms": stream_event.end_ms,
                    }
                )
            if self._interrupt_detector.current() is not None and self._active_response_id is not None:
                cumulative_text = stream_event.text
                if self._speech_session is not None:
                    cumulative_text = self._speech_session.get_confirmed_text() or cumulative_text
                decision = await self._interrupt_detector.observe_partial(
                    stream_event,
                    cumulative_transcript=cumulative_text,
                    assistant_text=self._active_assistant_text(),
                    output_echo=self._looks_like_current_output_echo(),
                    now=time.monotonic(),
                )
                await self._apply_interrupt_decision(decision, resume_on_reject=True)
                if decision.action is InterruptionDecisionAction.CONFIRM:
                    return
        elif isinstance(stream_event, StreamTranscript):
            self._awaiting_final_transcript = False
            self._awaiting_final_transcript_started_at = 0.0

            candidate = self._interrupt_detector.current()
            if (
                candidate is None
                and self._sm.state in {TurnState.SPEAKING, TurnState.PAUSED}
                and looks_like_self_echo(
                    stream_event.text,
                    self._active_assistant_text(),
                    min_words=self._config.policy.self_echo_min_words,
                    min_overlap=self._config.policy.self_echo_min_overlap,
                )
            ):
                await self._emit_interruption_false_positive(
                    vad_active_ms=max(0, stream_event.end_ms - stream_event.start_ms),
                    partial_transcript=stream_event.text,
                    reason="self_echo_transcript",
                )
                return
            if candidate is not None or self._sm.state in {TurnState.SPEAKING, TurnState.PAUSED}:
                audio_duration_ms = max(
                    candidate.vad_active_ms(time.monotonic()) if candidate is not None else 0,
                    stream_event.audio_duration_ms,
                    max(0, stream_event.end_ms - stream_event.start_ms),
                )
                audio = self._audio_history.mic_tail_for_duration_ms(audio_duration_ms)
                interrupt_decision = await self._interrupt_detector.observe_final(
                    stream_event,
                    assistant_text=self._active_assistant_text(),
                    output_echo=self._looks_like_current_output_echo(),
                    audio=audio,
                    sample_rate=TARGET_SAMPLE_RATE,
                    now=time.monotonic(),
                )
                await self._apply_interrupt_decision(interrupt_decision, resume_on_reject=True)
                if interrupt_decision.action is not InterruptionDecisionAction.CONFIRM:
                    if stream_event.utterance_id:
                        self._interrupt_detector.finish(stream_event.utterance_id)
                    return

            enrich_transcript(stream_event, self._config.language)

            if stream_event.eou_probability is not None:
                self._last_eou_probability = float(stream_event.eou_probability)
            self._endpoint_pause_history.record_since(self._last_speech_stopped_at)
            eou_threshold = EOUConfig().threshold
            finalization_decision = transcript_finalization.final_transcript_decision(
                stream_event,
                endpoint_timer_active=self._has_active_timer(TimerKey.ENDPOINTING.value),
                commit_delay_policy=self._endpoint_commit_delay,
                recent_pause_ms=self._endpoint_pause_history.values(),
                eou_threshold=eou_threshold,
                turn_detector=self._config.turn_detector,
            )
            if finalization_decision.eou_event is not None:
                await self._emit(finalization_decision.eou_event)
            self._transcript_finalizer.remember(stream_event)
            pending_text = self._transcript_finalizer.pending_text(stream_event.text)

            if self._sm.state == TurnState.THINKING:
                await self._emit_pending_transcript_done()
                self._interrupt_detector.finish(stream_event.utterance_id)
                return

            await self._event_queue.put(
                TurnEvent(
                    type=TurnEventType.USER_TRANSCRIPT_FINAL,
                    payload={
                        "text": pending_text,
                        "defer_commit": finalization_decision.defer_commit,
                        "commit_delay_ms": finalization_decision.commit_delay_ms,
                    },
                )
            )
            self._interrupt_detector.finish(stream_event.utterance_id)

    async def _ensure_response_stream(
        self,
        *,
        allow_interruptions: bool = True,
    ) -> ResponseStream | None:
        existing = self._response_lifecycle.open_uncommitted_stream()
        if existing is not None:
            return existing
        if self._tts_task and not self._tts_task.done():
            logger.warning("response stream requested while response task already active; ignoring")
            await self._emit_error("response already in flight")
            return None

        rejection_reason = self._response_start_rejection_reason()
        if rejection_reason is not None:
            if not self._response_rejection_reported:
                logger.warning("response stream rejected: %s", rejection_reason)
                await self._emit_error(f"response rejected: {rejection_reason}")
                self._response_rejection_reported = True
            return None

        stream = self._response_lifecycle.start_stream(allow_interruptions=allow_interruptions)
        self._response_rejection_reported = False
        self._interrupt_detector.reset()
        self._output_playout_started = False
        self._audio_output.reset_for_response()
        await self._event_queue.put(TurnEvent(type=TurnEventType.RESPONSE_STARTED))
        await self._emit_response_created(stream.response_id)
        self._tts_task = asyncio.create_task(self._run_response_stream(stream))
        return stream

    def _response_start_rejection_reason(self) -> str | None:
        state = self._sm.state
        if state not in _RESPONSE_START_STATES:
            return f"turn state is {state.value}"
        if state is TurnState.THINKING and self._input_speech_active:
            return "user speech is active"
        return None

    async def _run_response_stream(self, stream: ResponseStream) -> None:
        try:
            async with acquire_typed_adapter(
                self._scheduler,
                model=self._config.tts_model,
                adapter_type=TTSAdapter,
                expected_type="TTS",
            ) as adapter:
                await synthesize_response_stream(
                    adapter=adapter,
                    stream=stream,
                    voice=self._config.voice,
                    language=self._config.language,
                    on_audio_started=self._notify_tts_audio_started,
                    on_audio_chunk=self._handle_tts_chunk,
                )

                assistant_text = stream.assistant_context_text()
                if assistant_text:
                    self._pipeline.add_assistant_turn(assistant_text)

                await self._wait_for_estimated_playout()
                if self._audio_output.paused and self._response_stream is stream:
                    stream.pending_done = True
                    return
                await self._complete_response_stream(stream)
        except AdapterTypeMismatchError as exc:
            await self._fail_response(str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("TTS synthesis failed")
            await self._fail_response(str(exc))
        finally:
            self._response_lifecycle.clear_finished_stream_if_current(stream)

    async def _notify_tts_audio_started(self) -> None:
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        await self._event_queue.put(
            TurnEvent(
                type=TurnEventType.TTS_AUDIO_STARTED,
                payload={"_done": done},
            )
        )
        accepted = await done
        if not accepted or self._sm.state != TurnState.SPEAKING or self._response_stream is None:
            self._response_lifecycle.remember_cancelled_response()
            raise asyncio.CancelledError

    def _accepts_turn_event(self, event: TurnEvent) -> bool:
        return self._sm.accepts(event.type)

    async def _complete_response_stream(self, stream: ResponseStream) -> None:
        stream.pending_done = False
        self._mark_agent_speech_ended()
        self._output_playout_started = False
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        await self._event_queue.put(
            TurnEvent(
                type=TurnEventType.TTS_COMPLETED,
                payload={"_done": done, "_response_stream": stream},
            )
        )
        if asyncio.current_task() is not self._runner:
            await done

    async def _finalize_response_stream(self, stream: ResponseStream) -> None:
        if self._response_stream is not stream:
            return
        await self._emit_response_done(stream.response_id)
        self._response_lifecycle.finish_stream_if_current(stream)
        self._interrupt_detector.reset()

    async def _fail_response(self, message: str) -> None:
        self._interrupt_detector.reset()
        await self._emit_error(message)
        await self._event_queue.put(TurnEvent(type=TurnEventType.TTS_FAILED))

    async def _handle_tts_chunk(self, audio: bytes, sample_rate: int) -> None:
        if not audio:
            return
        output_sample_rate = self._client_sample_rate or self._config.sample_rate
        pcm_audio = np.frombuffer(audio, dtype=np.float32)
        if pcm_audio.size == 0:
            return
        if sample_rate != output_sample_rate:
            pcm_audio = resample_audio(pcm_audio, sample_rate, output_sample_rate)
        encoded_audio = float32_to_pcm16(pcm_audio)
        sequence = self._audio_output.next_sequence()
        if self._audio_output.hold_if_paused(encoded_audio, output_sample_rate, sequence):
            return
        await self._emit_output_audio(encoded_audio, output_sample_rate, sequence)

    def _mark_estimated_playout(self, pcm16_audio: bytes, sample_rate: int) -> None:
        self._audio_output.mark_playout(pcm16_audio, sample_rate)

    async def _wait_for_estimated_playout(self) -> None:
        if self._config.wait_for_output_playout is not None:
            await self._config.wait_for_output_playout()
            return
        delay_s = self._audio_output.playout_delay_s()
        if delay_s > 0:
            await asyncio.sleep(delay_s)

    async def _emit_output_audio(self, encoded_audio: bytes, sample_rate: int, sequence: int) -> None:
        stream = self._response_stream
        if stream is not None and not stream.audio_started:
            stream.audio_started = True
        if not self._config.output_playout_observed:
            self._mark_output_playout_started()
        await self._emit(
            {
                "type": WIRE_AUDIO_DELTA,
                "audio_pcm16": encoded_audio,
                "sample_rate": sample_rate,
                "audio_format": "pcm16",
                "response_id": self._active_response_id,
                "sequence": sequence,
            }
        )
        if not self._config.output_playout_observed:
            self._audio_history.remember_output_pcm16(encoded_audio, sample_rate)
        self._mark_estimated_playout(encoded_audio, sample_rate)

    def observe_output_playout(self, pcm16_audio: bytes, sample_rate: int) -> None:
        """Record audio after the transport has paced it for actual playout."""
        if self._closed or not pcm16_audio:
            return
        self._mark_output_playout_started()
        self._audio_history.remember_output_pcm16(pcm16_audio, sample_rate)

    def _mark_output_playout_started(self) -> None:
        if self._output_playout_started:
            return
        self._output_playout_started = True
        self._arm_aec_warmup()
        self._mark_agent_speech_started()

    def _arm_aec_warmup(self) -> None:
        self._speech_guard.arm_aec_warmup(self._config.policy.aec_warmup_ms)

    def _aec_warmup_active(self) -> bool:
        return self._speech_guard.aec_warmup_active()

    def _is_response_uninterruptible(self) -> bool:
        stream = self._response_lifecycle.stream
        if stream is None or stream.allow_interruptions:
            return False
        return self._sm.state in {TurnState.THINKING, TurnState.SPEAKING}

    def _mark_agent_speech_started(self) -> None:
        self._speech_guard.mark_speech_started()

    def _mark_agent_speech_ended(self) -> None:
        self._speech_guard.mark_speech_ended()

    def _is_in_self_echo_window(self) -> bool:
        if self._sm.state not in {TurnState.SPEAKING, TurnState.PAUSED, TurnState.THINKING}:
            return False
        return self._speech_guard.in_self_echo_window(self._config.policy.backchannel_end_cooldown_ms)

    def _looks_like_current_output_echo(self) -> bool:
        return self._audio_history.looks_like_current_output_echo()

    async def _execute(self, action: TurnAction) -> None:
        if action.type == TurnActionType.PAUSE_OUTPUT:
            self._audio_output.pause()
            if bool(action.payload.get("clear", True)):
                await self._emit_audio_clear()

        elif action.type == TurnActionType.RESUME_OUTPUT:
            stream = self._response_stream
            for pending_batch in self._audio_output.pending_resume_batches():
                for pending in pending_batch:
                    await self._emit_output_audio(pending.audio, pending.sample_rate, pending.sequence)
            self._audio_output.finish_resume()
            self._speech_guard.start_flutter_cooldown(self._config.policy.stable_speaking_min_ms)
            if stream is not None and stream.pending_done:
                await self._complete_response_stream(stream)

        elif action.type == TurnActionType.FLUSH_OUTPUT:
            self._audio_output.flush()
            self._audio_history.clear()
            await self._emit_audio_clear()

        elif action.type == TurnActionType.STOP_TTS:
            self._mark_agent_speech_ended()
            self._output_playout_started = False
            stream = self._response_stream
            self._response_lifecycle.remember_cancelled_response()
            if stream is not None:
                assistant_text = stream.assistant_context_text()
                if assistant_text:
                    self._pipeline.add_assistant_turn(assistant_text)
            if self._tts_task and not self._tts_task.done():
                await reap_task(self._tts_task)
            self._tts_task = None
            self._audio_output.reset_playout()
            self._response_lifecycle.clear_active_response(stream)
            self._interrupt_detector.reset()

        elif action.type == TurnActionType.CANCEL_RESPONSE:
            await self._emit_response_cancelled()

        elif action.type == TurnActionType.START_TIMER:
            key = action.payload["key"]
            duration_ms = int(action.payload["duration_ms"])
            await self._start_timer(key, duration_ms)

        elif action.type == TurnActionType.CANCEL_TIMER:
            await self._cancel_timer(action.payload["key"])

    @staticmethod
    def _resolve_event_future(
        done,
        exc: BaseException | None = None,
        *,
        result=None,
    ) -> None:
        if done is None or not hasattr(done, "done") or done.done():
            return
        if exc is not None:
            done.set_exception(exc)
            return
        done.set_result(result)

    async def _on_timer_expired(self, key: str) -> None:
        if key == TimerKey.CONFIRM_INTERRUPT.value:
            expected_candidate_id = self._interrupt_timer_candidate_id
            self._interrupt_timer_candidate_id = None
            candidate = self._interrupt_detector.current()
            if candidate is None or candidate.candidate_id != expected_candidate_id:
                logger.debug(
                    "ignoring stale interruption timer expected_candidate_id=%s current_candidate_id=%s",
                    expected_candidate_id,
                    candidate.candidate_id if candidate is not None else None,
                )
                return
            await self._evaluate_interrupt_candidate()
            return

        await self._event_queue.put(
            TurnEvent(
                type=TurnEventType.TIMER_ELAPSED,
                payload={"key": key},
            )
        )

    def _has_active_timer(self, key: str) -> bool:
        return self._timer_registry.has_active(key)

    async def _start_timer(self, key: str, duration_ms: int) -> None:
        await self._timer_registry.start(key, duration_ms)
        if key == TimerKey.CONFIRM_INTERRUPT.value:
            candidate = self._interrupt_detector.current()
            self._interrupt_timer_candidate_id = candidate.candidate_id if candidate is not None else None

    async def _emit_pending_transcript_done(self) -> None:
        payload = self._transcript_finalizer.pop()
        if payload is None:
            return
        self._transcript_finalizer.log(payload)
        await self._emit(payload)

    async def _emit_error(self, message: str) -> None:
        await self._emit({"type": WIRE_ERROR, "message": message})

    async def _emit_response_event(self, event_type: str, response_id: str | None) -> None:
        await self._emit({"type": event_type, "response_id": response_id})

    async def _emit_response_created(self, response_id: str) -> None:
        await self._emit_response_event(WIRE_RESPONSE_CREATED, response_id)

    async def _emit_response_committed(self, response_id: str) -> None:
        await self._emit_response_event(WIRE_RESPONSE_COMMITTED, response_id)

    async def _emit_response_done(self, response_id: str) -> None:
        await self._emit_response_event(WIRE_RESPONSE_DONE, response_id)

    async def _emit_response_cancelled(self) -> None:
        await self._emit_response_event(
            WIRE_RESPONSE_CANCELLED,
            self._response_lifecycle.active_or_cancelled_response_id(),
        )

    async def _emit_audio_clear(self) -> None:
        await self._emit_response_event(
            WIRE_AUDIO_CLEAR,
            self._response_lifecycle.active_or_cancelled_response_id(),
        )

    async def _emit_interruption_detected(
        self,
        *,
        vad_active_ms: int,
        partial_transcript: str | None,
        reason: str,
    ) -> None:
        logger.info(
            "interruption confirmed reason=%s vad_active_ms=%d transcript=%r",
            reason,
            vad_active_ms,
            partial_transcript,
        )
        await self._emit(
            {
                "type": WIRE_INTERRUPTION_DETECTED,
                "response_id": self._active_response_id,
                "vad_active_ms": vad_active_ms,
                "partial_transcript": partial_transcript,
                "reason": reason,
            }
        )

    async def _emit_interruption_false_positive(
        self,
        *,
        vad_active_ms: int,
        partial_transcript: str | None,
        reason: str,
    ) -> None:
        logger.info(
            "interruption rejected reason=%s vad_active_ms=%d transcript=%r",
            reason,
            vad_active_ms,
            partial_transcript,
        )
        await self._emit(
            {
                "type": WIRE_INTERRUPTION_FALSE_POSITIVE,
                "response_id": self._active_response_id,
                "vad_active_ms": vad_active_ms,
                "partial_transcript": partial_transcript,
                "reason": reason,
            }
        )

    def _active_assistant_text(self) -> str:
        return self._response_lifecycle.assistant_context_text(separator=" ")

    async def _apply_interrupt_decision(
        self,
        decision: InterruptionDecision,
        *,
        resume_on_reject: bool,
    ) -> None:
        if decision.action is InterruptionDecisionAction.DEFER:
            if decision.candidate_id is not None and decision.retry_after_ms > 0:
                await self._start_timer(
                    TimerKey.CONFIRM_INTERRUPT.value,
                    decision.retry_after_ms,
                )
            return
        if not decision.newly_decided:
            return

        await self._cancel_timer(TimerKey.CONFIRM_INTERRUPT.value)
        if decision.action is InterruptionDecisionAction.CONFIRM:
            await self._emit_interruption_detected(
                vad_active_ms=decision.vad_active_ms,
                partial_transcript=decision.transcript,
                reason=decision.reason,
            )
            await self._event_queue.put(
                TurnEvent(
                    type=TurnEventType.TIMER_ELAPSED,
                    payload={"key": TimerKey.CONFIRM_INTERRUPT.value},
                )
            )
            return

        await self._emit_interruption_false_positive(
            vad_active_ms=decision.vad_active_ms,
            partial_transcript=decision.transcript,
            reason=decision.reason,
        )
        if resume_on_reject:
            await self._event_queue.put(
                TurnEvent(
                    type=TurnEventType.SPEECH_STOPPED,
                    payload={"reason": decision.reason},
                )
            )

    async def _evaluate_interrupt_candidate(self) -> None:
        candidate = self._interrupt_detector.current()
        if candidate is None:
            return

        now = time.monotonic()
        vad_active_ms = max(
            candidate.vad_active_ms(now),
            candidate.latest_partial_duration_ms,
        )
        audio_tail = self._audio_history.mic_tail_for_duration_ms(vad_active_ms)
        try:
            decision = await self._interrupt_detector.evaluate_timeout(
                assistant_text=self._active_assistant_text(),
                output_echo=self._looks_like_current_output_echo(),
                aec_warmup_remaining_ms=self._speech_guard.aec_warmup_remaining_ms(),
                audio=audio_tail,
                sample_rate=TARGET_SAMPLE_RATE,
                last_eou_probability=self._last_eou_probability,
                now=now,
            )
        except Exception:
            logger.exception("interruption detector raised; rejecting candidate")
            await self._emit_interruption_false_positive(
                vad_active_ms=vad_active_ms,
                partial_transcript=candidate.cumulative_transcript or None,
                reason="detector_error",
            )
            self._interrupt_detector.reset()
            await self._event_queue.put(
                TurnEvent(
                    type=TurnEventType.SPEECH_STOPPED,
                    payload={"reason": "detector_error"},
                )
            )
            return
        await self._apply_interrupt_decision(decision, resume_on_reject=True)

    async def _cancel_timer(self, key: str) -> None:
        if key == TimerKey.CONFIRM_INTERRUPT.value:
            self._interrupt_timer_candidate_id = None
        await self._timer_registry.cancel(key)

    def _should_wait_for_final_transcript(self, event: TurnEvent) -> bool:
        return transcript_finalization.should_wait_for_pending_final_transcript(
            event,
            awaiting_final_transcript=self._awaiting_final_transcript,
            awaiting_started_at=self._awaiting_final_transcript_started_at,
            max_endpointing_delay_ms=self._config.policy.max_endpointing_delay_ms,
        )

    async def _emit(self, event: dict) -> None:
        if self._closed:
            return
        try:
            await self._on_event(event)
        except Exception:
            logger.exception("on_event handler raised")

    @property
    def _response_stream(self) -> ResponseStream | None:
        return self._response_lifecycle.stream

    @_response_stream.setter
    def _response_stream(self, stream: ResponseStream | None) -> None:
        self._response_lifecycle.stream = stream

    @property
    def _active_response_id(self) -> str | None:
        return self._response_lifecycle.active_response_id

    @_active_response_id.setter
    def _active_response_id(self, response_id: str | None) -> None:
        self._response_lifecycle.active_response_id = response_id

    @property
    def state(self) -> TurnState:
        return self._sm.state

    @property
    def pending_audio_count(self) -> int:
        return self._audio_output.pending_count
