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
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from vox.conversation import response_stream as response_streams
from vox.conversation import speech_guard as speech_guards
from vox.conversation import transcripts as transcript_finalization
from vox.conversation.audio_history import ConversationAudioHistory
from vox.conversation.audio_output import ResponseAudioOutput
from vox.conversation.interrupt import (
    HeuristicInterruptClassifier,
    InterruptClassifier,
)
from vox.conversation.interruption_detector import (
    EvidenceBasedInterruptDetector,
    InterruptDetector,
    InterruptionCandidateStatus,
    InterruptionDecision,
    InterruptionDecisionAction,
    candidate_timer_arming_ms,
)
from vox.conversation.profiles import DEFAULT_TURN_PROFILE, resolve_turn_profile
from vox.conversation.response_lifecycle import (
    ConversationResponseLifecycle,
    TerminalReason,
    TerminalRecord,
)
from vox.conversation.response_synthesis import synthesize_response_stream
from vox.conversation.speech_guard import AssistantSpeechGuard
from vox.conversation.state_machine import TurnStateMachine
from vox.conversation.timers import ConversationTimerLease, ConversationTimerRegistry
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
from vox.speech_context.service import SpeechContextService
from vox.speech_context.types import SpeechContext, speech_context_payload
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
AppendResult = response_streams.AppendResult


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
ERROR_CODE_RESPONSE_REJECTED_TURN_STATE = "response_rejected_turn_state"
ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH = "response_rejected_user_speech"
ERROR_CODE_RESPONSE_STALE_GENERATION = "response_stale_generation"
ERROR_CODE_RESPONSE_ALREADY_ACTIVE = "response_already_active"
ERROR_CODE_COMMAND_INVALID = "command_invalid"
ERROR_CODE_SESSION_FAILED = "session_failed"
ERROR_CODE_RESPONSE_FAILED = "response_failed"
RESPONSE_STREAM_QUEUE_MAX = response_streams.RESPONSE_STREAM_QUEUE_MAX


EventEmitter = Callable[[dict], Awaitable[None]]
AudioPreprocessor = Callable[[np.ndarray, int], np.ndarray | Awaitable[np.ndarray]]


@dataclass(frozen=True, slots=True)
class ResponseStartContext:
    turn_state: TurnState
    input_speech_active: bool
    candidate_id: int | None
    candidate_status: InterruptionCandidateStatus | None
    candidate_reason: str | None


@dataclass(frozen=True, slots=True)
class ResponseStartRejection:
    message: str
    code: str
    generation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseStartResult:
    context: ResponseStartContext
    response_id: str | None = None
    rejection: ResponseStartRejection | None = None

    def __post_init__(self) -> None:
        if (self.response_id is None) == (self.rejection is None):
            raise ValueError("response start result must be either accepted or rejected")

    @property
    def accepted(self) -> bool:
        return self.response_id is not None


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
    speech_context: bool = False

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
_LIFECYCLE_CRITICAL_ACTIONS = frozenset(
    {
        TurnActionType.PAUSE_OUTPUT,
        TurnActionType.RESUME_OUTPUT,
        TurnActionType.FLUSH_OUTPUT,
        TurnActionType.STOP_TTS,
        TurnActionType.CANCEL_RESPONSE,
    }
)
_TTS_STREAM_EVENT_TYPES = frozenset(
    {
        TurnEventType.TTS_AUDIO_STARTED,
        TurnEventType.TTS_COMPLETED,
        TurnEventType.TTS_FAILED,
    }
)


@dataclass
class _SessionCommand:
    operation: Callable[[], Awaitable[Any]]
    done: asyncio.Future


class ConversationSession:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        config: ConversationConfig,
        on_event: EventEmitter,
        speech_context_service: SpeechContextService | None = None,
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
            speech_context_service=speech_context_service,
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
            speech_context=config.speech_context,
        )
        self._pipeline.configure(self._stream_session_config)

        self._speech_session: SpeechSession | None = None
        self._partial_service: PartialTranscriptService | None = None
        if self._wants_partials:
            self._speech_session = SpeechSession()
            self._partial_service = PartialTranscriptService(
                transcribe_async_fn=self._pipeline.transcribe_async,
            )

        self._event_queue: asyncio.Queue[TurnEvent | _SessionCommand] = asyncio.Queue()
        self._timer_registry = ConversationTimerRegistry(self._on_timer_expired)
        self._interrupt_timer_candidate_id: int | None = None
        self._tts_task: asyncio.Task | None = None
        self._tts_reaper_tasks: set[asyncio.Task] = set()
        self._runner: asyncio.Task | None = None
        self._audio_output = ResponseAudioOutput(pace_to_playout=config.pace_response_done_to_audio)
        self._response_lifecycle = ConversationResponseLifecycle()
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
        self._speech_context_service = speech_context_service
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
        if self._tts_reaper_tasks:
            await asyncio.gather(*tuple(self._tts_reaper_tasks), return_exceptions=True)
        await reap_task(self._runner)

        self._awaiting_final_transcript = False
        self._awaiting_final_transcript_started_at = 0.0
        self._transcript_finalizer.clear()
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
                await self._emit_error(
                    f"audio preprocessor failed: {exc}",
                    code=ERROR_CODE_COMMAND_INVALID,
                )
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
            await self._emit_error(str(exc), code=ERROR_CODE_COMMAND_INVALID)
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

    async def start_response_stream(
        self,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
    ) -> ResponseStartResult:
        if self._closed:
            return ResponseStartResult(
                context=self._response_start_context(),
                rejection=ResponseStartRejection(
                    message="response rejected: session is closed",
                    code=ERROR_CODE_RESPONSE_REJECTED_TURN_STATE,
                    generation_id=generation_id,
                ),
            )
        return await self._submit_command(
            partial(
                self._attempt_response_start,
                allow_interruptions=allow_interruptions,
                generation_id=generation_id,
            )
        )

    async def append_response_text(
        self,
        text: str,
        *,
        allow_interruptions: bool = True,
        expected_response_id: str | None = None,
    ) -> AppendResult:
        if self._closed:
            return AppendResult.SESSION_CLOSED
        if expected_response_id is None and not text.strip():
            return AppendResult.NO_ACTIVE_RESPONSE
        admitted = await self._submit_command(
            partial(
                self._admit_response_text,
                allow_interruptions=allow_interruptions,
                expected_response_id=expected_response_id,
            )
        )
        if admitted is None:
            return AppendResult.SESSION_CLOSED
        if isinstance(admitted, AppendResult):
            return admitted
        if not text:
            return AppendResult.ACCEPTED
        return await admitted.append_text(text)

    async def _admit_response_text(
        self,
        *,
        allow_interruptions: bool,
        expected_response_id: str | None,
    ) -> ResponseStream | AppendResult:
        if self._closed:
            return AppendResult.SESSION_CLOSED
        if expected_response_id is None:
            result = await self._attempt_response_start(allow_interruptions=allow_interruptions)
            if result.rejection is not None:
                self._log_response_start_rejection(result.context, result.rejection)
                await self._emit_error(
                    result.rejection.message,
                    code=result.rejection.code,
                )
                return AppendResult.NO_ACTIVE_RESPONSE
            if result.response_id is None:
                return AppendResult.NO_ACTIVE_RESPONSE
            return self._response_lifecycle.appendable_stream(result.response_id)
        return self._response_lifecycle.appendable_stream(expected_response_id)

    async def commit_response_stream(self, *, expected_response_id: str | None = None) -> AppendResult:
        if self._closed:
            return AppendResult.SESSION_CLOSED
        admitted = await self._submit_command(
            partial(self._admit_response_commit, expected_response_id=expected_response_id)
        )
        if admitted is None:
            return AppendResult.SESSION_CLOSED
        if isinstance(admitted, AppendResult):
            return admitted
        return await admitted.enqueue_end()

    async def _admit_response_commit(self, *, expected_response_id: str | None) -> ResponseStream | AppendResult:
        if self._closed:
            return AppendResult.SESSION_CLOSED
        stream = self._response_lifecycle.stream
        if stream is None:
            return AppendResult.NO_ACTIVE_RESPONSE
        if expected_response_id is not None and stream.response_id != expected_response_id:
            return AppendResult.RESPONSE_MISMATCH
        if stream.committed:
            return AppendResult.RESPONSE_COMMITTED
        if not self._response_lifecycle.commit_stream(stream):
            return AppendResult.STREAM_ENDED
        await self._emit_response_committed(stream)
        return stream

    async def _submit_command(self, operation: Callable[[], Awaitable[Any]], *, default: Any = None) -> Any:
        runner = self._runner
        if runner is None or runner.done() or asyncio.current_task() is runner:
            return await operation()
        loop = asyncio.get_running_loop()
        done: asyncio.Future = loop.create_future()
        await self._event_queue.put(_SessionCommand(operation=operation, done=done))
        completed, _ = await asyncio.wait({done, runner}, return_when=asyncio.FIRST_COMPLETED)
        if done not in completed:
            done.cancel()
            return default
        return done.result()

    async def cancel_response(self) -> None:
        """Explicit client cancel — orthogonal to barge-in."""
        runner = self._runner
        if self._closed or runner is None or runner.done():
            return
        if asyncio.current_task() is runner:
            await self._process_turn_event(TurnEvent(type=TurnEventType.CLIENT_CANCEL, payload={}))
            return
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        await self._event_queue.put(
            TurnEvent(
                type=TurnEventType.CLIENT_CANCEL,
                payload={"_done": done},
            )
        )
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
                item = await self._event_queue.get()
            except asyncio.CancelledError:
                break

            if isinstance(item, _SessionCommand):
                await self._run_session_command(item)
                continue

            await self._process_turn_event(item)

    async def _process_turn_event(self, event: TurnEvent) -> None:
        prev_state = self._sm.state
        payload = event.payload if isinstance(event.payload, dict) else {}
        done = payload.get("_done")
        stream_ref = payload.get("_response_stream")
        if not isinstance(stream_ref, ResponseStream):
            stream_ref = None
        if (
            stream_ref is not None
            and event.type in _TTS_STREAM_EVENT_TYPES
            and stream_ref is not self._response_lifecycle.stream
        ):
            self._resolve_event_future(done, result=False)
            return
        if event.type == TurnEventType.CLIENT_CANCEL and isinstance(event.payload, dict):
            event.payload["has_active_response"] = self._response_pipeline_engaged()
        timer_lease = payload.get("_timer_lease")
        if isinstance(timer_lease, ConversationTimerLease) and not self._timer_registry.consume(timer_lease):
            self._resolve_event_future(done)
            return
        if (
            event.type == TurnEventType.TIMER_ELAPSED
            and payload.get("key") == TimerKey.ENDPOINTING.value
            and self._input_speech_active
        ):
            self._resolve_event_future(done)
            return
        if self._should_wait_for_final_transcript(event):
            await self._start_timer(
                TimerKey.ENDPOINTING.value,
                TRANSCRIPT_PENDING_STT_RECHECK_MS,
            )
            self._resolve_event_future(done)
            return
        try:
            accepted = self._accepts_turn_event(event)
            actions = self._sm.handle(event) if accepted else []
        except Exception as exc:
            logger.exception("state machine raised on event %s", event)
            self._resolve_event_future(done, exc)
            return

        failed_action: TurnAction | None = None
        teardown: tuple[ResponseStream, TerminalRecord] | None = None
        for action in actions:
            try:
                teardown = await self._execute(action, teardown)
            except Exception:
                if action.type in _LIFECYCLE_CRITICAL_ACTIONS:
                    logger.exception(
                        "lifecycle-critical action %s raised; forcing response pipeline recovery",
                        action.type.value,
                    )
                    failed_action = action
                    break
                logger.exception(
                    "bookkeeping action %s raised; skipped payload=%s",
                    action.type.value,
                    action.payload,
                )
        if failed_action is not None:
            await self._recover_from_action_failure(failed_action)

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

        if stream_ref is not None:
            await self._apply_tts_stream_outcome(event, stream_ref, accepted)

        self._resolve_event_future(done, result=accepted)

    async def _run_session_command(self, command: _SessionCommand) -> None:
        try:
            result = await command.operation()
        except Exception as exc:
            if command.done.done():
                logger.exception("session command raised after caller stopped waiting")
            else:
                command.done.set_exception(exc)
            return
        if not command.done.done():
            command.done.set_result(result)

    def _response_pipeline_engaged(self) -> bool:
        if self._response_lifecycle.stream is not None:
            return True
        if self._response_lifecycle.terminal is not None:
            return False
        return self._tts_task is not None and not self._tts_task.done()

    async def _apply_tts_stream_outcome(
        self,
        event: TurnEvent,
        stream: ResponseStream,
        accepted: bool,
    ) -> None:
        if event.type == TurnEventType.TTS_COMPLETED:
            if accepted:
                record = self._response_lifecycle.terminalize(stream, "done")
                if record is not None:
                    await self._emit_response_done(record)
                    self._interrupt_detector.reset()
            elif not stream.pending_done:
                self._response_lifecycle.terminalize(stream, "done")
        elif event.type == TurnEventType.TTS_FAILED:
            self._response_lifecycle.terminalize(stream, "failed")

    def _release_tts_task(self, task: asyncio.Task | None) -> None:
        if task is None:
            return
        task.cancel()
        reaper = asyncio.create_task(reap_task(task))
        self._tts_reaper_tasks.add(reaper)
        reaper.add_done_callback(self._tts_reaper_tasks.discard)

    async def _recover_from_action_failure(self, failed_action: TurnAction) -> None:
        stream = self._response_lifecycle.stream
        tts_task = self._tts_task
        tts_active = tts_task is not None and not tts_task.done()
        record: TerminalRecord | None = None
        if stream is not None:
            record = self._response_lifecycle.terminalize(stream, "cancelled")
        await self._emit_error(
            f"action {failed_action.type.value} failed; response pipeline reset",
            code=ERROR_CODE_SESSION_FAILED,
            recoverable=False,
        )
        with suppress(Exception):
            self._timer_registry.cancel_all()
        self._interrupt_timer_candidate_id = None
        with suppress(Exception):
            self._audio_output.flush()
        with suppress(Exception):
            self._audio_history.clear()
        if record is not None:
            await self._emit_audio_clear(record)
            await self._emit_response_cancelled(record)
        if tts_active:
            self._release_tts_task(tts_task)
        self._tts_task = None
        with suppress(Exception):
            self._interrupt_detector.reset()
        self._speech_guard.mark_speech_ended(self._config.policy.backchannel_end_cooldown_ms)
        self._output_playout_started = False
        self._sm.handle(TurnEvent(type=TurnEventType.RECOVER))

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

            interruption_states = {TurnState.THINKING, TurnState.SPEAKING, TurnState.PAUSED}
            if self._sm.state in interruption_states and stream_event.utterance_id <= 0:
                logger.warning("ignoring interruption candidate without utterance identity")
                await self._emit_interruption_false_positive(
                    vad_active_ms=0,
                    partial_transcript=None,
                    reason="missing_candidate_identity",
                )
                return

            if self._sm.state in interruption_states:
                self._interrupt_detector.begin(
                    utterance_id=stream_event.utterance_id,
                    started_at=time.monotonic(),
                    assistant_text=self._active_assistant_text(),
                )

            confirm_ms = self._interrupt_detector.confirm_window_ms(
                self._config.policy.min_interrupt_duration_ms,
                self._last_eou_probability,
            )
            await self._event_queue.put(
                TurnEvent(
                    type=TurnEventType.SPEECH_STARTED,
                    timestamp_ms=stream_event.timestamp_ms,
                    payload={
                        "confirm_window_ms": self._interrupt_timer_arming_ms(confirm_ms),
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
            if stream_event.text and stream_event.text.strip() and not self._suppresses_transcript_trust():
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
                and self._interrupt_detector.is_self_echo(
                    stream_event.text,
                    self._active_assistant_text(),
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
                decided = self._interrupt_detector.current()
                confirmed = (
                    decided is not None
                    and decided.utterance_id == stream_event.utterance_id
                    and decided.status is InterruptionCandidateStatus.CONFIRMED
                )
                await self._apply_interrupt_decision(interrupt_decision, resume_on_reject=True)
                if not confirmed:
                    if stream_event.utterance_id and not self._input_speech_active:
                        self._interrupt_detector.finish(stream_event.utterance_id)
                    return

            enrich_transcript(stream_event, self._config.language)

            self._last_eou_probability = (
                float(stream_event.eou_probability) if stream_event.eou_probability is not None else None
            )
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

    async def _attempt_response_start(
        self,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
    ) -> ResponseStartResult:
        context = self._response_start_context()
        existing = self._response_lifecycle.open_uncommitted_stream()
        if existing is not None:
            return ResponseStartResult(context=context, response_id=existing.response_id)
        if self._tts_task and not self._tts_task.done():
            rejection = ResponseStartRejection(
                message="response already in flight",
                code=ERROR_CODE_RESPONSE_ALREADY_ACTIVE,
                generation_id=generation_id,
            )
            return ResponseStartResult(context=context, rejection=rejection)

        rejection = self._response_start_rejection_reason()
        if rejection is not None:
            rejection_reason, rejection_code = rejection
            rejected = ResponseStartRejection(
                message=f"response rejected: {rejection_reason}",
                code=rejection_code,
                generation_id=generation_id,
            )
            return ResponseStartResult(context=context, rejection=rejected)

        stream = self._response_lifecycle.start_stream(
            allow_interruptions=allow_interruptions,
            generation_id=generation_id,
        )
        if not (context.input_speech_active and context.candidate_status is InterruptionCandidateStatus.REJECTED):
            self._interrupt_detector.reset()
        self._output_playout_started = False
        self._audio_output.reset_for_response()
        await self._event_queue.put(TurnEvent(type=TurnEventType.RESPONSE_STARTED))
        await self._emit_response_created(stream)
        self._tts_task = asyncio.create_task(self._run_response_stream(stream))
        return ResponseStartResult(context=context, response_id=stream.response_id)

    def _response_start_context(self) -> ResponseStartContext:
        candidate = self._interrupt_detector.current()
        return ResponseStartContext(
            turn_state=self._sm.state,
            input_speech_active=self._input_speech_active,
            candidate_id=candidate.candidate_id if candidate is not None else None,
            candidate_status=candidate.status if candidate is not None else None,
            candidate_reason=candidate.decision_reason if candidate is not None else None,
        )

    @staticmethod
    def _log_response_start_rejection(
        context: ResponseStartContext,
        rejection: ResponseStartRejection,
    ) -> None:
        logger.warning(
            "response stream rejected code=%s state=%s input_speech_active=%s "
            "candidate_id=%s candidate_status=%s candidate_reason=%s",
            rejection.code,
            context.turn_state.value,
            context.input_speech_active,
            context.candidate_id,
            context.candidate_status.value if context.candidate_status is not None else None,
            context.candidate_reason,
        )

    def _response_start_rejection_reason(self) -> tuple[str, str] | None:
        state = self._sm.state
        if state not in _RESPONSE_START_STATES:
            return f"turn state is {state.value}", ERROR_CODE_RESPONSE_REJECTED_TURN_STATE
        if state is TurnState.THINKING and self._input_speech_active:
            candidate = self._interrupt_detector.current()
            unfinished_turn = transcript_finalization.eou_indicates_incomplete_turn(
                self._last_eou_probability,
                threshold=EOUConfig().threshold,
            )
            if candidate is None or candidate.status is not InterruptionCandidateStatus.REJECTED or unfinished_turn:
                return "user speech is active", ERROR_CODE_RESPONSE_REJECTED_USER_SPEECH
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
                    on_audio_started=partial(self._notify_tts_audio_started, stream),
                    on_audio_chunk=partial(self._handle_tts_chunk, stream),
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
            await self._fail_response(str(exc), stream)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("TTS synthesis failed")
            await self._fail_response(str(exc), stream)

    async def _notify_tts_audio_started(self, stream: ResponseStream | None = None) -> None:
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        payload: dict[str, Any] = {"_done": done}
        if stream is not None:
            payload["_response_stream"] = stream
        await self._event_queue.put(
            TurnEvent(
                type=TurnEventType.TTS_AUDIO_STARTED,
                payload=payload,
            )
        )
        accepted = await done
        current = self._response_lifecycle.stream
        rejected = (
            not accepted
            or self._sm.state != TurnState.SPEAKING
            or current is None
            or (stream is not None and current is not stream)
        )
        if rejected:
            await self._submit_command(partial(self._reject_started_stream, stream))
            raise asyncio.CancelledError

    async def _reject_started_stream(self, stream: ResponseStream | None) -> None:
        if stream is not None:
            self._response_lifecycle.terminalize(stream, "cancelled")

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

    async def _fail_response(self, message: str, stream: ResponseStream | None = None) -> None:
        self._interrupt_detector.reset()
        await self._emit_error(
            message,
            code=ERROR_CODE_RESPONSE_FAILED,
            generation_id=stream.generation_id if stream is not None else None,
        )
        payload: dict[str, Any] = {}
        if stream is not None:
            payload["_response_stream"] = stream
        await self._event_queue.put(TurnEvent(type=TurnEventType.TTS_FAILED, payload=payload))

    async def _handle_tts_chunk(self, stream: ResponseStream, audio: bytes, sample_rate: int) -> None:
        if not audio:
            return
        if stream.closed or stream is not self._response_lifecycle.stream:
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
        self._speech_guard.arm(speech_guards.TTS_START_WARMUP, self._config.policy.aec_warmup_ms)
        self._mark_agent_speech_started()

    def _is_response_uninterruptible(self) -> bool:
        stream = self._response_lifecycle.stream
        if stream is None or stream.allow_interruptions:
            return False
        return self._sm.state in {TurnState.THINKING, TurnState.SPEAKING}

    def _mark_agent_speech_started(self) -> None:
        self._speech_guard.mark_speech_started()

    def _mark_agent_speech_ended(self) -> None:
        self._speech_guard.mark_speech_ended(self._config.policy.backchannel_end_cooldown_ms)

    def _suppresses_transcript_trust(self) -> bool:
        if self._sm.state not in {TurnState.SPEAKING, TurnState.PAUSED, TurnState.THINKING}:
            return False
        return self._speech_guard.suppresses_transcript_trust(time.monotonic())

    def _looks_like_current_output_echo(self) -> bool:
        return self._audio_history.looks_like_current_output_echo()

    def _interrupt_timer_arming_ms(self, confirm_ms: int) -> int:
        now = time.monotonic()
        distrust_ms = (
            self._speech_guard.interrupt_evidence_distrust_remaining_ms(now)
            if self._speech_guard.suppresses_interrupt_evidence(now)
            else 0
        )
        return candidate_timer_arming_ms(
            confirm_window_ms=confirm_ms,
            false_interruption_timeout_ms=self._config.policy.false_interruption_timeout_ms,
            echo_exposed=self._output_playout_started and self._looks_like_current_output_echo(),
            evidence_distrust_remaining_ms=distrust_ms,
        )

    async def _execute(
        self,
        action: TurnAction,
        teardown: tuple[ResponseStream, TerminalRecord] | None = None,
    ) -> tuple[ResponseStream, TerminalRecord] | None:
        if action.type == TurnActionType.PAUSE_OUTPUT:
            self._audio_output.pause()
            if bool(action.payload.get("clear", True)):
                await self._emit_audio_clear(self._response_stream)

        elif action.type == TurnActionType.RESUME_OUTPUT:
            stream = self._response_stream
            for pending_batch in self._audio_output.pending_resume_batches():
                for pending in pending_batch:
                    await self._emit_output_audio(pending.audio, pending.sample_rate, pending.sequence)
            self._audio_output.finish_resume()
            self._speech_guard.arm(speech_guards.RESUME_STABILITY, speech_guards.RESUME_STABILITY_MS)
            if stream is not None and stream.pending_done:
                await self._complete_response_stream(stream)

        elif action.type == TurnActionType.FLUSH_OUTPUT:
            self._audio_output.flush()
            self._audio_history.clear()
            teardown = self._terminalize_current_stream("cancelled", teardown)
            if teardown is not None:
                await self._emit_audio_clear(teardown[1])

        elif action.type == TurnActionType.STOP_TTS:
            self._mark_agent_speech_ended()
            self._output_playout_started = False
            teardown = self._terminalize_current_stream("cancelled", teardown)
            if teardown is not None:
                assistant_text = teardown[0].assistant_context_text()
                if assistant_text:
                    self._pipeline.add_assistant_turn(assistant_text)
            if self._tts_task and not self._tts_task.done():
                self._release_tts_task(self._tts_task)
            self._tts_task = None
            self._audio_output.reset_playout()
            self._interrupt_detector.reset()

        elif action.type == TurnActionType.CANCEL_RESPONSE:
            teardown = self._terminalize_current_stream("cancelled", teardown)
            if teardown is not None:
                await self._emit_response_cancelled(teardown[1])

        elif action.type == TurnActionType.START_TIMER:
            key = action.payload["key"]
            duration_ms = int(action.payload["duration_ms"])
            await self._start_timer(key, duration_ms)

        elif action.type == TurnActionType.CANCEL_TIMER:
            await self._cancel_timer(action.payload["key"])

        return teardown

    def _terminalize_current_stream(
        self,
        reason: TerminalReason,
        teardown: tuple[ResponseStream, TerminalRecord] | None,
    ) -> tuple[ResponseStream, TerminalRecord] | None:
        stream = self._response_lifecycle.stream
        if stream is None:
            return teardown
        record = self._response_lifecycle.terminalize(stream, reason)
        if record is None:
            return teardown
        return stream, record

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

    async def _on_timer_expired(self, lease: ConversationTimerLease) -> None:
        key = lease.key
        if key == TimerKey.CONFIRM_INTERRUPT.value:
            if not self._timer_registry.consume(lease):
                return
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
                payload={"key": key, "_timer_lease": lease},
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
        payload, audio = self._transcript_finalizer.pop_with_audio()
        if payload is None:
            return
        if self._config.speech_context:
            if len(audio) > 1 and self._speech_context_service is not None:
                try:
                    context = await self._speech_context_service.analyze_chunks(
                        audio,
                        timeline_offset_ms=int(payload.get("start_ms") or 0),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Conversation speech context analysis failed")
                    context = SpeechContext(
                        status="failed",
                        unavailable=("prosody", "audio_events"),
                    )
                payload["speech_context"] = speech_context_payload(context)
            elif "speech_context" not in payload:
                payload["speech_context"] = speech_context_payload(
                    SpeechContext(
                        status="failed",
                        unavailable=("prosody", "audio_events"),
                    )
                )
        self._transcript_finalizer.log(payload)
        await self._emit(payload)

    async def _emit_error(
        self,
        message: str,
        *,
        code: str,
        recoverable: bool = True,
        generation_id: str | None = None,
    ) -> None:
        payload = {
            "type": WIRE_ERROR,
            "message": message,
            "code": code,
            "recoverable": recoverable,
        }
        if generation_id:
            payload["generation_id"] = generation_id
        await self._emit(payload)

    async def _emit_response_event(
        self,
        event_type: str,
        source: ResponseStream | TerminalRecord | None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": event_type,
            "response_id": source.response_id if source is not None else None,
        }
        if source is not None and source.generation_id:
            payload["generation_id"] = source.generation_id
        await self._emit(payload)

    async def _emit_response_created(self, stream: ResponseStream) -> None:
        await self._emit_response_event(WIRE_RESPONSE_CREATED, stream)

    async def _emit_response_committed(self, stream: ResponseStream) -> None:
        await self._emit_response_event(WIRE_RESPONSE_COMMITTED, stream)

    async def _emit_response_done(self, record: TerminalRecord) -> None:
        await self._emit_response_event(WIRE_RESPONSE_DONE, record)

    async def _emit_response_cancelled(self, record: TerminalRecord) -> None:
        await self._emit_response_event(WIRE_RESPONSE_CANCELLED, record)

    async def _emit_audio_clear(self, source: ResponseStream | TerminalRecord | None) -> None:
        await self._emit_response_event(WIRE_AUDIO_CLEAR, source)

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
            self._interruption_event_payload(
                WIRE_INTERRUPTION_DETECTED,
                vad_active_ms=vad_active_ms,
                partial_transcript=partial_transcript,
                reason=reason,
            )
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
            self._interruption_event_payload(
                WIRE_INTERRUPTION_FALSE_POSITIVE,
                vad_active_ms=vad_active_ms,
                partial_transcript=partial_transcript,
                reason=reason,
            )
        )

    def _interruption_event_payload(
        self,
        event_type: str,
        *,
        vad_active_ms: int,
        partial_transcript: str | None,
        reason: str,
    ) -> dict[str, Any]:
        stream = self._response_stream
        payload: dict[str, Any] = {
            "type": event_type,
            "response_id": stream.response_id if stream is not None else None,
            "vad_active_ms": vad_active_ms,
            "partial_transcript": partial_transcript,
            "reason": reason,
        }
        if stream is not None and stream.generation_id:
            payload["generation_id"] = stream.generation_id
        return payload

    def _active_assistant_text(self) -> str:
        return self._response_lifecycle.assistant_context_text(separator=" ")

    async def _apply_interrupt_decision(
        self,
        decision: InterruptionDecision,
        *,
        resume_on_reject: bool,
    ) -> None:
        if decision.action is InterruptionDecisionAction.DEFER:
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

    @property
    def _active_response_id(self) -> str | None:
        stream = self._response_lifecycle.stream
        return stream.response_id if stream is not None else None

    @property
    def state(self) -> TurnState:
        return self._sm.state

    @property
    def response_active(self) -> bool:
        return self._response_pipeline_engaged()

    @property
    def active_response_id(self) -> str | None:
        return self._active_response_id

    @property
    def active_generation_id(self) -> str | None:
        stream = self._response_lifecycle.stream
        return stream.generation_id if stream is not None else None

    @property
    def terminal_record(self) -> TerminalRecord | None:
        return self._response_lifecycle.terminal

    @property
    def pending_audio_count(self) -> int:
        return self._audio_output.pending_count
