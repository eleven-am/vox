"""Async orchestrator for a single voice conversation.

Wires together:
  * the pure `TurnStateMachine`
  * Vox's existing streaming pipeline (VAD + STT + EOU)
  * a TTS adapter acquired from the scheduler
  * a timer registry (asyncio tasks)
  * a pause-buffer for held TTS audio during barge-in confirmation
  * an event emitter for client-facing notifications

Concurrency model
-----------------
One `asyncio.Task` drives the state machine (`_run_loop`); it is the **only**
mutator of `_paused`, `_timers`, and `_tts_task`. The audio ingest path and the
TTS task push into queues/buffers; the main loop pulls from them. There are a
few small races on `_pending_audio` (TTS task appends while main loop clears)
that are tolerated: in those paths we cancel TTS immediately afterward, so any
leaked chunk is discarded when the task unwinds.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass

import numpy as np

from vox.conversation.interrupt import HeuristicInterruptClassifier, InterruptClassifier
from vox.conversation.state_machine import TurnStateMachine
from vox.conversation.text_buffer import StreamingTextBuffer, split_for_tts
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
from vox.core.scheduler import Scheduler
from vox.streaming.annotation import enrich_transcript
from vox.streaming.codecs import float32_to_pcm16, pcm16_to_float32, resample_audio
from vox.streaming.partials import PartialTranscriptService
from vox.streaming.pipeline import StreamPipeline
from vox.streaming.session import SpeechSession
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    SpeechStarted,
    SpeechStopped,
    StreamSessionConfig,
    StreamTranscript,
)

logger = logging.getLogger(__name__)



WIRE_SPEECH_STARTED = "input_audio_buffer.speech_started"
WIRE_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
WIRE_TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"
WIRE_RESPONSE_CREATED = "response.created"
WIRE_AUDIO_DELTA = "response.audio.delta"
WIRE_RESPONSE_DONE = "response.done"
WIRE_RESPONSE_CANCELLED = "response.cancelled"
WIRE_RESPONSE_COMMITTED = "response.committed"
WIRE_STATE_CHANGED = "turn.state_changed"
WIRE_ERROR = "error"


EventEmitter = Callable[[dict], Awaitable[None]]


@dataclass
class ConversationConfig:
    stt_model: str
    tts_model: str
    voice: str | None = None
    language: str = "en"
    sample_rate: int = TARGET_SAMPLE_RATE
    policy: TurnPolicy = None  # type: ignore[assignment]

    interrupt_classifier: InterruptClassifier | None = None

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = TurnPolicy()
        if self.interrupt_classifier is None:
            self.interrupt_classifier = HeuristicInterruptClassifier(language=self.language)


_RESPONSE_STREAM_END = object()






RESPONSE_STREAM_QUEUE_MAX = 1024
TRANSCRIPT_CONTINUATION_COMMIT_MS = 1200


@dataclass
class _ResponseStream:
    queue: asyncio.Queue[str | object]
    committed: bool = False


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

        self._wants_partials = bool(self._config.interrupt_classifier.wants_short_circuit())

        self._pipeline = StreamPipeline(scheduler=scheduler)
        self._stream_session_config = StreamSessionConfig(
            language=config.language,
            sample_rate=TARGET_SAMPLE_RATE,
            model=config.stt_model,
            partials=self._wants_partials,
            include_word_timestamps=False,
        )
        self._pipeline.configure(self._stream_session_config)

        self._speech_session: SpeechSession | None = None
        self._partial_service: PartialTranscriptService | None = None
        self._latest_partial: StreamTranscript | None = None
        if self._wants_partials:
            self._speech_session = SpeechSession()
            self._partial_service = PartialTranscriptService(
                transcribe_async_fn=self._pipeline.transcribe_async,
            )

        self._event_queue: asyncio.Queue[TurnEvent] = asyncio.Queue()
        self._timers: dict[str, asyncio.Task] = {}
        self._tts_task: asyncio.Task | None = None
        self._runner: asyncio.Task | None = None
        self._paused: bool = False
        self._pending_audio: list[tuple[bytes, int]] = []
        self._response_stream: _ResponseStream | None = None
        self._closed: bool = False
        self._client_sample_rate: int = config.sample_rate


        self._flutter_cooldown_until: float = 0.0


        self._last_eou_probability: float | None = None


        self._vad_started_at: float | None = None



        self._audio_ring: np.ndarray = np.empty(0, dtype=np.float32)
        self._audio_ring_max_samples: int = TARGET_SAMPLE_RATE * 2





    async def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(self._run_loop())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for task in list(self._timers.values()):
            task.cancel()
        self._timers.clear()

        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()

        if self._runner and not self._runner.done():
            self._runner.cancel()

        for task in (self._tts_task, self._runner):
            if task is None:
                continue
            with suppress(asyncio.CancelledError, Exception):
                await task

        self._pipeline.shutdown()

    async def wait_until_settled(self, *, poll_interval_s: float = 0.01) -> None:
        """Drain timers, queued actions, and any in-flight TTS after client EOF.

        The runner task stays alive for the session lifetime, so "settled" here
        means there is no pending work left that can emit additional events.
        """
        while True:
            timers_active = any(not task.done() for task in self._timers.values())
            tts_active = self._tts_task is not None and not self._tts_task.done()
            queue_busy = not self._event_queue.empty()
            if not timers_active and not tts_active and not queue_busy:
                await asyncio.sleep(0)
                timers_active = any(not task.done() for task in self._timers.values())
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
            audio = resample_audio(audio, source_rate, TARGET_SAMPLE_RATE)



        if audio.size:
            self._audio_ring = np.concatenate([self._audio_ring, audio])
            if self._audio_ring.size > self._audio_ring_max_samples:
                self._audio_ring = self._audio_ring[-self._audio_ring_max_samples:]
            if self._speech_session is not None:
                self._speech_session.append_audio(audio)

        try:
            async for stream_event in self._pipeline.process_audio(audio):
                await self._forward_stream_event(stream_event)
        except Exception as exc:
            logger.exception("pipeline.process_audio raised")
            await self._emit({"type": WIRE_ERROR, "message": str(exc)})
            return

        if (
            self._wants_partials
            and self._partial_service is not None
            and self._speech_session is not None
            and self._speech_session.is_active()
        ):
            try:
                partial = await self._partial_service.generate_partial_async(
                    self._speech_session, self._stream_session_config,
                )
            except Exception:
                logger.exception("partial transcript generation raised")
                partial = None
            if partial is not None:
                await self._forward_stream_event(partial)

    async def submit_response_text(self, text: str) -> None:
        """Agent delivers the reply text; session kicks off TTS."""
        if self._closed:
            return
        await self.append_response_text(text)
        await self.commit_response_stream()

    async def start_response_stream(self) -> None:
        if self._closed:
            return
        await self._ensure_response_stream()

    async def append_response_text(self, text: str) -> None:
        if self._closed or not text or not text.strip():
            return
        stream = await self._ensure_response_stream()
        if stream is None:
            return
        await stream.queue.put(text)

    async def commit_response_stream(self) -> None:
        stream = self._response_stream
        if stream is None or stream.committed:
            return
        stream.committed = True



        await self._emit({"type": WIRE_RESPONSE_COMMITTED})
        await stream.queue.put(_RESPONSE_STREAM_END)

    async def cancel_response(self) -> None:
        """Explicit client cancel — orthogonal to barge-in."""
        await self._event_queue.put(TurnEvent(type=TurnEventType.CLIENT_CANCEL))





    async def _run_loop(self) -> None:
        while not self._closed:
            try:
                event = await self._event_queue.get()
            except asyncio.CancelledError:
                break

            prev_state = self._sm.state
            try:
                actions = self._sm.handle(event)
            except Exception:
                logger.exception("state machine raised on event %s", event)
                continue

            for action in actions:
                try:
                    await self._execute(action)
                except Exception:
                    logger.exception("action %s raised", action.type.value)

            if self._sm.state != prev_state:
                await self._emit({
                    "type": WIRE_STATE_CHANGED,
                    "state": self._sm.state.value,
                    "previous_state": prev_state.value,
                })





    async def _forward_stream_event(self, stream_event) -> None:
        if isinstance(stream_event, SpeechStarted):
            if self._speech_session is not None:
                self._speech_session.start_speech()
            await self._emit({
                "type": WIRE_SPEECH_STARTED,
                "timestamp_ms": stream_event.timestamp_ms,
            })


            if self._sm.state == TurnState.SPEAKING and time.monotonic() < self._flutter_cooldown_until:
                logger.debug(
                    "flutter cooldown active; suppressing SPEECH_STARTED state transition"
                )
                return
            self._vad_started_at = time.monotonic()
            confirm_ms = self._config.interrupt_classifier.confirm_window_ms(
                self._config.policy.min_interrupt_duration_ms,
                self._last_eou_probability,
            )
            await self._event_queue.put(TurnEvent(
                type=TurnEventType.SPEECH_STARTED,
                timestamp_ms=stream_event.timestamp_ms,
                payload={"confirm_window_ms": confirm_ms},
            ))
        elif isinstance(stream_event, SpeechStopped):
            if self._speech_session is not None:
                self._speech_session.stop_speech()
            self._vad_started_at = None
            await self._emit({
                "type": WIRE_SPEECH_STOPPED,
                "timestamp_ms": stream_event.timestamp_ms,
            })
            await self._event_queue.put(TurnEvent(
                type=TurnEventType.SPEECH_STOPPED,
                timestamp_ms=stream_event.timestamp_ms,
            ))
        elif isinstance(stream_event, StreamTranscript) and stream_event.is_partial:
            self._latest_partial = stream_event
            if (
                self._sm.state == TurnState.PAUSED
                and self._config.interrupt_classifier.should_short_circuit(stream_event.text)
                and self._has_active_timer(TimerKey.CONFIRM_INTERRUPT.value)
            ):
                await self._cancel_timer(TimerKey.CONFIRM_INTERRUPT.value)
                await self._event_queue.put(TurnEvent(
                    type=TurnEventType.TIMER_ELAPSED,
                    payload={"key": TimerKey.CONFIRM_INTERRUPT.value},
                ))
        elif isinstance(stream_event, StreamTranscript):
            self._latest_partial = None


            enrich_transcript(stream_event, self._config.language)

            if stream_event.eou_probability is not None:
                self._last_eou_probability = float(stream_event.eou_probability)
            payload = {
                "type": WIRE_TRANSCRIPT_DONE,
                "transcript": stream_event.text,
                "language": self._config.language,
                "start_ms": stream_event.start_ms,
                "end_ms": stream_event.end_ms,
            }
            if stream_event.eou_probability is not None:
                payload["eou_probability"] = stream_event.eou_probability
            if stream_event.entities:
                payload["entities"] = stream_event.entities
            if stream_event.topics:
                payload["topics"] = stream_event.topics
            if stream_event.words:
                payload["words"] = stream_event.words
            await self._emit(payload)

            defer_commit = self._has_active_timer(TimerKey.ENDPOINTING.value)
            await self._event_queue.put(TurnEvent(
                type=TurnEventType.USER_TRANSCRIPT_FINAL,
                payload={
                    "text": stream_event.text,
                    "defer_commit": defer_commit,
                    "commit_delay_ms": self._transcript_commit_delay_ms() if defer_commit else 0,
                },
            ))





    async def _ensure_response_stream(self) -> _ResponseStream | None:
        if self._response_stream is not None and not self._response_stream.committed:
            return self._response_stream
        if self._tts_task and not self._tts_task.done():
            logger.warning("response stream requested while response task already active; ignoring")
            await self._emit({
                "type": WIRE_ERROR,
                "message": "response already in flight",
            })
            return None

        stream = _ResponseStream(queue=asyncio.Queue(maxsize=RESPONSE_STREAM_QUEUE_MAX))
        self._response_stream = stream
        await self._event_queue.put(TurnEvent(type=TurnEventType.RESPONSE_STARTED))
        await self._emit({"type": WIRE_RESPONSE_CREATED})
        self._tts_task = asyncio.create_task(self._run_response_stream(stream))
        return stream

    async def _run_response_stream(self, stream: _ResponseStream) -> None:
        audio_started = False
        full_text_parts: list[str] = []
        text_buffer = StreamingTextBuffer()
        try:
            async with self._scheduler.acquire(self._config.tts_model) as adapter:
                if not isinstance(adapter, TTSAdapter):
                    await self._fail_response(
                        f"model {self._config.tts_model!r} is not a TTS adapter"
                    )
                    return

                adapter_cap = int(getattr(adapter.info(), "max_input_chars", 0) or 0)
                while True:
                    item = await stream.queue.get()
                    if item is _RESPONSE_STREAM_END:
                        break
                    item_text = str(item)
                    full_text_parts.append(item_text)
                    for text in text_buffer.push(item_text):
                        audio_started = await self._synthesize_text(
                            adapter,
                            text,
                            audio_started=audio_started,
                            max_input_chars=adapter_cap,
                        )

                for text in text_buffer.flush():
                    audio_started = await self._synthesize_text(
                        adapter,
                        text,
                        audio_started=audio_started,
                        max_input_chars=adapter_cap,
                    )






                full_text = "".join(full_text_parts).strip()
                if full_text:
                    self._pipeline.add_assistant_turn(full_text)

                await self._event_queue.put(TurnEvent(type=TurnEventType.TTS_COMPLETED))
                await self._emit({"type": WIRE_RESPONSE_DONE})
        except asyncio.CancelledError:


            raise
        except Exception as exc:
            logger.exception("TTS synthesis failed")
            await self._fail_response(str(exc))
        finally:
            if self._response_stream is stream:
                self._response_stream = None

    async def _synthesize_text(
        self,
        adapter: TTSAdapter,
        text: str,
        *,
        audio_started: bool,
        max_input_chars: int,
    ) -> bool:
        for chunk_text in split_for_tts(text, max_chars=max_input_chars):
            async for chunk in adapter.synthesize(
                chunk_text,
                voice=self._config.voice,
                language=self._config.language,
            ):
                if chunk.is_final and not chunk.audio:
                    continue
                if not audio_started:
                    audio_started = True
                    await self._event_queue.put(TurnEvent(
                        type=TurnEventType.TTS_AUDIO_STARTED,
                    ))
                await self._handle_tts_chunk(chunk.audio, chunk.sample_rate)

        return audio_started

    async def _fail_response(self, message: str) -> None:
        await self._emit({"type": WIRE_ERROR, "message": message})
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
        if self._paused:
            self._pending_audio.append((encoded_audio, output_sample_rate))
            return
        await self._emit({
            "type": WIRE_AUDIO_DELTA,
            "audio": base64.b64encode(encoded_audio).decode("ascii"),
            "sample_rate": output_sample_rate,
            "audio_format": "pcm16",
        })





    async def _execute(self, action: TurnAction) -> None:
        if action.type == TurnActionType.PAUSE_OUTPUT:
            self._paused = True

        elif action.type == TurnActionType.RESUME_OUTPUT:
            pending, self._pending_audio = self._pending_audio, []
            self._paused = False


            cooldown_s = self._config.policy.stable_speaking_min_ms / 1000.0
            self._flutter_cooldown_until = time.monotonic() + cooldown_s
            for audio, sample_rate in pending:
                await self._emit({
                    "type": WIRE_AUDIO_DELTA,
                    "audio": base64.b64encode(audio).decode("ascii"),
                    "sample_rate": sample_rate,
                    "audio_format": "pcm16",
                })

        elif action.type == TurnActionType.FLUSH_OUTPUT:
            self._pending_audio = []

        elif action.type == TurnActionType.STOP_TTS:
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await self._tts_task
            self._tts_task = None
            self._response_stream = None

        elif action.type == TurnActionType.CANCEL_RESPONSE:
            await self._emit({"type": WIRE_RESPONSE_CANCELLED})

        elif action.type == TurnActionType.START_TIMER:
            key = action.payload["key"]
            duration_ms = int(action.payload["duration_ms"])
            await self._cancel_timer(key)
            self._timers[key] = asyncio.create_task(self._timer_task(key, duration_ms))

        elif action.type == TurnActionType.CANCEL_TIMER:
            await self._cancel_timer(action.payload["key"])

    async def _timer_task(self, key: str, duration_ms: int) -> None:
        try:
            await asyncio.sleep(duration_ms / 1000.0)
        except asyncio.CancelledError:
            return

        if self._timers.get(key) is not asyncio.current_task():
            return
        self._timers.pop(key, None)

        if key == TimerKey.CONFIRM_INTERRUPT.value:
            await self._evaluate_interrupt_candidate()
            return

        await self._event_queue.put(TurnEvent(
            type=TurnEventType.TIMER_ELAPSED,
            payload={"key": key},
        ))

    def _has_active_timer(self, key: str) -> bool:
        task = self._timers.get(key)
        return task is not None and not task.done()

    def _transcript_commit_delay_ms(self) -> int:
        return max(
            0,
            min(
                self._config.policy.max_endpointing_delay_ms,
                TRANSCRIPT_CONTINUATION_COMMIT_MS,
            ),
        )

    async def _evaluate_interrupt_candidate(self) -> None:
        """Consult the classifier before confirming a barge-in.

        Classifier signals:
          * how long VAD has been "active" since the confirm window began
          * the last user turn's EOU probability (conversational context)
          * the most recent N ms of audio (so the classifier can detect cases
            where the user's voice has decayed but Silero's silence padding
            hasn't emitted SpeechStopped yet — e.g. "mhmm" backchannels)
          * a partial transcript of the PAUSED-window audio — lets classifiers
            that care (with user-supplied keyword sets, per-language intent
            models, etc.) short-circuit the audio-only heuristics.

        Decision:
          * real interrupt → TIMER_ELAPSED → state machine → INTERRUPTED
          * backchannel    → synthetic SPEECH_STOPPED → state machine resumes
                             SPEAKING, anti-flutter cooldown arms automatically
        """
        vad_active_ms = 0
        if self._vad_started_at is not None:
            vad_active_ms = max(0, int((time.monotonic() - self._vad_started_at) * 1000))

        audio_tail: np.ndarray | None = None
        if vad_active_ms > 0 and self._audio_ring.size > 0:
            tail_samples = min(
                self._audio_ring.size,
                max(1, vad_active_ms * TARGET_SAMPLE_RATE // 1000),
            )
            audio_tail = self._audio_ring[-tail_samples:]

        partial_transcript = (
            self._latest_partial.text if self._latest_partial is not None else None
        )

        try:
            is_real = await self._config.interrupt_classifier.is_real_interrupt(
                audio_tail,
                partial_transcript,
                self._last_eou_probability,
                vad_active_ms,
                TARGET_SAMPLE_RATE,
            )
        except Exception:
            logger.exception("interrupt classifier raised; defaulting to real interrupt")
            is_real = True

        if is_real:
            await self._event_queue.put(TurnEvent(
                type=TurnEventType.TIMER_ELAPSED,
                payload={"key": TimerKey.CONFIRM_INTERRUPT.value},
            ))
        else:
            logger.debug(
                "classifier rejected barge-in (vad_active=%dms); resuming TTS",
                vad_active_ms,
            )


            self._vad_started_at = None
            await self._event_queue.put(TurnEvent(
                type=TurnEventType.SPEECH_STOPPED,
                payload={"reason": "backchannel"},
            ))

    async def _cancel_timer(self, key: str) -> None:
        task = self._timers.pop(key, None)
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task





    async def _emit(self, event: dict) -> None:
        if self._closed:
            return
        try:
            await self._on_event(event)
        except Exception:
            logger.exception("on_event handler raised")





    @property
    def state(self) -> TurnState:
        return self._sm.state

    @property
    def pending_audio_count(self) -> int:
        return len(self._pending_audio)
