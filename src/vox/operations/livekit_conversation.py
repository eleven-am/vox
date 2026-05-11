from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import numpy as np

from vox.conversation import TurnPolicy
from vox.conversation.text_buffer import split_for_tts
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.cloned_voices import resolve_voice_request
from vox.core.ner import annotate, entity_to_dict
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationSessionConfig,
    ConvEvent,
    ConvInterruptionDetectedEvent,
    ConvInterruptionFalsePositiveEvent,
    ConvResponseCancelledEvent,
    ConvResponseCommittedEvent,
    ConvResponseCreatedEvent,
    ConvResponseDoneEvent,
    ConvSessionCreatedEvent,
    ConvSpeechStartedEvent,
    ConvSpeechStoppedEvent,
    ConvStateChangedEvent,
    ConvTranscriptDoneEvent,
    ConvTurnEouPredictedEvent,
)
from vox.operations.errors import SessionAlreadyConfiguredError, SessionNotConfiguredError, WrongModelTypeError
from vox.streaming.codecs import float32_to_pcm16, pcm16_to_float32, resample_audio
from vox.streaming.types import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)


class LiveKitConversation:
    """LiveKit-backed media conversation for Vox RTC sessions.

    Browser audio and assistant playback flow through LiveKit. Vox keeps only the
    developer control channel and event normalization here.
    """

    def __init__(
        self,
        *,
        scheduler: Any,
        store: Any | None,
        livekit_url: str,
        room: str,
        agent_token: str,
    ) -> None:
        self._scheduler = scheduler
        self._store = store
        self._livekit_url = livekit_url
        self._room_name = room
        self._agent_token = agent_token
        self._config: ConversationSessionConfig | None = None
        self._events: asyncio.Queue[ConvEvent] = asyncio.Queue()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

        self._lk_room: Any | None = None
        self._lk_session: Any | None = None
        self._lk_agent: Any | None = None

        self._response_counter = 0
        self._pending_response_id: str | None = None
        self._pending_text_parts: list[str] = []
        self._active_response_id: str | None = None
        self._active_handle: Any | None = None
        self._cancelled_response_ids: set[str] = set()
        self._clear_sent_response_ids: set[str] = set()
        self._last_user_speech_started_at: float | None = None
        self._last_agent_state = "idle"
        self._last_user_state = "listening"
        self._session_started_at = time.time()

    @property
    def config(self) -> ConversationSessionConfig | None:
        return self._config

    async def start_session(self, config: ConversationSessionConfig) -> None:
        if self._config is not None:
            raise SessionAlreadyConfiguredError()
        self._loop = asyncio.get_running_loop()
        self._config = config
        self._session_started_at = time.time()

        rtc, agents, silero = _load_livekit_modules()
        vad = silero.VAD.load(
            sample_rate=16_000,
            min_speech_duration=0.05,
            min_silence_duration=max(0.2, _policy(config).min_endpointing_delay_ms / 1000.0),
            prefix_padding_duration=0.3,
        )
        turn_handling = _turn_handling_options(config)
        stt = _VoxLiveKitSTT(
            scheduler=self._scheduler,
            model=config.stt_model,
            language=config.language,
        )._impl
        tts = _VoxLiveKitTTS(
            scheduler=self._scheduler,
            store=self._store,
            model=config.tts_model,
            voice=config.voice,
            language=config.language,
            sample_rate=config.sample_rate,
        )._impl
        self._lk_room = rtc.Room()
        self._lk_session = agents.AgentSession(
            stt=stt,
            vad=vad,
            tts=tts,
            llm=None,
            turn_handling=turn_handling,
            aec_warmup_duration=3.0,
        )
        self._wire_livekit_events(self._lk_session)
        self._lk_agent = agents.Agent(
            instructions=(
                "You are a Vox media bridge. Do not generate replies by yourself; "
                "only speak text supplied by the external developer control channel."
            ),
            llm=None,
            stt=stt,
            vad=vad,
            tts=tts,
            turn_handling=turn_handling,
        )
        await self._lk_room.connect(self._livekit_url, self._agent_token)
        await self._lk_session.start(self._lk_agent, room=self._lk_room, record=False)
        await self._events.put(ConvSessionCreatedEvent(config=config))

    async def ingest_pcm16(self, _pcm16: bytes, sample_rate: int | None = None) -> None:
        raise SessionNotConfiguredError()

    async def start_response(self) -> None:
        self._require_started()
        if self._pending_response_id is not None:
            await self._events.put(ConvErrorEvent(message="response already in flight"))
            return
        self._response_counter += 1
        response_id = f"resp_{self._response_counter}"
        self._pending_response_id = response_id
        self._pending_text_parts = []
        await self._events.put(ConvResponseCreatedEvent(response_id=response_id))

    async def append_response_text(self, text: str) -> None:
        self._require_started()
        if self._pending_response_id is None:
            await self.start_response()
        self._pending_text_parts.append(text)

    async def commit_response(self) -> None:
        self._require_started()
        if self._pending_response_id is None:
            await self.start_response()
        response_id = self._pending_response_id or ""
        text = "".join(self._pending_text_parts).strip()
        self._pending_response_id = None
        self._pending_text_parts = []
        if not text:
            await self._events.put(ConvErrorEvent(message="response.commit requires response text"))
            return

        previous_id = self._active_response_id
        previous_handle = self._active_handle
        if previous_id and previous_handle is not None and not previous_handle.done():
            await self._clear_and_cancel_active(previous_id, previous_handle)

        assert self._lk_session is not None
        self._active_response_id = response_id
        self._cancelled_response_ids.discard(response_id)
        self._clear_sent_response_ids.discard(response_id)
        await self._events.put(ConvResponseCommittedEvent(response_id=response_id))
        handle = self._lk_session.say(text, allow_interruptions=True, add_to_chat_ctx=True)
        self._active_handle = handle
        handle.add_done_callback(lambda done_handle: self._schedule(self._speech_done(done_handle, response_id)))

    async def cancel_response(self) -> None:
        self._require_started()
        response_id = self._active_response_id or self._pending_response_id or ""
        handle = self._active_handle
        self._pending_response_id = None
        self._pending_text_parts = []
        if handle is not None and not handle.done():
            with suppress(Exception):
                handle.interrupt(force=True)
        if response_id:
            await self._events.put(ConvAudioClearEvent(response_id=response_id))
            await self._events.put(ConvResponseCancelledEvent(response_id=response_id))
            self._cancelled_response_ids.add(response_id)
        else:
            await self._events.put(ConvResponseCancelledEvent(response_id=""))

    async def report_error(self, message: str) -> None:
        await self._events.put(ConvErrorEvent(message=message))

    async def end_of_stream(self) -> None:
        await self._events.put(ConvDoneEvent())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._active_handle is not None and not self._active_handle.done():
            with suppress(Exception):
                self._active_handle.interrupt(force=True)
        if self._lk_session is not None:
            with suppress(Exception):
                await self._lk_session.aclose()
        if self._lk_room is not None:
            with suppress(Exception):
                await self._lk_room.disconnect()

    async def events(self) -> AsyncIterator[ConvEvent]:
        while True:
            event = await self._events.get()
            yield event
            if isinstance(event, ConvDoneEvent):
                return

    def _wire_livekit_events(self, session: Any) -> None:
        @session.on("user_state_changed")
        def _on_user_state_changed(event: Any) -> None:
            self._schedule(self._handle_user_state(event.old_state, event.new_state, event.created_at))

        @session.on("agent_state_changed")
        def _on_agent_state_changed(event: Any) -> None:
            self._schedule(self._handle_agent_state(event.old_state, event.new_state))

        @session.on("user_input_transcribed")
        def _on_user_input_transcribed(event: Any) -> None:
            if bool(getattr(event, "is_final", False)):
                self._schedule(self._handle_transcript(event))

        @session.on("agent_false_interruption")
        def _on_agent_false_interruption(event: Any) -> None:
            self._schedule(self._handle_false_interruption(event))

        @session.on("error")
        def _on_error(event: Any) -> None:
            error = getattr(event, "error", event)
            self._schedule(self._events.put(ConvErrorEvent(message=str(error))))

    def _schedule(self, coro: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(lambda: asyncio.create_task(coro))

    async def _handle_user_state(self, old_state: str, new_state: str, created_at: float) -> None:
        self._last_user_state = new_state
        if new_state == "speaking":
            self._last_user_speech_started_at = created_at
            await self._events.put(ConvSpeechStartedEvent(timestamp_ms=self._relative_ms(created_at)))
            response_id = self._active_response_id
            handle = self._active_handle
            if (
                response_id
                and handle is not None
                and not handle.done()
                and self._last_agent_state == "speaking"
            ):
                await self._emit_interruption_started(response_id)
        elif old_state == "speaking":
            await self._events.put(ConvSpeechStoppedEvent(timestamp_ms=self._relative_ms(created_at)))

    async def _handle_agent_state(self, old_state: str, new_state: str) -> None:
        previous = _map_agent_state(old_state)
        state = _map_agent_state(new_state)
        self._last_agent_state = new_state
        await self._events.put(ConvStateChangedEvent(state=state, previous_state=previous))

    async def _handle_transcript(self, event: Any) -> None:
        transcript = str(getattr(event, "transcript", "") or "").strip()
        if not transcript:
            return
        language = str(getattr(event, "language", None) or (self._config.language if self._config else "en"))
        end_ms = self._relative_ms(getattr(event, "created_at", time.time()))
        start_ms = self._relative_ms(self._last_user_speech_started_at) if self._last_user_speech_started_at else 0
        if end_ms < start_ms:
            end_ms = start_ms
        entities, topics = annotate(transcript, language)
        await self._events.put(ConvTurnEouPredictedEvent(
            probability=1.0,
            threshold=0.5,
            decision="complete",
            action="commit",
            delay_ms=0,
            turn_detector="livekit",
            start_ms=start_ms,
            end_ms=end_ms,
        ))
        await self._events.put(ConvTranscriptDoneEvent(
            transcript=transcript,
            language=language,
            start_ms=start_ms,
            end_ms=end_ms,
            eou_probability=1.0,
            entities=tuple(entity_to_dict(entity) for entity in entities),
            topics=tuple(topics),
            words=(),
        ))

    async def _handle_false_interruption(self, _event: Any) -> None:
        response_id = self._active_response_id or ""
        if response_id:
            await self._events.put(ConvInterruptionFalsePositiveEvent(
                response_id=response_id,
                vad_active_ms=0,
                partial_transcript=None,
            ))

    async def _emit_interruption_started(self, response_id: str) -> None:
        if response_id not in self._clear_sent_response_ids:
            await self._events.put(ConvAudioClearEvent(response_id=response_id))
            self._clear_sent_response_ids.add(response_id)
        await self._events.put(ConvInterruptionDetectedEvent(
            response_id=response_id,
            vad_active_ms=0,
            partial_transcript=None,
        ))
        await self._events.put(ConvStateChangedEvent(state="paused", previous_state="speaking"))

    async def _clear_and_cancel_active(self, response_id: str, handle: Any) -> None:
        if response_id not in self._clear_sent_response_ids:
            await self._events.put(ConvAudioClearEvent(response_id=response_id))
            self._clear_sent_response_ids.add(response_id)
        if not handle.done():
            with suppress(Exception):
                handle.interrupt(force=True)
        if response_id not in self._cancelled_response_ids:
            await self._events.put(ConvResponseCancelledEvent(response_id=response_id))
            self._cancelled_response_ids.add(response_id)

    async def _speech_done(self, handle: Any, response_id: str) -> None:
        interrupted = bool(getattr(handle, "interrupted", False))
        if interrupted and response_id not in self._cancelled_response_ids:
            if response_id not in self._clear_sent_response_ids:
                await self._events.put(ConvAudioClearEvent(response_id=response_id))
                self._clear_sent_response_ids.add(response_id)
            await self._events.put(ConvResponseCancelledEvent(response_id=response_id))
            self._cancelled_response_ids.add(response_id)
        await self._events.put(ConvResponseDoneEvent(response_id=response_id))
        if self._active_response_id == response_id:
            self._active_response_id = None
            self._active_handle = None

    def _relative_ms(self, created_at: float | None) -> int:
        if created_at is None:
            return 0
        return max(0, int((created_at - self._session_started_at) * 1000))

    def _require_started(self) -> None:
        if self._config is None or self._lk_session is None:
            raise SessionNotConfiguredError()


class _VoxLiveKitSTT:
    def __init__(self, *, scheduler: Any, model: str, language: str) -> None:
        rtc, agents, _ = _load_livekit_modules()
        self._rtc = rtc
        self._agents = agents
        self._scheduler = scheduler
        self._model = model
        self._language = language or "en"
        self._impl = self._build_impl()

    def _build_impl(self) -> Any:
        rtc = self._rtc
        agents = self._agents
        scheduler = self._scheduler
        model = self._model
        default_language = self._language

        class VoxSTT(agents.stt.STT):
            def __init__(self) -> None:
                super().__init__(
                    capabilities=agents.stt.STTCapabilities(
                        streaming=False,
                        interim_results=False,
                    )
                )

            @property
            def model(self) -> str:
                return model

            @property
            def provider(self) -> str:
                return "vox"

            async def _recognize_impl(self, buffer: Any, *, language: Any, conn_options: Any) -> Any:
                frames = buffer if isinstance(buffer, list) else [buffer]
                frames = [frame for frame in frames if frame is not None]
                if not frames:
                    return agents.stt.SpeechEvent(
                        type=agents.stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=uuid.uuid4().hex,
                        alternatives=[
                            agents.stt.SpeechData(language=default_language, text="", confidence=0.0)
                        ],
                    )
                combined = rtc.combine_audio_frames(frames)
                pcm = bytes(memoryview(combined.data).cast("B"))
                audio = pcm16_to_float32(pcm)
                source_rate = int(combined.sample_rate or TARGET_SAMPLE_RATE)
                if source_rate != TARGET_SAMPLE_RATE:
                    audio = resample_audio(audio, source_rate, TARGET_SAMPLE_RATE)
                language_code = language if isinstance(language, str) and language else default_language
                async with scheduler.acquire(model) as adapter:
                    if not isinstance(adapter, STTAdapter):
                        raise WrongModelTypeError(model, "STT")
                    result = await asyncio.to_thread(
                        adapter.transcribe,
                        audio,
                        language=language_code,
                        word_timestamps=False,
                    )
                duration_s = (result.duration_ms / 1000.0) if result.duration_ms else (
                    audio.size / float(TARGET_SAMPLE_RATE)
                )
                return agents.stt.SpeechEvent(
                    type=agents.stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=uuid.uuid4().hex,
                    alternatives=[
                        agents.stt.SpeechData(
                            language=result.language or language_code,
                            text=result.text or "",
                            start_time=0.0,
                            end_time=duration_s,
                            confidence=1.0,
                        )
                    ],
                )

        return VoxSTT()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


class _VoxLiveKitTTS:
    def __init__(
        self,
        *,
        scheduler: Any,
        store: Any | None,
        model: str,
        voice: str | None,
        language: str,
        sample_rate: int,
    ) -> None:
        _, agents, _ = _load_livekit_modules()
        self._agents = agents
        self._impl = self._build_impl(
            scheduler=scheduler,
            store=store,
            model=model,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
        )

    def _build_impl(
        self,
        *,
        scheduler: Any,
        store: Any | None,
        model: str,
        voice: str | None,
        language: str,
        sample_rate: int,
    ) -> Any:
        agents = self._agents
        target_sample_rate = int(sample_rate or TARGET_SAMPLE_RATE)

        class VoxTTS(agents.tts.TTS):
            def __init__(self) -> None:
                super().__init__(
                    capabilities=agents.tts.TTSCapabilities(streaming=False),
                    sample_rate=target_sample_rate,
                    num_channels=1,
                )

            @property
            def model(self) -> str:
                return model

            @property
            def provider(self) -> str:
                return "vox"

            def synthesize(self, text: str, *, conn_options: Any = None) -> Any:
                return VoxTTSChunkedStream(
                    tts=self,
                    input_text=text,
                    conn_options=conn_options or agents.DEFAULT_API_CONNECT_OPTIONS,
                )

        class VoxTTSChunkedStream(agents.tts.ChunkedStream):
            async def _run(self, output_emitter: Any) -> None:
                output_emitter.initialize(
                    request_id=uuid.uuid4().hex,
                    sample_rate=target_sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                    frame_size_ms=40,
                )
                async with scheduler.acquire(model) as adapter:
                    if not isinstance(adapter, TTSAdapter):
                        raise WrongModelTypeError(model, "TTS")
                    resolved_voice = voice
                    resolved_language = language
                    reference_audio = None
                    reference_text = None
                    if store is not None:
                        resolved_voice, resolved_language, reference_audio, reference_text = resolve_voice_request(
                            adapter,
                            store,
                            voice,
                            language,
                        )
                    max_chars = int(getattr(adapter.info(), "max_input_chars", 0) or 0)
                    for part in split_for_tts(self.input_text, max_chars=max_chars):
                        async for chunk in adapter.synthesize(
                            part,
                            voice=resolved_voice,
                            language=resolved_language,
                            reference_audio=reference_audio,
                            reference_text=reference_text,
                        ):
                            if not chunk.audio:
                                continue
                            audio = np.frombuffer(chunk.audio, dtype=np.float32)
                            if audio.size == 0:
                                continue
                            if chunk.sample_rate != target_sample_rate:
                                audio = resample_audio(audio, chunk.sample_rate, target_sample_rate)
                            output_emitter.push(float32_to_pcm16(audio))
                output_emitter.flush()

        return VoxTTS()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


def _load_livekit_modules() -> tuple[Any, Any, Any]:
    try:
        from livekit import agents, rtc
        from livekit.plugins import silero
    except ImportError as exc:
        raise RuntimeError(
            "LiveKit RTC requires livekit-agents and livekit-plugins-silero to be installed"
        ) from exc
    return rtc, agents, silero


def _policy(config: ConversationSessionConfig) -> TurnPolicy:
    return config.policy or TurnPolicy()


def _turn_handling_options(config: ConversationSessionConfig) -> dict[str, Any]:
    policy = _policy(config)
    min_endpoint = max(0.05, policy.min_endpointing_delay_ms / 1000.0)
    max_endpoint = max(min_endpoint, policy.max_endpointing_delay_ms / 1000.0)
    return {
        "turn_detection": "vad",
        "endpointing": {
            "mode": "dynamic" if policy.dynamic_endpointing else "fixed",
            "min_delay": min_endpoint,
            "max_delay": max_endpoint,
        },
        "interruption": {
            "enabled": bool(policy.allow_interrupt_while_speaking),
            "mode": "vad",
            "min_duration": max(0.05, policy.speaking_interrupt_min_duration_ms / 1000.0),
            "min_words": int(policy.speaking_interrupt_min_words),
            "false_interruption_timeout": max(0.1, policy.false_interruption_timeout_ms / 1000.0),
            "resume_false_interruption": True,
            "discard_audio_if_uninterruptible": True,
        },
        "preemptive_generation": {"enabled": False},
    }


def _map_agent_state(state: str) -> str:
    if state in {"initializing", "listening"}:
        return "idle"
    return state
