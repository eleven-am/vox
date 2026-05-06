from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from vox.conversation import TurnPolicy
from vox.conversation.session import (
    WIRE_AUDIO_DELTA,
    WIRE_ERROR,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_COMMITTED,
    WIRE_RESPONSE_CREATED,
    WIRE_RESPONSE_DONE,
    WIRE_SPEECH_STARTED,
    WIRE_SPEECH_STOPPED,
    WIRE_STATE_CHANGED,
    WIRE_TRANSCRIPT_DONE,
    ConversationConfig,
    ConversationSession,
)
from vox.operations.errors import (
    InvalidConfigError,
    SessionAlreadyConfiguredError,
    SessionNotConfiguredError,
)
from vox.streaming.types import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationSessionConfig:
    stt_model: str
    tts_model: str
    voice: str | None = None
    language: str = "en"
    sample_rate: int = TARGET_SAMPLE_RATE
    policy: TurnPolicy | None = None


@dataclass(frozen=True)
class ConvSessionCreatedEvent:
    config: ConversationSessionConfig


@dataclass(frozen=True)
class ConvSpeechStartedEvent:
    timestamp_ms: int


@dataclass(frozen=True)
class ConvSpeechStoppedEvent:
    timestamp_ms: int


@dataclass(frozen=True)
class ConvTranscriptDoneEvent:
    transcript: str
    language: str
    start_ms: int
    end_ms: int
    eou_probability: float | None
    entities: tuple[dict, ...]
    topics: tuple[str, ...]
    words: tuple[dict, ...]


@dataclass(frozen=True)
class ConvResponseCreatedEvent:
    pass


@dataclass(frozen=True)
class ConvAudioDeltaEvent:
    audio_b64: str
    sample_rate: int
    audio_format: str


@dataclass(frozen=True)
class ConvResponseDoneEvent:
    pass


@dataclass(frozen=True)
class ConvResponseCancelledEvent:
    pass


@dataclass(frozen=True)
class ConvResponseCommittedEvent:
    pass


@dataclass(frozen=True)
class ConvStateChangedEvent:
    state: str
    previous_state: str


@dataclass(frozen=True)
class ConvErrorEvent:
    message: str


@dataclass(frozen=True)
class ConvDoneEvent:
    pass


ConvEvent = (
    ConvSessionCreatedEvent
    | ConvSpeechStartedEvent
    | ConvSpeechStoppedEvent
    | ConvTranscriptDoneEvent
    | ConvResponseCreatedEvent
    | ConvAudioDeltaEvent
    | ConvResponseDoneEvent
    | ConvResponseCancelledEvent
    | ConvResponseCommittedEvent
    | ConvStateChangedEvent
    | ConvErrorEvent
    | ConvDoneEvent
)


def parse_session_update(payload: dict) -> ConversationSessionConfig:
    sess = payload.get("session") or payload
    stt_model = sess.get("stt_model") or sess.get("input_audio_transcription", {}).get("model")
    tts_model = sess.get("tts_model") or sess.get("output_audio_generation", {}).get("model")
    if not stt_model:
        raise InvalidConfigError("session.update requires 'stt_model'")
    if not tts_model:
        raise InvalidConfigError("session.update requires 'tts_model'")

    policy_in = sess.get("turn_policy") or sess.get("policy") or {}
    policy_kwargs = {}
    for field_name in (
        "allow_interrupt_while_speaking",
        "min_interrupt_duration_ms",
        "max_endpointing_delay_ms",
        "stable_speaking_min_ms",
    ):
        if field_name in policy_in:
            policy_kwargs[field_name] = policy_in[field_name]
    policy = TurnPolicy(**policy_kwargs) if policy_kwargs else TurnPolicy()

    return ConversationSessionConfig(
        stt_model=stt_model,
        tts_model=tts_model,
        voice=sess.get("voice"),
        language=sess.get("language", "en") or "en",
        sample_rate=int(sess.get("sample_rate") or TARGET_SAMPLE_RATE),
        policy=policy,
    )


def _wire_event_to_session_event(event: dict) -> ConvEvent | None:
    t = event.get("type")
    if t == WIRE_SPEECH_STARTED:
        return ConvSpeechStartedEvent(timestamp_ms=int(event.get("timestamp_ms") or 0))
    if t == WIRE_SPEECH_STOPPED:
        return ConvSpeechStoppedEvent(timestamp_ms=int(event.get("timestamp_ms") or 0))
    if t == WIRE_TRANSCRIPT_DONE:
        return ConvTranscriptDoneEvent(
            transcript=str(event.get("transcript", "")),
            language=str(event.get("language", "")),
            start_ms=int(event.get("start_ms") or 0),
            end_ms=int(event.get("end_ms") or 0),
            eou_probability=(
                float(event["eou_probability"])
                if event.get("eou_probability") is not None
                else None
            ),
            entities=tuple(event.get("entities") or ()),
            topics=tuple(event.get("topics") or ()),
            words=tuple(event.get("words") or ()),
        )
    if t == WIRE_RESPONSE_CREATED:
        return ConvResponseCreatedEvent()
    if t == WIRE_AUDIO_DELTA:
        return ConvAudioDeltaEvent(
            audio_b64=str(event.get("audio") or ""),
            sample_rate=int(event.get("sample_rate") or 0),
            audio_format=str(event.get("audio_format") or "pcm16"),
        )
    if t == WIRE_RESPONSE_DONE:
        return ConvResponseDoneEvent()
    if t == WIRE_RESPONSE_CANCELLED:
        return ConvResponseCancelledEvent()
    if t == WIRE_RESPONSE_COMMITTED:
        return ConvResponseCommittedEvent()
    if t == WIRE_STATE_CHANGED:
        return ConvStateChangedEvent(
            state=str(event.get("state", "")),
            previous_state=str(event.get("previous_state", "")),
        )
    if t == WIRE_ERROR:
        return ConvErrorEvent(message=str(event.get("message", "")))
    logger.debug("unmapped conversation wire event: %s", t)
    return None


class ConversationOrchestrator:

    def __init__(self, *, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._session: ConversationSession | None = None
        self._config: ConversationSessionConfig | None = None
        self._events: asyncio.Queue[ConvEvent] = asyncio.Queue()
        self._closed = False

    @property
    def config(self) -> ConversationSessionConfig | None:
        return self._config

    async def start_session(self, config: ConversationSessionConfig) -> None:
        if self._session is not None:
            raise SessionAlreadyConfiguredError()
        policy = config.policy or TurnPolicy()
        engine_config = ConversationConfig(
            stt_model=config.stt_model,
            tts_model=config.tts_model,
            voice=config.voice,
            language=config.language,
            sample_rate=config.sample_rate,
            policy=policy,
        )
        self._config = config
        self._session = ConversationSession(
            scheduler=self._scheduler,
            config=engine_config,
            on_event=self._on_engine_event,
        )
        await self._session.start()
        await self._events.put(ConvSessionCreatedEvent(config=config))

    async def ingest_pcm16(self, pcm16: bytes, sample_rate: int | None = None) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.ingest_audio(pcm16, sample_rate=sample_rate)

    async def start_response(self) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.start_response_stream()

    async def append_response_text(self, text: str) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.append_response_text(text)

    async def commit_response(self) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.commit_response_stream()

    async def cancel_response(self) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.cancel_response()

    async def report_error(self, message: str) -> None:
        await self._events.put(ConvErrorEvent(message=message))

    async def end_of_stream(self) -> None:
        if self._session is not None:
            with suppress(Exception):
                await self._session.commit_response_stream()
            with suppress(Exception):
                await self._session.wait_until_settled()
        await self._events.put(ConvDoneEvent())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            with suppress(Exception):
                await self._session.close()

    async def events(self) -> AsyncIterator[ConvEvent]:
        while True:
            event = await self._events.get()
            yield event
            if isinstance(event, ConvDoneEvent):
                return

    async def _on_engine_event(self, event: dict) -> None:
        mapped = _wire_event_to_session_event(event)
        if mapped is None:
            return
        await self._events.put(mapped)


def serialize_session_config(config: ConversationSessionConfig) -> dict:
    policy = config.policy or TurnPolicy()
    return {
        "stt_model": config.stt_model,
        "tts_model": config.tts_model,
        "voice": config.voice,
        "language": config.language,
        "sample_rate": config.sample_rate,
        "output_sample_rate": config.sample_rate,
        "output_audio_format": "pcm16",
        "turn_policy": {
            "allow_interrupt_while_speaking": policy.allow_interrupt_while_speaking,
            "min_interrupt_duration_ms": policy.min_interrupt_duration_ms,
            "max_endpointing_delay_ms": policy.max_endpointing_delay_ms,
            "stable_speaking_min_ms": policy.stable_speaking_min_ms,
        },
    }
