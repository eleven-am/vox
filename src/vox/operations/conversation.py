from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from vox.conversation import TurnPolicy
from vox.conversation.profiles import (
    DEFAULT_TURN_PROFILE,
    resolve_turn_policy,
)
from vox.conversation.session import (
    WIRE_AUDIO_CLEAR,
    WIRE_AUDIO_DELTA,
    WIRE_ERROR,
    WIRE_INTERRUPTION_DETECTED,
    WIRE_INTERRUPTION_FALSE_POSITIVE,
    WIRE_RESPONSE_CANCELLED,
    WIRE_RESPONSE_COMMITTED,
    WIRE_RESPONSE_CREATED,
    WIRE_RESPONSE_DONE,
    WIRE_SPEECH_STARTED,
    WIRE_SPEECH_STOPPED,
    WIRE_STATE_CHANGED,
    WIRE_TRANSCRIPT_DELTA,
    WIRE_TRANSCRIPT_DONE,
    WIRE_TURN_EOU_PREDICTED,
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

WIRE_SESSION_CREATED = "session.created"


@dataclass(frozen=True)
class ConversationSessionConfig:
    stt_model: str
    tts_model: str
    voice: str | None = None
    language: str = "en"
    sample_rate: int = TARGET_SAMPLE_RATE
    turn_profile: str = DEFAULT_TURN_PROFILE
    vad_backend: str = "silero"
    turn_detector: str = "livekit"
    policy: TurnPolicy | None = None
    include_word_timestamps: bool = False


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
class ConvTranscriptDeltaEvent:
    delta: str
    start_ms: int
    end_ms: int


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
    response_id: str = ""


@dataclass(frozen=True)
class ConvAudioDeltaEvent:
    audio_b64: str
    sample_rate: int
    audio_format: str
    response_id: str = ""
    sequence: int = 0


@dataclass(frozen=True)
class ConvAudioClearEvent:
    response_id: str = ""


@dataclass(frozen=True)
class ConvResponseDoneEvent:
    response_id: str = ""


@dataclass(frozen=True)
class ConvResponseCancelledEvent:
    response_id: str = ""


@dataclass(frozen=True)
class ConvResponseCommittedEvent:
    response_id: str = ""


@dataclass(frozen=True)
class ConvInterruptionDetectedEvent:
    response_id: str
    vad_active_ms: int
    partial_transcript: str | None


@dataclass(frozen=True)
class ConvInterruptionFalsePositiveEvent:
    response_id: str
    vad_active_ms: int
    partial_transcript: str | None
    reason: str | None = None


@dataclass(frozen=True)
class ConvTurnEouPredictedEvent:
    probability: float
    threshold: float
    decision: str
    action: str
    delay_ms: int
    turn_detector: str
    start_ms: int
    end_ms: int


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
    | ConvTranscriptDeltaEvent
    | ConvTranscriptDoneEvent
    | ConvResponseCreatedEvent
    | ConvAudioDeltaEvent
    | ConvAudioClearEvent
    | ConvResponseDoneEvent
    | ConvResponseCancelledEvent
    | ConvResponseCommittedEvent
    | ConvInterruptionDetectedEvent
    | ConvInterruptionFalsePositiveEvent
    | ConvTurnEouPredictedEvent
    | ConvStateChangedEvent
    | ConvErrorEvent
    | ConvDoneEvent
)


def serialize_conversation_event(event: ConvEvent) -> dict | None:
    if isinstance(event, ConvSessionCreatedEvent):
        return {
            "type": WIRE_SESSION_CREATED,
            "session": serialize_session_config(event.config),
        }
    if isinstance(event, ConvSpeechStartedEvent):
        return {"type": WIRE_SPEECH_STARTED, "timestamp_ms": event.timestamp_ms}
    if isinstance(event, ConvSpeechStoppedEvent):
        return {"type": WIRE_SPEECH_STOPPED, "timestamp_ms": event.timestamp_ms}
    if isinstance(event, ConvTranscriptDeltaEvent):
        return {
            "type": WIRE_TRANSCRIPT_DELTA,
            "delta": event.delta,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
        }
    if isinstance(event, ConvTranscriptDoneEvent):
        payload: dict = {
            "type": WIRE_TRANSCRIPT_DONE,
            "transcript": event.transcript,
            "language": event.language,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
        }
        if event.eou_probability is not None:
            payload["eou_probability"] = event.eou_probability
        if event.entities:
            payload["entities"] = list(event.entities)
        if event.topics:
            payload["topics"] = list(event.topics)
        if event.words:
            payload["words"] = list(event.words)
        return payload
    if isinstance(event, ConvResponseCreatedEvent):
        return {"type": WIRE_RESPONSE_CREATED, "response_id": event.response_id}
    if isinstance(event, ConvAudioDeltaEvent):
        return {
            "type": WIRE_AUDIO_DELTA,
            "audio": event.audio_b64,
            "sample_rate": event.sample_rate,
            "audio_format": event.audio_format,
            "response_id": event.response_id,
            "sequence": event.sequence,
        }
    if isinstance(event, ConvAudioClearEvent):
        return {"type": WIRE_AUDIO_CLEAR, "response_id": event.response_id}
    if isinstance(event, ConvResponseDoneEvent):
        return {"type": WIRE_RESPONSE_DONE, "response_id": event.response_id}
    if isinstance(event, ConvResponseCancelledEvent):
        return {"type": WIRE_RESPONSE_CANCELLED, "response_id": event.response_id}
    if isinstance(event, ConvResponseCommittedEvent):
        return {"type": WIRE_RESPONSE_COMMITTED, "response_id": event.response_id}
    if isinstance(event, ConvInterruptionDetectedEvent):
        return {
            "type": WIRE_INTERRUPTION_DETECTED,
            "response_id": event.response_id,
            "vad_active_ms": event.vad_active_ms,
            "partial_transcript": event.partial_transcript,
        }
    if isinstance(event, ConvInterruptionFalsePositiveEvent):
        payload_fp: dict = {
            "type": WIRE_INTERRUPTION_FALSE_POSITIVE,
            "response_id": event.response_id,
            "vad_active_ms": event.vad_active_ms,
            "partial_transcript": event.partial_transcript,
        }
        if event.reason:
            payload_fp["reason"] = event.reason
        return payload_fp
    if isinstance(event, ConvTurnEouPredictedEvent):
        return {
            "type": WIRE_TURN_EOU_PREDICTED,
            "probability": event.probability,
            "threshold": event.threshold,
            "decision": event.decision,
            "action": event.action,
            "delay_ms": event.delay_ms,
            "turn_detector": event.turn_detector,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
        }
    if isinstance(event, ConvStateChangedEvent):
        return {
            "type": WIRE_STATE_CHANGED,
            "state": event.state,
            "previous_state": event.previous_state,
        }
    if isinstance(event, ConvErrorEvent):
        return {"type": WIRE_ERROR, "message": event.message}
    return None


def parse_session_update(payload: dict) -> ConversationSessionConfig:
    sess = payload.get("session") or payload
    stt_model = sess.get("stt_model") or sess.get("input_audio_transcription", {}).get("model")
    tts_model = sess.get("tts_model") or sess.get("output_audio_generation", {}).get("model")
    if not stt_model:
        raise InvalidConfigError("session.update requires 'stt_model'")
    if not tts_model:
        raise InvalidConfigError("session.update requires 'tts_model'")

    turn_profile = str(sess.get("turn_profile") or sess.get("profile") or DEFAULT_TURN_PROFILE)
    policy_in = sess.get("turn_policy") or sess.get("policy") or {}
    policy_kwargs = {}
    for field_name in (
        "allow_interrupt_while_speaking",
        "min_interrupt_duration_ms",
        "max_endpointing_delay_ms",
        "stable_speaking_min_ms",
        "false_interruption_timeout_ms",
        "min_interrupt_words",
        "partial_interrupts",
        "dynamic_endpointing",
        "min_endpointing_delay_ms",
        "speaking_interrupt_min_duration_ms",
        "speaking_interrupt_min_words",
        "self_echo_min_words",
        "self_echo_min_overlap",
        "aec_warmup_ms",
        "backchannel_end_cooldown_ms",
        "vad_min_silence_ms",
    ):
        if field_name in policy_in:
            policy_kwargs[field_name] = policy_in[field_name]
    try:
        turn_profile, policy = resolve_turn_policy(turn_profile, policy_kwargs)
    except ValueError as exc:
        raise InvalidConfigError(str(exc)) from exc

    return ConversationSessionConfig(
        stt_model=stt_model,
        tts_model=tts_model,
        voice=sess.get("voice"),
        language=sess.get("language", "en") or "en",
        sample_rate=int(sess.get("sample_rate") or TARGET_SAMPLE_RATE),
        turn_profile=turn_profile,
        vad_backend=str(sess.get("vad_backend") or sess.get("vad") or "silero"),
        turn_detector=str(sess.get("turn_detector") or sess.get("eou_model") or "livekit"),
        policy=policy,
        include_word_timestamps=bool(sess.get("include_word_timestamps") or False),
    )


def parse_allow_interruptions(message: dict) -> bool:
    response = message.get("response")
    if isinstance(response, dict) and "allow_interruptions" in response:
        return bool(response["allow_interruptions"])
    if "allow_interruptions" in message:
        return bool(message["allow_interruptions"])
    return True


def parse_response_text(message: dict, preferred_key: str) -> str | None:
    response = message.get("response")
    if isinstance(response, dict):
        value = response.get(preferred_key) or response.get("text") or response.get("delta")
        if value is not None:
            return str(value)
    value = message.get(preferred_key) or message.get("text") or message.get("delta")
    if value is None:
        return None
    return str(value)


async def execute_conversation_command(
    orchestrator: ConversationOrchestrator,
    message: dict,
    *,
    allow_input_audio: bool = True,
    client_event_handler: Callable[[str, Any], Awaitable[None] | None] | None = None,
    require_config_message: str = "send session.update first",
    unknown_message_label: str = "unknown message type",
) -> None:
    msg_type = message.get("type")
    if not msg_type:
        raise InvalidConfigError("missing 'type' field")

    if msg_type == "session.update":
        await execute_conversation_session_update(
            orchestrator,
            parse_session_update(message),
        )
        return

    if msg_type == "client.event" and client_event_handler is not None:
        event_name, payload = parse_client_event_command(message)
        result = client_event_handler(event_name, payload)
        if result is not None:
            await result
        return

    if orchestrator.config is None:
        raise InvalidConfigError(require_config_message)

    if msg_type == "input_audio_buffer.append" and allow_input_audio:
        raw_pcm = message.get("audio_pcm16")
        audio_b64 = message.get("audio")
        if raw_pcm is not None:
            pcm = bytes(raw_pcm)
        elif audio_b64:
            try:
                pcm = base64.b64decode(audio_b64)
            except Exception as exc:  # noqa: BLE001
                raise InvalidConfigError(f"invalid base64 audio: {exc}") from exc
        else:
            raise InvalidConfigError("audio field required")
        sample_rate = int(message.get("sample_rate", 0)) or None
        await orchestrator.ingest_pcm16(pcm, sample_rate=sample_rate)
        return

    if msg_type == "response.start":
        allow_interruptions = parse_allow_interruptions(message)
        await orchestrator.start_response(allow_interruptions=allow_interruptions)
        return

    if msg_type == "response.delta":
        text = parse_response_text(message, "delta")
        if not text:
            raise InvalidConfigError("response.delta requires 'delta' text")
        allow_interruptions = parse_allow_interruptions(message)
        await orchestrator.append_response_text(text, allow_interruptions=allow_interruptions)
        return

    if msg_type == "response.commit":
        await orchestrator.commit_response()
        return

    if msg_type == "response.cancel":
        await orchestrator.cancel_response()
        return

    if msg_type == "response.replace_text":
        text = parse_response_text(message, "text")
        if not text:
            raise InvalidConfigError("response.replace_text requires 'text'")
        allow_interruptions = parse_allow_interruptions(message)
        await orchestrator.replace_response_text(text, allow_interruptions=allow_interruptions)
        return

    raise InvalidConfigError(f"{unknown_message_label}: {msg_type!r}")


async def execute_conversation_session_update(
    orchestrator: ConversationOrchestrator,
    config: ConversationSessionConfig,
    *,
    already_configured_message: str = "session already configured",
) -> None:
    try:
        await orchestrator.start_session(config)
    except SessionAlreadyConfiguredError as exc:
        raise InvalidConfigError(already_configured_message) from exc


def parse_client_event_command(message: Any) -> tuple[str, Any]:
    if not isinstance(message, dict):
        raise InvalidConfigError("client.event requires a JSON object")
    event_name = message.get("event")
    if not isinstance(event_name, str) or not event_name.strip():
        raise InvalidConfigError("client.event requires a non-empty string 'event'")
    return event_name.strip(), message.get("payload")


def _wire_event_to_session_event(event: dict) -> ConvEvent | None:
    t = event.get("type")
    if t == WIRE_SPEECH_STARTED:
        return ConvSpeechStartedEvent(timestamp_ms=int(event.get("timestamp_ms") or 0))
    if t == WIRE_SPEECH_STOPPED:
        return ConvSpeechStoppedEvent(timestamp_ms=int(event.get("timestamp_ms") or 0))
    if t == WIRE_TRANSCRIPT_DELTA:
        return ConvTranscriptDeltaEvent(
            delta=str(event.get("delta", "")),
            start_ms=int(event.get("start_ms") or 0),
            end_ms=int(event.get("end_ms") or 0),
        )
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
        return ConvResponseCreatedEvent(response_id=str(event.get("response_id") or ""))
    if t == WIRE_AUDIO_DELTA:
        return ConvAudioDeltaEvent(
            audio_b64=str(event.get("audio") or ""),
            sample_rate=int(event.get("sample_rate") or 0),
            audio_format=str(event.get("audio_format") or "pcm16"),
            response_id=str(event.get("response_id") or ""),
            sequence=int(event.get("sequence") or 0),
        )
    if t == WIRE_AUDIO_CLEAR:
        return ConvAudioClearEvent(response_id=str(event.get("response_id") or ""))
    if t == WIRE_RESPONSE_DONE:
        return ConvResponseDoneEvent(response_id=str(event.get("response_id") or ""))
    if t == WIRE_RESPONSE_CANCELLED:
        return ConvResponseCancelledEvent(response_id=str(event.get("response_id") or ""))
    if t == WIRE_RESPONSE_COMMITTED:
        return ConvResponseCommittedEvent(response_id=str(event.get("response_id") or ""))
    if t == WIRE_INTERRUPTION_DETECTED:
        return ConvInterruptionDetectedEvent(
            response_id=str(event.get("response_id") or ""),
            vad_active_ms=int(event.get("vad_active_ms") or 0),
            partial_transcript=(
                str(event["partial_transcript"])
                if event.get("partial_transcript") is not None
                else None
            ),
        )
    if t == WIRE_INTERRUPTION_FALSE_POSITIVE:
        return ConvInterruptionFalsePositiveEvent(
            response_id=str(event.get("response_id") or ""),
            vad_active_ms=int(event.get("vad_active_ms") or 0),
            partial_transcript=(
                str(event["partial_transcript"])
                if event.get("partial_transcript") is not None
                else None
            ),
            reason=str(event["reason"]) if event.get("reason") else None,
        )
    if t == WIRE_TURN_EOU_PREDICTED:
        return ConvTurnEouPredictedEvent(
            probability=float(event.get("probability") or 0.0),
            threshold=float(event.get("threshold") or 0.0),
            decision=str(event.get("decision") or ""),
            action=str(event.get("action") or ""),
            delay_ms=int(event.get("delay_ms") or 0),
            turn_detector=str(event.get("turn_detector") or ""),
            start_ms=int(event.get("start_ms") or 0),
            end_ms=int(event.get("end_ms") or 0),
        )
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

    def __init__(
        self,
        *,
        scheduler: Any,
        pace_response_done_to_audio: bool = False,
        audio_sink: Callable[[ConvAudioDeltaEvent], Awaitable[None]] | None = None,
        wait_for_output_playout: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._pace_response_done_to_audio = pace_response_done_to_audio
        self._audio_sink = audio_sink
        self._wait_for_output_playout = wait_for_output_playout
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
            vad_backend=config.vad_backend,
            turn_detector=config.turn_detector,
            policy=policy,
            turn_profile=config.turn_profile,
            include_word_timestamps=config.include_word_timestamps,
            pace_response_done_to_audio=self._pace_response_done_to_audio,
            wait_for_output_playout=self._wait_for_output_playout,
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

    async def start_response(self, *, allow_interruptions: bool = True) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.start_response_stream(allow_interruptions=allow_interruptions)

    async def append_response_text(self, text: str, *, allow_interruptions: bool = True) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.append_response_text(text, allow_interruptions=allow_interruptions)

    async def replace_response_text(self, text: str, *, allow_interruptions: bool = True) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.replace_response_text(text, allow_interruptions=allow_interruptions)

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

    async def end_of_stream(self, *, flush_response: bool = True) -> None:
        if self._session is not None and flush_response:
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
        if self._audio_sink is not None and isinstance(mapped, ConvAudioDeltaEvent):
            await self._audio_sink(mapped)
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
        "turn_profile": config.turn_profile,
        "vad_backend": config.vad_backend,
        "turn_detector": config.turn_detector,
        "include_word_timestamps": config.include_word_timestamps,
        "output_sample_rate": config.sample_rate,
        "output_audio_format": "pcm16",
        "turn_policy": {
            "allow_interrupt_while_speaking": policy.allow_interrupt_while_speaking,
            "min_interrupt_duration_ms": policy.min_interrupt_duration_ms,
            "max_endpointing_delay_ms": policy.max_endpointing_delay_ms,
            "stable_speaking_min_ms": policy.stable_speaking_min_ms,
            "false_interruption_timeout_ms": policy.false_interruption_timeout_ms,
            "min_interrupt_words": policy.min_interrupt_words,
            "partial_interrupts": policy.partial_interrupts,
            "dynamic_endpointing": policy.dynamic_endpointing,
            "min_endpointing_delay_ms": policy.min_endpointing_delay_ms,
            "speaking_interrupt_min_duration_ms": policy.speaking_interrupt_min_duration_ms,
            "speaking_interrupt_min_words": policy.speaking_interrupt_min_words,
            "self_echo_min_words": policy.self_echo_min_words,
            "self_echo_min_overlap": policy.self_echo_min_overlap,
            "aec_warmup_ms": policy.aec_warmup_ms,
            "backchannel_end_cooldown_ms": policy.backchannel_end_cooldown_ms,
            "vad_min_silence_ms": policy.vad_min_silence_ms,
        },
    }
