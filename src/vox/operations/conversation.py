from __future__ import annotations

import asyncio
import base64
import json
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
    ERROR_CODE_COMMAND_INVALID,
    ERROR_CODE_RESPONSE_ALREADY_ACTIVE,
    ERROR_CODE_RESPONSE_REJECTED_TURN_STATE,
    ERROR_CODE_RESPONSE_STALE_GENERATION,
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
    AppendResult,
    ConversationConfig,
    ConversationSession,
)
from vox.operations.errors import (
    ConversationCommandError,
    InvalidConfigError,
    SessionAlreadyConfiguredError,
    SessionNotConfiguredError,
)
from vox.streaming.eou import DEFAULT_TURN_DETECTOR
from vox.streaming.types import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)

WIRE_SESSION_CREATED = "session.created"
WIRE_CLIENT_EVENT = "client.event"
WIRE_BROWSER_EVENT = "browser.event"
WIRE_RTC_CLIENT_DISCONNECTED = "rtc.client.disconnected"
SESSION_UPDATE_STT_MODEL_FIELDS = (
    "stt_model",
    "input_audio_transcription.model",
)
SESSION_UPDATE_TTS_MODEL_FIELDS = (
    "tts_model",
    "output_audio_generation.model",
)
SESSION_UPDATE_TURN_PROFILE_FIELDS = (
    "turn_profile",
    "profile",
)
SESSION_UPDATE_VAD_BACKEND_FIELDS = (
    "vad_backend",
    "vad",
)
SESSION_UPDATE_TURN_DETECTOR_FIELDS = (
    "turn_detector",
    "eou_model",
)
SESSION_UPDATE_POLICY_FIELDS = (
    "turn_policy",
    "policy",
)
RESPONSE_COMMAND_ENVELOPE_FIELDS = ("response",)
RESPONSE_TEXT_COMPATIBILITY_FIELDS = ("text", "delta")
RESPONSE_ALLOW_INTERRUPTION_FIELD = "allow_interruptions"
TURN_POLICY_OVERRIDE_FIELDS = (
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
)


@dataclass(frozen=True)
class ConversationSessionConfig:
    stt_model: str
    tts_model: str
    voice: str | None = None
    language: str = "en"
    sample_rate: int = TARGET_SAMPLE_RATE
    turn_profile: str = DEFAULT_TURN_PROFILE
    vad_backend: str = "silero"
    turn_detector: str = DEFAULT_TURN_DETECTOR
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
    generation_id: str | None = None


@dataclass(frozen=True)
class ConvAudioDeltaEvent:
    audio: bytes
    sample_rate: int
    audio_format: str
    response_id: str = ""
    sequence: int = 0


@dataclass(frozen=True)
class ConvAudioClearEvent:
    response_id: str = ""
    generation_id: str | None = None


@dataclass(frozen=True)
class ConvResponseDoneEvent:
    response_id: str = ""
    generation_id: str | None = None


@dataclass(frozen=True)
class ConvResponseCancelledEvent:
    response_id: str = ""
    generation_id: str | None = None


@dataclass(frozen=True)
class ConvResponseCommittedEvent:
    response_id: str = ""
    generation_id: str | None = None


@dataclass(frozen=True)
class ConvInterruptionDetectedEvent:
    response_id: str
    vad_active_ms: int
    partial_transcript: str | None
    reason: str | None = None
    generation_id: str | None = None


@dataclass(frozen=True)
class ConvInterruptionFalsePositiveEvent:
    response_id: str
    vad_active_ms: int
    partial_transcript: str | None
    reason: str | None = None
    generation_id: str | None = None


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
    code: str = ""
    recoverable: bool = True
    generation_id: str | None = None


@dataclass(frozen=True)
class ConvDoneEvent:
    pass


_APPEND_RESULT_WIRE_ERRORS: dict[AppendResult, tuple[str, str]] = {
    AppendResult.SESSION_CLOSED: (
        ERROR_CODE_RESPONSE_REJECTED_TURN_STATE,
        "session is closed",
    ),
    AppendResult.NO_ACTIVE_RESPONSE: (
        ERROR_CODE_RESPONSE_STALE_GENERATION,
        "no active response for this generation",
    ),
    AppendResult.RESPONSE_MISMATCH: (
        ERROR_CODE_RESPONSE_STALE_GENERATION,
        "response id does not match the active response",
    ),
    AppendResult.RESPONSE_COMMITTED: (
        ERROR_CODE_RESPONSE_STALE_GENERATION,
        "response is already committed",
    ),
    AppendResult.STREAM_ENDED: (
        ERROR_CODE_RESPONSE_STALE_GENERATION,
        "response stream has ended",
    ),
}


def conversation_error_fields(exc: BaseException) -> tuple[str, bool, str | None]:
    code = getattr(exc, "code", None)
    recoverable = getattr(exc, "recoverable", True)
    generation_id = getattr(exc, "generation_id", None)
    if not isinstance(code, str) or not code:
        code = ERROR_CODE_COMMAND_INVALID
    if not isinstance(generation_id, str) or not generation_id:
        generation_id = None
    return code, bool(recoverable), generation_id


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
        return _with_generation_id(
            {"type": WIRE_RESPONSE_CREATED, "response_id": event.response_id},
            event.generation_id,
        )
    if isinstance(event, ConvAudioDeltaEvent):
        return {
            "type": WIRE_AUDIO_DELTA,
            "audio": base64.b64encode(event.audio).decode("ascii"),
            "sample_rate": event.sample_rate,
            "audio_format": event.audio_format,
            "response_id": event.response_id,
            "sequence": event.sequence,
        }
    if isinstance(event, ConvAudioClearEvent):
        return _with_generation_id(
            {"type": WIRE_AUDIO_CLEAR, "response_id": event.response_id},
            event.generation_id,
        )
    if isinstance(event, ConvResponseDoneEvent):
        return _with_generation_id(
            {"type": WIRE_RESPONSE_DONE, "response_id": event.response_id},
            event.generation_id,
        )
    if isinstance(event, ConvResponseCancelledEvent):
        return _with_generation_id(
            {"type": WIRE_RESPONSE_CANCELLED, "response_id": event.response_id},
            event.generation_id,
        )
    if isinstance(event, ConvResponseCommittedEvent):
        return _with_generation_id(
            {"type": WIRE_RESPONSE_COMMITTED, "response_id": event.response_id},
            event.generation_id,
        )
    if isinstance(event, ConvInterruptionDetectedEvent):
        payload: dict = {
            "type": WIRE_INTERRUPTION_DETECTED,
            "response_id": event.response_id,
            "vad_active_ms": event.vad_active_ms,
            "partial_transcript": event.partial_transcript,
        }
        if event.reason:
            payload["reason"] = event.reason
        return _with_generation_id(payload, event.generation_id)
    if isinstance(event, ConvInterruptionFalsePositiveEvent):
        payload_fp: dict = {
            "type": WIRE_INTERRUPTION_FALSE_POSITIVE,
            "response_id": event.response_id,
            "vad_active_ms": event.vad_active_ms,
            "partial_transcript": event.partial_transcript,
        }
        if event.reason:
            payload_fp["reason"] = event.reason
        return _with_generation_id(payload_fp, event.generation_id)
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
        return _with_generation_id(
            {
                "type": WIRE_ERROR,
                "message": event.message,
                "code": event.code,
                "recoverable": event.recoverable,
            },
            event.generation_id,
        )
    return None


def _with_generation_id(payload: dict, generation_id: str | None) -> dict:
    if generation_id:
        payload["generation_id"] = generation_id
    return payload


def _wire_generation_id(event: dict) -> str | None:
    value = event.get("generation_id")
    if value is None:
        return None
    generation_id = str(value).strip()
    return generation_id or None


def conversation_wire_event_payload(wire: dict) -> tuple[str, dict]:
    event_type = wire.get("type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise InvalidConfigError("conversation wire event requires a non-empty string 'type'")
    payload = dict(wire)
    payload.pop("type", None)
    return event_type.strip(), payload


def parse_session_update(payload: dict) -> ConversationSessionConfig:
    sess = session_update_payload(payload)
    stt_model = _first_session_value(sess, SESSION_UPDATE_STT_MODEL_FIELDS)
    tts_model = _first_session_value(sess, SESSION_UPDATE_TTS_MODEL_FIELDS)
    return build_conversation_session_config(
        stt_model=stt_model,
        tts_model=tts_model,
        voice=sess.get("voice"),
        language=sess.get("language", "en") or "en",
        sample_rate=int(sess.get("sample_rate") or TARGET_SAMPLE_RATE),
        turn_profile=str(_first_session_value(sess, SESSION_UPDATE_TURN_PROFILE_FIELDS) or DEFAULT_TURN_PROFILE),
        vad_backend=str(_first_session_value(sess, SESSION_UPDATE_VAD_BACKEND_FIELDS) or "silero"),
        turn_detector=str(_first_session_value(sess, SESSION_UPDATE_TURN_DETECTOR_FIELDS) or DEFAULT_TURN_DETECTOR),
        policy_overrides=_first_session_mapping(sess, SESSION_UPDATE_POLICY_FIELDS),
        include_word_timestamps=bool(sess.get("include_word_timestamps") or False),
    )


def build_conversation_session_config(
    *,
    stt_model: str | None,
    tts_model: str | None,
    voice: str | None = None,
    language: str = "en",
    sample_rate: int = TARGET_SAMPLE_RATE,
    turn_profile: str = DEFAULT_TURN_PROFILE,
    vad_backend: str = "silero",
    turn_detector: str = DEFAULT_TURN_DETECTOR,
    policy_overrides: dict[str, Any] | None = None,
    include_word_timestamps: bool = False,
) -> ConversationSessionConfig:
    if not stt_model:
        raise InvalidConfigError("session.update requires 'stt_model'")
    if not tts_model:
        raise InvalidConfigError("session.update requires 'tts_model'")

    policy_in = policy_overrides or {}
    policy_kwargs = {}
    for field_name in TURN_POLICY_OVERRIDE_FIELDS:
        if field_name in policy_in:
            policy_kwargs[field_name] = policy_in[field_name]
    try:
        turn_profile, policy = resolve_turn_policy(turn_profile, policy_kwargs)
    except ValueError as exc:
        raise InvalidConfigError(str(exc)) from exc

    return ConversationSessionConfig(
        stt_model=stt_model,
        tts_model=tts_model,
        voice=voice,
        language=language,
        sample_rate=sample_rate,
        turn_profile=turn_profile,
        vad_backend=vad_backend,
        turn_detector=turn_detector,
        policy=policy,
        include_word_timestamps=include_word_timestamps,
    )


def session_update_payload(payload: dict) -> dict:
    session = payload.get("session")
    if isinstance(session, dict):
        return session
    return payload


def _first_session_value(session: dict, fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = _session_field_value(session, field)
        if value:
            return value
    return None


def _first_session_mapping(session: dict, fields: tuple[str, ...]) -> dict:
    for field in fields:
        value = _session_field_value(session, field)
        if isinstance(value, dict):
            return value
    return {}


def _session_field_value(session: dict, field: str) -> Any:
    current: Any = session
    for part in field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def parse_allow_interruptions(message: dict) -> bool:
    for payload in response_command_payloads(message):
        if RESPONSE_ALLOW_INTERRUPTION_FIELD in payload:
            return bool(payload[RESPONSE_ALLOW_INTERRUPTION_FIELD])
    return True


def parse_response_text(message: dict, preferred_key: str) -> str | None:
    fields = response_text_fields(preferred_key)
    for payload in response_command_payloads(message):
        for field in fields:
            value = payload.get(field)
            if value is not None:
                return str(value)
    return None


def parse_response_generation_id(message: dict) -> str | None:
    for payload in response_command_payloads(message):
        value = payload.get("generation_id", payload.get("generationId"))
        if value is not None:
            generation_id = str(value).strip()
            return generation_id or None
    return None


def response_command_payloads(message: dict) -> tuple[dict, ...]:
    payloads: list[dict] = []
    for field in RESPONSE_COMMAND_ENVELOPE_FIELDS:
        value = message.get(field)
        if isinstance(value, dict):
            payloads.append(value)
    payloads.append(message)
    return tuple(payloads)


def response_text_fields(preferred_key: str) -> tuple[str, ...]:
    fields = [preferred_key]
    fields.extend(field for field in RESPONSE_TEXT_COMPATIBILITY_FIELDS if field != preferred_key)
    return tuple(fields)


def parse_client_event_command(message: Any) -> tuple[str, Any]:
    if not isinstance(message, dict):
        raise InvalidConfigError("client.event requires a JSON object")
    return parse_client_event_parts(
        message.get("event"),
        message.get("payload"),
        non_empty_message="client.event requires a non-empty string 'event'",
    )


def parse_client_event_parts(
    event_name: Any,
    payload: Any,
    *,
    non_empty_message: str = "client.event requires a non-empty string 'event'",
) -> tuple[str, Any]:
    if not isinstance(event_name, str) or not event_name.strip():
        raise InvalidConfigError(non_empty_message)
    return event_name.strip(), payload


def client_event_payload(event_name: str, payload: Any) -> dict:
    return {
        "event": event_name,
        "payload": payload,
    }


def client_event_payload_json(event_name: str, payload: Any) -> str:
    return json.dumps(client_event_payload(event_name, payload))


def control_event_client_payload_json(payload: Any) -> str:
    return json.dumps(payload)


def browser_event_wire(session_id: str, event_name: str, payload: Any) -> dict:
    return {
        "type": WIRE_BROWSER_EVENT,
        "session_id": session_id,
        "event": event_name,
        "payload": payload,
    }


def control_event_as_client_event(event: dict) -> tuple[str, Any]:
    event_type = event.get("type")
    if event_type in {WIRE_CLIENT_EVENT, WIRE_BROWSER_EVENT}:
        return parse_client_event_command(event)
    if not isinstance(event_type, str) or not event_type.strip():
        raise InvalidConfigError("control event requires a non-empty string 'type'")
    return event_type.strip(), {key: value for key, value in event.items() if key != "type"}


def client_disconnected_wire(
    session_id: str,
    *,
    reason: str,
    connection_state: str | None = None,
    ice_connection_state: str | None = None,
    data_channel_state: str | None = None,
) -> dict:
    payload = {
        "type": WIRE_RTC_CLIENT_DISCONNECTED,
        "session_id": session_id,
        "reason": reason,
    }
    if connection_state is not None:
        payload["connection_state"] = connection_state
    if ice_connection_state is not None:
        payload["ice_connection_state"] = ice_connection_state
    if data_channel_state is not None:
        payload["data_channel_state"] = data_channel_state
    return payload


def parse_conversation_wire_event(event: dict) -> ConvEvent | None:
    t = event.get("type")
    if t == WIRE_SESSION_CREATED:
        return ConvSessionCreatedEvent(config=parse_session_update(event))
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
            eou_probability=(float(event["eou_probability"]) if event.get("eou_probability") is not None else None),
            entities=tuple(event.get("entities") or ()),
            topics=tuple(event.get("topics") or ()),
            words=tuple(event.get("words") or ()),
        )
    if t == WIRE_RESPONSE_CREATED:
        return ConvResponseCreatedEvent(
            response_id=str(event.get("response_id") or ""),
            generation_id=_wire_generation_id(event),
        )
    if t == WIRE_AUDIO_DELTA:
        raw_audio = event.get("audio_pcm16")
        if raw_audio is not None:
            pcm16 = bytes(raw_audio)
        else:
            audio_b64 = event.get("audio")
            try:
                pcm16 = base64.b64decode(audio_b64) if audio_b64 else b""
            except Exception as exc:  # noqa: BLE001
                raise InvalidConfigError(f"invalid base64 audio event: {exc}") from exc
        return ConvAudioDeltaEvent(
            audio=pcm16,
            sample_rate=int(event.get("sample_rate") or 0),
            audio_format=str(event.get("audio_format") or "pcm16"),
            response_id=str(event.get("response_id") or ""),
            sequence=int(event.get("sequence") or 0),
        )
    if t == WIRE_AUDIO_CLEAR:
        return ConvAudioClearEvent(
            response_id=str(event.get("response_id") or ""),
            generation_id=_wire_generation_id(event),
        )
    if t == WIRE_RESPONSE_DONE:
        return ConvResponseDoneEvent(
            response_id=str(event.get("response_id") or ""),
            generation_id=_wire_generation_id(event),
        )
    if t == WIRE_RESPONSE_CANCELLED:
        return ConvResponseCancelledEvent(
            response_id=str(event.get("response_id") or ""),
            generation_id=_wire_generation_id(event),
        )
    if t == WIRE_RESPONSE_COMMITTED:
        return ConvResponseCommittedEvent(
            response_id=str(event.get("response_id") or ""),
            generation_id=_wire_generation_id(event),
        )
    if t == WIRE_INTERRUPTION_DETECTED:
        return ConvInterruptionDetectedEvent(
            response_id=str(event.get("response_id") or ""),
            vad_active_ms=int(event.get("vad_active_ms") or 0),
            partial_transcript=(
                str(event["partial_transcript"]) if event.get("partial_transcript") is not None else None
            ),
            reason=str(event["reason"]) if event.get("reason") else None,
            generation_id=_wire_generation_id(event),
        )
    if t == WIRE_INTERRUPTION_FALSE_POSITIVE:
        return ConvInterruptionFalsePositiveEvent(
            response_id=str(event.get("response_id") or ""),
            vad_active_ms=int(event.get("vad_active_ms") or 0),
            partial_transcript=(
                str(event["partial_transcript"]) if event.get("partial_transcript") is not None else None
            ),
            reason=str(event["reason"]) if event.get("reason") else None,
            generation_id=_wire_generation_id(event),
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
        raw_recoverable = event.get("recoverable")
        return ConvErrorEvent(
            message=str(event.get("message", "")),
            code=str(event.get("code") or ""),
            recoverable=True if raw_recoverable is None else bool(raw_recoverable),
            generation_id=_wire_generation_id(event),
        )
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
        output_playout_observed: bool = False,
    ) -> None:
        self._scheduler = scheduler
        self._pace_response_done_to_audio = pace_response_done_to_audio
        self._audio_sink = audio_sink
        self._wait_for_output_playout = wait_for_output_playout
        self._output_playout_observed = output_playout_observed
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
            output_playout_observed=self._output_playout_observed,
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

    def observe_output_playout(self, pcm16: bytes, sample_rate: int) -> None:
        if self._session is not None:
            self._session.observe_output_playout(pcm16, sample_rate)

    @property
    def response_active(self) -> bool:
        return self._session is not None and self._session.response_active

    @property
    def active_generation_id(self) -> str | None:
        if self._session is None:
            return None
        return self._session.active_generation_id

    async def start_response(
        self,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
    ) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        if self._session.response_active:
            raise ConversationCommandError(
                "response already active",
                code=ERROR_CODE_RESPONSE_ALREADY_ACTIVE,
                generation_id=generation_id,
            )
        result = await self._session.start_response_stream(
            allow_interruptions=allow_interruptions,
            generation_id=generation_id,
        )
        context = result.context
        logger.info(
            "response start generation_id=%s accepted=%s response_id=%s state=%s "
            "input_speech_active=%s candidate_id=%s candidate_status=%s candidate_reason=%s",
            generation_id,
            result.accepted,
            result.response_id,
            context.turn_state.value,
            context.input_speech_active,
            context.candidate_id,
            context.candidate_status.value if context.candidate_status is not None else None,
            context.candidate_reason,
        )
        if result.rejection is not None:
            raise ConversationCommandError(
                result.rejection.message,
                code=result.rejection.code,
                generation_id=result.rejection.generation_id,
            )
        if result.response_id is None:
            raise RuntimeError("response start returned neither an id nor a rejection")

    async def append_response_text(
        self,
        text: str,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
    ) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        response_id = self._admitted_response_id(self._session, generation_id, command="delta")
        result = await self._session.append_response_text(
            text,
            allow_interruptions=allow_interruptions,
            expected_response_id=response_id,
        )
        self._raise_for_append_result(result, generation_id=generation_id)

    async def replace_response_text(self, text: str, *, allow_interruptions: bool = True) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        await self._session.replace_response_text(text, allow_interruptions=allow_interruptions)

    async def commit_response(self, *, generation_id: str | None = None) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        response_id = self._admitted_response_id(self._session, generation_id, command="commit")
        result = await self._session.commit_response_stream(
            expected_response_id=response_id,
        )
        self._raise_for_append_result(result, generation_id=generation_id)

    async def cancel_response(self, *, generation_id: str | None = None) -> None:
        if self._session is None:
            raise SessionNotConfiguredError()
        if generation_id is not None and generation_id != self._live_generation_identity():
            raise ConversationCommandError(
                "stale response generation",
                code=ERROR_CODE_RESPONSE_STALE_GENERATION,
                generation_id=generation_id,
            )
        await self._session.cancel_response()

    async def report_error(
        self,
        message: str,
        *,
        code: str = ERROR_CODE_COMMAND_INVALID,
        recoverable: bool = True,
        generation_id: str | None = None,
    ) -> None:
        await self._events.put(
            ConvErrorEvent(
                message=message,
                code=code,
                recoverable=recoverable,
                generation_id=generation_id,
            )
        )

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
        mapped = parse_conversation_wire_event(event)
        if mapped is None:
            return
        if self._audio_sink is not None and isinstance(mapped, ConvAudioDeltaEvent):
            await self._audio_sink(mapped)
            return
        await self._events.put(mapped)

    def _live_generation_identity(self) -> str | None:
        if self._session is None:
            return None
        response_id = self._session.active_response_id
        if response_id is None:
            return None
        return self._session.active_generation_id or response_id

    def _admitted_response_id(self, session: ConversationSession, generation_id: str | None, *, command: str) -> str:
        response_id = session.active_response_id
        if response_id is None:
            raise ConversationCommandError(
                f"response.start required before response.{command}",
                code=ERROR_CODE_RESPONSE_STALE_GENERATION,
                generation_id=generation_id,
            )
        if generation_id is not None and generation_id != (session.active_generation_id or response_id):
            raise ConversationCommandError(
                "stale response generation",
                code=ERROR_CODE_RESPONSE_STALE_GENERATION,
                generation_id=generation_id,
            )
        return response_id

    def _raise_for_append_result(self, result: AppendResult, *, generation_id: str | None) -> None:
        if result is AppendResult.ACCEPTED:
            return
        code, message = _APPEND_RESULT_WIRE_ERRORS[result]
        raise ConversationCommandError(
            message,
            code=code,
            generation_id=generation_id,
        )


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
