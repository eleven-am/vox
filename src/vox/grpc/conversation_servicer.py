"""gRPC ConversationService: bidi streaming agent-facing voice orchestration.

Mirrors the WS /v1/conversation protocol (see server/routes/conversation.py).
One bidi RPC per call; client messages drive a `ConversationOrchestrator`; server
messages are produced by the orchestrator's event stream.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from contextlib import suppress

from vox.conversation import TurnPolicy
from vox.conversation.profiles import DEFAULT_TURN_PROFILE, resolve_turn_policy
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationOrchestrator,
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
from vox.operations.errors import (
    InvalidConfigError,
    OperationError,
    SessionAlreadyConfiguredError,
)
from vox.streaming.types import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)


def _int_or_zero(value) -> int:
    return int(value) if value is not None else 0


def _pb_to_config(update: vox_pb2.ConversationSessionUpdate) -> ConversationSessionConfig:
    if not update.stt_model:
        raise InvalidConfigError("session_update requires stt_model")
    if not update.tts_model:
        raise InvalidConfigError("session_update requires tts_model")

    has_policy = update.HasField("policy")
    policy_pb = update.policy
    policy_overrides: dict[str, int | float | bool] = {}
    if has_policy:
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
        ):
            if policy_pb.HasField(field_name):
                policy_overrides[field_name] = getattr(policy_pb, field_name)
    try:
        turn_profile, policy = resolve_turn_policy(
            update.turn_profile or DEFAULT_TURN_PROFILE,
            policy_overrides,
        )
    except ValueError as exc:
        raise InvalidConfigError(str(exc)) from exc

    return ConversationSessionConfig(
        stt_model=update.stt_model,
        tts_model=update.tts_model,
        voice=update.voice or None,
        language=update.language or "en",
        sample_rate=update.sample_rate or TARGET_SAMPLE_RATE,
        turn_profile=turn_profile,
        vad_backend=update.vad_backend or "silero",
        turn_detector=update.turn_detector or "livekit",
        policy=policy,
    )


def _error_pb(message: str) -> vox_pb2.ConverseServerMessage:
    return vox_pb2.ConverseServerMessage(error=vox_pb2.ConversationError(message=message))


def _event_to_pb(event: ConvEvent) -> vox_pb2.ConverseServerMessage | None:
    if isinstance(event, ConvSessionCreatedEvent):
        policy = event.config.policy or TurnPolicy()
        session_created = vox_pb2.ConversationSessionCreated(
            turn_profile=event.config.turn_profile,
        )
        session_created.policy.allow_interrupt_while_speaking = policy.allow_interrupt_while_speaking
        session_created.policy.min_interrupt_duration_ms = policy.min_interrupt_duration_ms
        session_created.policy.max_endpointing_delay_ms = policy.max_endpointing_delay_ms
        session_created.policy.stable_speaking_min_ms = policy.stable_speaking_min_ms
        session_created.policy.false_interruption_timeout_ms = policy.false_interruption_timeout_ms
        session_created.policy.min_interrupt_words = policy.min_interrupt_words
        session_created.policy.partial_interrupts = policy.partial_interrupts
        session_created.policy.dynamic_endpointing = policy.dynamic_endpointing
        session_created.policy.min_endpointing_delay_ms = policy.min_endpointing_delay_ms
        session_created.policy.speaking_interrupt_min_duration_ms = policy.speaking_interrupt_min_duration_ms
        session_created.policy.speaking_interrupt_min_words = policy.speaking_interrupt_min_words
        session_created.policy.self_echo_min_words = policy.self_echo_min_words
        session_created.policy.self_echo_min_overlap = policy.self_echo_min_overlap
        session_created.policy.aec_warmup_ms = policy.aec_warmup_ms
        session_created.policy.backchannel_end_cooldown_ms = policy.backchannel_end_cooldown_ms
        return vox_pb2.ConverseServerMessage(
            session_created=session_created,
        )
    if isinstance(event, ConvSpeechStartedEvent):
        return vox_pb2.ConverseServerMessage(
            speech_started=vox_pb2.ConversationSpeechStarted(timestamp_ms=event.timestamp_ms),
        )
    if isinstance(event, ConvSpeechStoppedEvent):
        return vox_pb2.ConverseServerMessage(
            speech_stopped=vox_pb2.ConversationSpeechStopped(timestamp_ms=event.timestamp_ms),
        )
    if isinstance(event, ConvTranscriptDoneEvent):
        msg = vox_pb2.ConversationTranscriptDone(
            transcript=event.transcript,
            language=event.language,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
        )
        if event.eou_probability is not None:
            msg.eou_probability = event.eou_probability
        for ent in event.entities:
            msg.entities.append(vox_pb2.Entity(
                type=ent.get("type", ""),
                text=ent.get("text", ""),
                start_char=_int_or_zero(ent.get("start_char")),
                end_char=_int_or_zero(ent.get("end_char")),
            ))
        for topic in event.topics:
            msg.topics.append(str(topic))
        for word in event.words:
            pb_word = vox_pb2.WordTimestamp(
                word=str(word.get("word", "")),
                start_ms=_int_or_zero(word.get("start_ms")),
                end_ms=_int_or_zero(word.get("end_ms")),
            )
            if word.get("confidence") is not None:
                pb_word.confidence = float(word["confidence"])
            msg.words.append(pb_word)
        return vox_pb2.ConverseServerMessage(transcript_done=msg)
    if isinstance(event, ConvResponseCreatedEvent):
        return vox_pb2.ConverseServerMessage(
            response_created=vox_pb2.ConversationResponseCreated(response_id=event.response_id),
        )
    if isinstance(event, ConvAudioDeltaEvent):
        pcm = base64.b64decode(event.audio_b64) if event.audio_b64 else b""
        return vox_pb2.ConverseServerMessage(
            audio_delta=vox_pb2.ConversationAudioDelta(
                audio=pcm,
                sample_rate=event.sample_rate,
                response_id=event.response_id,
                sequence=event.sequence,
            ),
        )
    if isinstance(event, ConvAudioClearEvent):
        return vox_pb2.ConverseServerMessage(
            audio_clear=vox_pb2.ConversationAudioClear(response_id=event.response_id),
        )
    if isinstance(event, ConvResponseDoneEvent):
        return vox_pb2.ConverseServerMessage(
            response_done=vox_pb2.ConversationResponseDone(response_id=event.response_id),
        )
    if isinstance(event, ConvResponseCancelledEvent):
        return vox_pb2.ConverseServerMessage(
            response_cancelled=vox_pb2.ConversationResponseCancelled(response_id=event.response_id),
        )
    if isinstance(event, ConvResponseCommittedEvent):
        return vox_pb2.ConverseServerMessage(
            response_committed=vox_pb2.ConversationResponseCommitted(response_id=event.response_id),
        )
    if isinstance(event, ConvInterruptionDetectedEvent):
        return vox_pb2.ConverseServerMessage(
            interruption_detected=vox_pb2.ConversationInterruptionDetected(
                response_id=event.response_id,
                vad_active_ms=event.vad_active_ms,
                partial_transcript=event.partial_transcript or "",
            ),
        )
    if isinstance(event, ConvInterruptionFalsePositiveEvent):
        return vox_pb2.ConverseServerMessage(
            interruption_false_positive=vox_pb2.ConversationInterruptionFalsePositive(
                response_id=event.response_id,
                vad_active_ms=event.vad_active_ms,
                partial_transcript=event.partial_transcript or "",
            ),
        )
    if isinstance(event, ConvTurnEouPredictedEvent):
        return vox_pb2.ConverseServerMessage(
            turn_eou_predicted=vox_pb2.ConversationTurnEouPredicted(
                probability=event.probability,
                threshold=event.threshold,
                decision=event.decision,
                action=event.action,
                delay_ms=event.delay_ms,
                turn_detector=event.turn_detector,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            ),
        )
    if isinstance(event, ConvStateChangedEvent):
        return vox_pb2.ConverseServerMessage(
            state_changed=vox_pb2.ConversationStateChanged(
                state=event.state, previous_state=event.previous_state,
            ),
        )
    if isinstance(event, ConvErrorEvent):
        return _error_pb(event.message)
    return None


def _wire_event_to_pb(event: dict) -> vox_pb2.ConverseServerMessage | None:
    from vox.operations.conversation import _wire_event_to_session_event

    mapped = _wire_event_to_session_event(event)
    if mapped is None:
        return None
    return _event_to_pb(mapped)


class ConversationServicer(vox_pb2_grpc.ConversationServiceServicer):
    def __init__(self, store: BlobStore, registry: ModelRegistry, scheduler: Scheduler) -> None:
        self._store = store
        self._registry = registry
        self._scheduler = scheduler

    async def Converse(
        self,
        request_iterator: AsyncIterator[vox_pb2.ConverseClientMessage],
        context,
    ) -> AsyncIterator[vox_pb2.ConverseServerMessage]:
        orchestrator = ConversationOrchestrator(scheduler=self._scheduler)
        out_queue: asyncio.Queue[vox_pb2.ConverseServerMessage | None] = asyncio.Queue()

        async def pump_events() -> None:
            try:
                async for event in orchestrator.events():
                    pb = _event_to_pb(event)
                    if pb is not None:
                        await out_queue.put(pb)
                    if isinstance(event, ConvDoneEvent):
                        break
            finally:
                await out_queue.put(None)

        async def drain_client() -> None:
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break
                    kind = client_msg.WhichOneof("msg")

                    if kind == "session_update":
                        try:
                            config = _pb_to_config(client_msg.session_update)
                            await orchestrator.start_session(config)
                        except SessionAlreadyConfiguredError:
                            await out_queue.put(_error_pb("session already configured"))
                        except OperationError as exc:
                            await out_queue.put(_error_pb(str(exc)))
                        continue

                    if orchestrator.config is None:
                        await out_queue.put(_error_pb("send session_update first"))
                        continue

                    if kind == "audio_append":
                        await orchestrator.ingest_pcm16(
                            client_msg.audio_append.pcm16,
                            sample_rate=client_msg.audio_append.sample_rate or None,
                        )
                    elif kind == "response_start":
                        await orchestrator.start_response()
                    elif kind == "response_delta":
                        await orchestrator.append_response_text(client_msg.response_delta.delta)
                    elif kind == "response_commit":
                        await orchestrator.commit_response()
                    elif kind == "response_cancel":
                        await orchestrator.cancel_response()
                    else:
                        await out_queue.put(_error_pb(f"unknown message kind: {kind!r}"))
            finally:
                await orchestrator.end_of_stream()

        emit_task = asyncio.create_task(pump_events())
        client_task = asyncio.create_task(drain_client())
        try:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                yield item
        finally:
            client_task.cancel()
            emit_task.cancel()
            with suppress(Exception):
                await client_task
            with suppress(Exception):
                await emit_task
            await orchestrator.close()
