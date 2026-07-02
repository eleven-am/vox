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

from vox.conversation import TurnPolicy
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.core.tasks import reap_task
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.grpc.conversation_commands import converse_client_message_to_command
from vox.operations.conversation import (
    ConvAudioClearEvent,
    ConvAudioDeltaEvent,
    ConvDoneEvent,
    ConvErrorEvent,
    ConversationOrchestrator,
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
    ConvTranscriptDeltaEvent,
    ConvTranscriptDoneEvent,
    ConvTurnEouPredictedEvent,
    execute_conversation_command,
    execute_conversation_session_update,
)
from vox.operations.errors import OperationError

logger = logging.getLogger(__name__)


def _int_or_zero(value) -> int:
    return int(value) if value is not None else 0


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
        session_created.policy.vad_min_silence_ms = policy.vad_min_silence_ms
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
    if isinstance(event, ConvTranscriptDeltaEvent):
        return vox_pb2.ConverseServerMessage(
            transcript_delta=vox_pb2.ConversationTranscriptDelta(
                delta=event.delta,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            ),
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
                reason=event.reason or "",
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

                    try:
                        command = converse_client_message_to_command(client_msg)
                        if command.kind == "session_update":
                            assert command.config is not None
                            await execute_conversation_session_update(orchestrator, command.config)
                        else:
                            assert command.message is not None
                            await execute_conversation_command(
                                orchestrator,
                                command.message,
                                require_config_message="send session_update first",
                                unknown_message_label="unknown message kind",
                            )
                    except OperationError as exc:
                        await out_queue.put(_error_pb(str(exc)))
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
            await reap_task(client_task)
            await reap_task(emit_task)
            await orchestrator.close()
