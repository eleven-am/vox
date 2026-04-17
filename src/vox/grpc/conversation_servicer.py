"""gRPC ConversationService: bidi streaming agent-facing voice orchestration.

Mirrors the WS /v1/conversation protocol (see server/routes/conversation.py).
One bidi RPC per call; client messages drive a `ConversationSession`; server
messages are produced by the session's event emitter.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

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
from vox.core.registry import ModelRegistry
from vox.core.scheduler import Scheduler
from vox.core.store import BlobStore
from vox.grpc import vox_pb2, vox_pb2_grpc
from vox.streaming.types import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)


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
        out_queue: asyncio.Queue[vox_pb2.ConverseServerMessage] = asyncio.Queue()
        session: ConversationSession | None = None

        async def on_event(event: dict) -> None:
            pb = _wire_event_to_pb(event)
            if pb is not None:
                await out_queue.put(pb)

        async def drain_client() -> None:
            nonlocal session
            try:
                async for client_msg in request_iterator:
                    if context.cancelled():
                        break
                    kind = client_msg.WhichOneof("msg")

                    if kind == "session_update":
                        if session is not None:
                            await out_queue.put(_error_pb("session already configured"))
                            continue
                        try:
                            config = _pb_to_config(client_msg.session_update)
                        except ValueError as exc:
                            await out_queue.put(_error_pb(str(exc)))
                            continue
                        session = ConversationSession(
                            scheduler=self._scheduler, config=config, on_event=on_event,
                        )
                        await session.start()
                        await out_queue.put(vox_pb2.ConverseServerMessage(
                            session_created=vox_pb2.ConversationSessionCreated(),
                        ))

                    elif session is None:
                        await out_queue.put(_error_pb("send session_update first"))

                    elif kind == "audio_append":
                        pcm = client_msg.audio_append.pcm16
                        sample_rate = client_msg.audio_append.sample_rate or None
                        await session.ingest_audio(pcm, sample_rate=sample_rate)

                    elif kind == "response_start":
                        await session.start_response_stream()

                    elif kind == "response_delta":
                        await session.append_response_text(client_msg.response_delta.delta)

                    elif kind == "response_commit":
                        await session.commit_response_stream()

                    elif kind == "response_cancel":
                        await session.cancel_response()

                    else:
                        await out_queue.put(_error_pb(f"unknown message kind: {kind!r}"))
            finally:
                if session is not None and not context.cancelled():
                    await session.commit_response_stream()
                    await session.wait_until_settled()



                await out_queue.put(None)  # type: ignore[arg-type]

        client_task = asyncio.create_task(drain_client())
        try:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                yield item
        finally:
            client_task.cancel()
            with _suppress():
                await client_task
            if session is not None:
                await session.close()







class _suppress:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return True


def _pb_to_config(update: vox_pb2.ConversationSessionUpdate) -> ConversationConfig:
    if not update.stt_model:
        raise ValueError("session_update requires stt_model")
    if not update.tts_model:
        raise ValueError("session_update requires tts_model")

    policy_pb = update.policy

    default_policy = TurnPolicy()
    policy = TurnPolicy(
        allow_interrupt_while_speaking=(
            policy_pb.allow_interrupt_while_speaking
            if update.HasField("policy")
            else default_policy.allow_interrupt_while_speaking
        ),
        min_interrupt_duration_ms=(
            policy_pb.min_interrupt_duration_ms
            or default_policy.min_interrupt_duration_ms
        ),
        max_endpointing_delay_ms=(
            policy_pb.max_endpointing_delay_ms
            or default_policy.max_endpointing_delay_ms
        ),
        stable_speaking_min_ms=(
            policy_pb.stable_speaking_min_ms
            or default_policy.stable_speaking_min_ms
        ),
    )

    return ConversationConfig(
        stt_model=update.stt_model,
        tts_model=update.tts_model,
        voice=update.voice or None,
        language=update.language or "en",
        sample_rate=update.sample_rate or TARGET_SAMPLE_RATE,
        policy=policy,
    )


def _wire_event_to_pb(event: dict) -> vox_pb2.ConverseServerMessage | None:
    t = event.get("type")
    if t == WIRE_SPEECH_STARTED:
        return vox_pb2.ConverseServerMessage(
            speech_started=vox_pb2.ConversationSpeechStarted(
                timestamp_ms=int(event.get("timestamp_ms", 0)),
            ),
        )
    if t == WIRE_SPEECH_STOPPED:
        return vox_pb2.ConverseServerMessage(
            speech_stopped=vox_pb2.ConversationSpeechStopped(
                timestamp_ms=int(event.get("timestamp_ms", 0)),
            ),
        )
    if t == WIRE_TRANSCRIPT_DONE:
        msg = vox_pb2.ConversationTranscriptDone(
            transcript=event.get("transcript", ""),
            language=event.get("language", ""),
            start_ms=int(event.get("start_ms", 0)),
            end_ms=int(event.get("end_ms", 0)),
        )
        if event.get("eou_probability") is not None:
            msg.eou_probability = float(event["eou_probability"])
        for ent in event.get("entities") or []:
            msg.entities.append(vox_pb2.Entity(
                type=ent.get("type", ""),
                text=ent.get("text", ""),
                start_char=int(ent.get("start_char", 0)),
                end_char=int(ent.get("end_char", 0)),
            ))
        for topic in event.get("topics") or []:
            msg.topics.append(str(topic))
        for word in event.get("words") or []:
            pb_word = vox_pb2.WordTimestamp(
                word=str(word.get("word", "")),
                start_ms=int(word.get("start_ms", 0)),
                end_ms=int(word.get("end_ms", 0)),
            )
            if word.get("confidence") is not None:
                pb_word.confidence = float(word["confidence"])
            msg.words.append(pb_word)
        return vox_pb2.ConverseServerMessage(transcript_done=msg)
    if t == WIRE_RESPONSE_CREATED:
        return vox_pb2.ConverseServerMessage(
            response_created=vox_pb2.ConversationResponseCreated(),
        )
    if t == WIRE_AUDIO_DELTA:
        import base64

        audio_b64 = event.get("audio") or ""
        pcm = base64.b64decode(audio_b64) if audio_b64 else b""
        return vox_pb2.ConverseServerMessage(
            audio_delta=vox_pb2.ConversationAudioDelta(
                audio=pcm,
                sample_rate=int(event.get("sample_rate", 0)),
            ),
        )
    if t == WIRE_RESPONSE_DONE:
        return vox_pb2.ConverseServerMessage(
            response_done=vox_pb2.ConversationResponseDone(),
        )
    if t == WIRE_RESPONSE_CANCELLED:
        return vox_pb2.ConverseServerMessage(
            response_cancelled=vox_pb2.ConversationResponseCancelled(),
        )
    if t == WIRE_RESPONSE_COMMITTED:
        return vox_pb2.ConverseServerMessage(
            response_committed=vox_pb2.ConversationResponseCommitted(),
        )
    if t == WIRE_STATE_CHANGED:
        return vox_pb2.ConverseServerMessage(
            state_changed=vox_pb2.ConversationStateChanged(
                state=event.get("state", ""),
                previous_state=event.get("previous_state", ""),
            ),
        )
    if t == WIRE_ERROR:
        return _error_pb(event.get("message", ""))
    logger.debug("unmapped event: %s", t)
    return None


def _error_pb(message: str) -> vox_pb2.ConverseServerMessage:
    return vox_pb2.ConverseServerMessage(
        error=vox_pb2.ConversationError(message=message),
    )
