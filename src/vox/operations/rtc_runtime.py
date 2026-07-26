"""Canonical RTC signaling and conversation lifecycle."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from vox.core.scheduler import Scheduler
from vox.operations.conversation import ConvEvent, deliver_wire_with_lifecycle_retry
from vox.operations.conversation_commands import (
    ClientEventCommand,
    ConversationCommand,
    SessionUpdateCommand,
)
from vox.operations.conversation_runtime import ConversationRuntime
from vox.operations.errors import (
    InvalidConfigError,
    RtcControlAlreadyAttachedError,
    RtcControlTransportMismatchError,
    RtcSessionNotFoundError,
)
from vox.operations.rtc_signaling import (
    RtcCandidateRequest,
    RtcOfferRequest,
    add_server_rtc_candidate,
    exchange_server_rtc_offer,
)
from vox.server.rtc_media import stamp_negotiation_generation
from vox.server.rtc_registry import RtcControlTransport, RtcSessionRegistry
from vox.server.rtc_session_io import (
    create_rtc_orchestrator,
    prepare_rtc_control_event,
    send_client_event_to_browser,
)
from vox.speech_context.service import SpeechContextService

RtcEventHandler = Callable[[dict[str, Any]], Awaitable[Any]]
RtcConversationEventHandler = Callable[[ConvEvent, dict[str, Any]], Awaitable[Any]]

logger = logging.getLogger(__name__)
MAX_PENDING_REMOTE_CANDIDATES = 256


@dataclass(frozen=True, slots=True)
class RtcOfferCommand:
    offer_type: str
    sdp: str
    restart: bool = False
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class RtcCandidateCommand:
    candidate: str | None
    sdp_mid: str | None = None
    sdp_m_line_index: int | None = None
    username_fragment: str | None = None
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class RtcCloseCommand:
    reason: str = "client_closed"


RtcCommand = ConversationCommand | RtcOfferCommand | RtcCandidateCommand | RtcCloseCommand


class RtcRuntime:
    """Own one attached RTC session independently of its control transport."""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        store: Any | None = None,
        registry: RtcSessionRegistry,
        session_id: str,
        transport: RtcControlTransport,
        emit: RtcEventHandler,
        emit_conversation: RtcConversationEventHandler | None = None,
        speech_context_service: SpeechContextService | None = None,
    ) -> None:
        record = registry.get(session_id)
        if record is None:
            raise RtcSessionNotFoundError(session_id)
        if record.expected_control_transport != transport:
            raise RtcControlTransportMismatchError(
                session_id,
                expected=record.expected_control_transport,
                received=transport,
            )
        if record.control_attached:
            raise RtcControlAlreadyAttachedError(session_id)
        record = registry.attach_control(session_id, transport=transport)
        if record is None:
            raise RtcControlAlreadyAttachedError(session_id)
        self.session_id = session_id
        self.transport = transport
        self.record = record
        self._registry = registry
        self._emit = emit
        self._emit_conversation = emit_conversation
        self._dispatch_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed_event = asyncio.Event()
        self._terminal_close_task: asyncio.Task[None] | None = None
        self._answer_barrier_token: object | None = None
        self._answer_barrier_active = False
        self._answer_barrier_generation: int | None = None
        self._answer_barrier_attempt_id: str | None = None
        self._active_negotiation_attempt_id: str | None = None
        self._buffered_local_candidates: list[dict[str, Any]] = []
        self._started = False
        self._closed = False

        orchestrator = create_rtc_orchestrator(
            scheduler=scheduler,
            store=store,
            record=record,
            speech_context_service=speech_context_service,
        )
        self.conversation = ConversationRuntime(
            orchestrator,
            allow_input_audio=False,
            client_event_handler=lambda event_name, payload: send_client_event_to_browser(
                record,
                event_name,
                payload,
            ),
            require_config_message="send session.update first",
            unknown_message_label="unknown RTC control message type",
            flush_response_on_close=False,
        )

    async def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise InvalidConfigError("RTC session is closed")
        await self._emit(
            {
                "type": "rtc.session.attached",
                "session_id": self.session_id,
                "provider": self.transport,
            }
        )
        self._started = True
        self.conversation.start_event_pump(self._emit_conversation_event)
        self.conversation.start_background_task(self._pump_media_events())
        self.conversation.start_background_task(self._pump_client_events())

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    async def dispatch(self, command: RtcCommand) -> None:
        async with self._dispatch_lock:
            if self._closed:
                raise InvalidConfigError("RTC session is closed")
            if isinstance(command, RtcOfferCommand):
                await self._apply_offer(command)
                return
            if isinstance(command, RtcCandidateCommand):
                await self._apply_candidate(command)
                return
            if isinstance(command, RtcCloseCommand):
                await self.close(reason=command.reason)
                return
            if not self.record.remote_description_set and not isinstance(
                command,
                SessionUpdateCommand | ClientEventCommand,
            ):
                raise InvalidConfigError("send rtc.offer first")
            await self.conversation.dispatch(command)

    async def _apply_offer(self, command: RtcOfferCommand) -> None:
        if command.offer_type != "offer" or not command.sdp.strip():
            raise InvalidConfigError("rtc.offer requires an SDP offer")
        if command.restart and command.generation is None:
            raise InvalidConfigError("RTC restart offer requires a generation")
        if self.record.remote_description_set:
            if not command.restart:
                raise InvalidConfigError("RTC offer already applied; set restart=true for an ICE restart")
        elif command.restart:
            raise InvalidConfigError("RTC restart requires an active negotiation")

        pending = tuple(self.record.pending_remote_candidates)
        if command.generation is not None and any(candidate.generation is None for candidate in pending):
            raise InvalidConfigError("RTC candidate generation required for generated negotiation")
        applicable = tuple(candidate for candidate in pending if candidate.generation == command.generation)
        self._begin_answer_barrier(command.generation)
        result = None
        try:
            result = await exchange_server_rtc_offer(
                registry=self._registry,
                request=RtcOfferRequest(
                    session_id=self.session_id,
                    offer_type=command.offer_type,
                    sdp=command.sdp,
                    restart=command.restart,
                    generation=command.generation,
                ),
            )
            self._answer_barrier_attempt_id = result.attempt_id
            for candidate in applicable:
                await self._send_candidate(candidate, mark_complete=False)
            answer = stamp_negotiation_generation(
                self.record,
                {
                    "type": "rtc.answer",
                    "session_id": self.session_id,
                    "answer": {"type": result.answer_type, "sdp": result.sdp},
                },
                generation=command.generation,
            )
            delivered = await self._emit(answer)
            if delivered is False:
                raise InvalidConfigError("RTC answer delivery failed")
            await result.commit()
        except BaseException as error:
            if result is not None:
                interrupted = await self._rollback_offer(result)
            else:
                interrupted = False
            self._finish_answer_barrier(success=False)
            if interrupted and not isinstance(error, asyncio.CancelledError):
                raise asyncio.CancelledError from error
            raise
        self.record.browser_attached = True
        self.record.negotiation_generation = command.generation
        self._active_negotiation_attempt_id = result.attempt_id
        self.record.remote_description_set = True
        self.record.remote_candidates_complete = any(candidate.candidate is None for candidate in applicable)
        self.record.pending_remote_candidates[:] = [
            candidate for candidate in pending if self._candidate_is_future(candidate, command.generation)
        ]
        self._finish_answer_barrier(success=True)

    async def _apply_candidate(self, command: RtcCandidateCommand) -> None:
        if self.record.remote_description_set:
            generation = self.record.negotiation_generation
            if generation is not None and command.generation is None:
                raise InvalidConfigError("RTC candidate generation required for generated negotiation")
            if command.generation != generation:
                if command.generation is not None and (generation is None or command.generation > generation):
                    self._queue_remote_candidate(command)
                return
        else:
            self._queue_remote_candidate(command)
            return
        if self.record.remote_candidates_complete:
            if command.candidate is None:
                return
            raise InvalidConfigError("RTC candidate received after end-of-candidates")
        await self._send_candidate(command)

    async def _send_candidate(
        self,
        command: RtcCandidateCommand,
        *,
        mark_complete: bool = True,
    ) -> None:
        await add_server_rtc_candidate(
            registry=self._registry,
            request=RtcCandidateRequest(
                session_id=self.session_id,
                candidate=command.candidate,
                sdp_mid=command.sdp_mid,
                sdp_m_line_index=command.sdp_m_line_index,
                username_fragment=command.username_fragment,
                generation=command.generation,
            ),
        )
        if mark_complete and command.candidate is None:
            self.record.remote_candidates_complete = True

    def _queue_remote_candidate(self, command: RtcCandidateCommand) -> None:
        matching = (
            candidate
            for candidate in self.record.pending_remote_candidates
            if candidate.generation == command.generation
        )
        if command.candidate is not None and any(candidate.candidate is None for candidate in matching):
            raise InvalidConfigError("RTC candidate received after end-of-candidates")
        if len(self.record.pending_remote_candidates) >= MAX_PENDING_REMOTE_CANDIDATES:
            raise InvalidConfigError("too many RTC candidates before offer")
        self.record.pending_remote_candidates.append(command)

    @staticmethod
    def _candidate_is_future(
        candidate: RtcCandidateCommand,
        generation: int | None,
    ) -> bool:
        if candidate.generation is None:
            return False
        return generation is None or candidate.generation > generation

    def _begin_answer_barrier(self, generation: int | None) -> None:
        self._answer_barrier_token = object()
        self._answer_barrier_active = True
        self._answer_barrier_generation = generation
        self._answer_barrier_attempt_id = None
        self._buffered_local_candidates = []

    def _finish_answer_barrier(self, *, success: bool) -> None:
        token = self._answer_barrier_token
        if token is None:
            return
        if not success:
            self._buffered_local_candidates.clear()
            self._clear_answer_barrier(token)
            return
        if self._buffered_local_candidates:
            self.conversation.start_background_task(
                self._flush_buffered_local_candidates(
                    token,
                    self._buffered_local_candidates,
                    self._answer_barrier_attempt_id,
                )
            )
            return
        self._clear_answer_barrier(token)

    def _clear_answer_barrier(self, token: object) -> None:
        if self._answer_barrier_token is not token:
            return
        self._answer_barrier_token = None
        self._answer_barrier_active = False
        self._answer_barrier_generation = None
        self._answer_barrier_attempt_id = None
        self._buffered_local_candidates = []

    async def _flush_buffered_local_candidates(
        self,
        token: object,
        candidates: list[dict[str, Any]],
        barrier_attempt_id: str | None,
    ) -> None:
        while candidates:
            event = candidates.pop(0)
            attempt_id = event.get("_rtc_attempt_id")
            if attempt_id is None or barrier_attempt_id is None or attempt_id == barrier_attempt_id:
                await self._emit_media_wire(event)
        self._clear_answer_barrier(token)

    async def _rollback_offer(self, result: Any) -> bool:
        task = asyncio.create_task(result.rollback())
        current = asyncio.current_task()
        interrupted = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                interrupted = True
                if current is not None:
                    current.uncancel()
        task.result()
        return interrupted

    async def _emit_conversation_event(self, event: ConvEvent) -> None:
        prepared = prepare_rtc_control_event(
            record=self.record,
            session_id=self.session_id,
            event=event,
        )
        wire = prepared.wire
        if wire is None:
            return
        emit_conversation = self._emit_conversation
        attempt = (
            (lambda: emit_conversation(event, wire)) if emit_conversation is not None else (lambda: self._emit(wire))
        )
        await deliver_wire_with_lifecycle_retry(wire, attempt, on_lost=self._log_lost_lifecycle_event)

    def _log_lost_lifecycle_event(self, event_type: str) -> None:
        logger.error(
            "Lost lifecycle event %s for RTC session %s after retry",
            event_type,
            self.session_id,
        )

    async def _pump_media_events(self) -> None:
        queue = self.record.media_events
        if queue is None:
            return
        while True:
            event = await queue.get()
            if event is None:
                return
            attempt_id = event.get("_rtc_attempt_id")
            if (
                event.get("type") == "rtc.ice_candidate"
                and self._answer_barrier_active
                and event.get("generation") == self._answer_barrier_generation
            ):
                if len(self._buffered_local_candidates) >= MAX_PENDING_REMOTE_CANDIDATES:
                    await self._emit(
                        stamp_negotiation_generation(
                            self.record,
                            {
                                "type": "rtc.signaling_error",
                                "message": "too many local RTC candidates before answer",
                            },
                            generation=self._answer_barrier_generation,
                        )
                    )
                    self._schedule_terminal_close("signaling_error")
                    return
                self._buffered_local_candidates.append(event)
                continue
            if attempt_id is not None and attempt_id != self._active_negotiation_attempt_id:
                continue
            await self._emit_media_wire(event)
            if event.get("type") == "rtc.signaling_error":
                self._schedule_terminal_close("signaling_error")
                return

    async def _emit_media_wire(self, event: dict[str, Any]) -> None:
        if "_rtc_attempt_id" in event:
            event = dict(event)
            event.pop("_rtc_attempt_id", None)
        await self._emit(event)

    async def _pump_client_events(self) -> None:
        queue = self.record.control_events
        if queue is None:
            return
        while True:
            event = await queue.get()
            if event is None:
                return
            await self._emit(event)
            if event.get("type") == "rtc.client_disconnected":
                self._schedule_terminal_close(str(event.get("reason") or "client_disconnected"))
                return

    def _schedule_terminal_close(self, reason: str) -> None:
        if self._closed or self._terminal_close_task is not None:
            return
        task = asyncio.create_task(self.close(reason=reason))
        self._terminal_close_task = task
        task.add_done_callback(self._terminal_close_finished)

    def _terminal_close_finished(self, task: asyncio.Task[None]) -> None:
        self._terminal_close_task = None
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.exception("RTC terminal cleanup failed for %s", self.session_id)

    async def close(self, *, reason: str = "transport_closed") -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            error: BaseException | None = None
            try:
                await self.conversation.close()
            except BaseException as exc:
                error = exc
            try:
                await self._registry.close_attached(self.record, orchestrator=None)
            except BaseException as exc:
                if error is None:
                    error = exc
            try:
                await self._emit(
                    {
                        "type": "rtc.session.closed",
                        "session_id": self.session_id,
                        "reason": reason,
                    }
                )
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                self._closed_event.set()
            if error is not None:
                raise error
