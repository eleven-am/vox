from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from vox.core.tasks import drain_task, reap_task
from vox.operations.conversation import ConversationOrchestrator, ConvEvent
from vox.operations.conversation_commands import (
    AudioAppendCommand,
    ClientEventCommand,
    ConversationCommand,
    ResponseCancelCommand,
    ResponseCommitCommand,
    ResponseDeltaCommand,
    ResponseReplaceTextCommand,
    ResponseStartCommand,
    SessionUpdateCommand,
    UnknownCommand,
)
from vox.operations.errors import (
    InvalidConfigError,
    SessionAlreadyConfiguredError,
)

EventHandler = Callable[[ConvEvent], Awaitable[None]]
ClientEventHandler = Callable[[str, Any], Awaitable[None] | None]


class ConversationRuntime:
    """Own one transport-neutral conversation and its asynchronous lifecycle."""

    def __init__(
        self,
        orchestrator: ConversationOrchestrator,
        *,
        allow_input_audio: bool = True,
        client_event_handler: ClientEventHandler | None = None,
        require_config_message: str = "send session.update first",
        already_configured_message: str = "session already configured",
        unknown_message_label: str = "unknown message type",
        flush_response_on_close: bool = True,
    ) -> None:
        self.orchestrator = orchestrator
        self._allow_input_audio = allow_input_audio
        self._client_event_handler = client_event_handler
        self._require_config_message = require_config_message
        self._already_configured_message = already_configured_message
        self._unknown_message_label = unknown_message_label
        self._flush_response_on_close = flush_response_on_close
        self._event_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._end_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False

    def start_event_pump(self, handler: EventHandler) -> asyncio.Task[None]:
        if self._event_task is not None:
            raise RuntimeError("conversation event pump already started")
        self._event_task = asyncio.create_task(self._pump_events(handler))
        return self._event_task

    def start_background_task(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        if self._closing:
            coroutine.close()
            raise RuntimeError("conversation runtime is closing")
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)

        def completed(done: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)
        return task

    async def dispatch(self, command: ConversationCommand) -> None:
        if isinstance(command, SessionUpdateCommand):
            try:
                await self.orchestrator.start_session(command.config)
            except SessionAlreadyConfiguredError as exc:
                raise InvalidConfigError(self._already_configured_message) from exc
            return

        if isinstance(command, ClientEventCommand) and self._client_event_handler is not None:
            result = self._client_event_handler(command.event, command.payload)
            if result is not None:
                await result
            return

        if self.orchestrator.config is None:
            raise InvalidConfigError(self._require_config_message)

        if isinstance(command, AudioAppendCommand):
            if not self._allow_input_audio:
                self._raise_unknown("input_audio_buffer.append")
            await self.orchestrator.ingest_pcm16(command.pcm16, sample_rate=command.sample_rate)
            return
        if isinstance(command, ResponseStartCommand):
            await self.orchestrator.start_response(
                allow_interruptions=command.allow_interruptions,
                generation_id=command.generation_id,
                output=command.output,
            )
            return
        if isinstance(command, ResponseDeltaCommand):
            if not command.text:
                raise InvalidConfigError("response.delta requires 'delta' text")
            await self.orchestrator.append_response_text(
                command.text,
                allow_interruptions=command.allow_interruptions,
                generation_id=command.generation_id,
            )
            return
        if isinstance(command, ResponseCommitCommand):
            await self.orchestrator.commit_response(generation_id=command.generation_id)
            return
        if isinstance(command, ResponseCancelCommand):
            await self.orchestrator.cancel_response(generation_id=command.generation_id)
            return
        if isinstance(command, ResponseReplaceTextCommand):
            if not command.text:
                raise InvalidConfigError("response.replace_text requires 'text'")
            await self.orchestrator.replace_response_text(
                command.text,
                allow_interruptions=command.allow_interruptions,
            )
            return
        if isinstance(command, ClientEventCommand):
            self._raise_unknown("client.event")
        if isinstance(command, UnknownCommand):
            self._raise_unknown(command.name)
        raise TypeError(f"unsupported conversation command: {type(command).__name__}")

    async def end_input(self) -> None:
        async with self._lifecycle_lock:
            if self._end_task is None:
                self._end_task = asyncio.create_task(
                    self.orchestrator.end_of_stream(
                        flush_response=self._flush_response_on_close,
                    )
                )
            task = self._end_task
        await self._await_owned_task(task)

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_once())
            task = self._close_task
        await self._await_owned_task(task)

    @staticmethod
    async def _await_owned_task(task: asyncio.Task[None]) -> None:
        await asyncio.shield(task)

    async def _pump_events(self, handler: EventHandler) -> None:
        async for event in self.orchestrator.events():
            await handler(event)

    async def _close_once(self) -> None:
        self._closing = True
        end_error: BaseException | None = None
        try:
            await self.end_input()
        except BaseException as exc:
            end_error = exc
        finally:
            if end_error is None:
                await drain_task(self._event_task)
            else:
                await reap_task(self._event_task)
            tasks = tuple(self._background_tasks)
            for task in tasks:
                await reap_task(task)
            await self.orchestrator.close()
            owned = tuple(task for task in (self._event_task, *tasks) if task is not None and not task.done())
            if owned:
                await asyncio.gather(*owned, return_exceptions=True)
        if end_error is not None:
            raise end_error

    def _raise_unknown(self, name: str) -> None:
        raise InvalidConfigError(f"{self._unknown_message_label}: {name!r}")
