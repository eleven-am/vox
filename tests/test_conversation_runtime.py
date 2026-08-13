from __future__ import annotations

import asyncio

import pytest

from vox.conversation.response_output import ResponseOutputOptions
from vox.operations.conversation import (
    ConvDoneEvent,
    ConversationSessionConfig,
    ConvResponseCreatedEvent,
)
from vox.operations.conversation_commands import (
    AudioAppendCommand,
    ClientEventCommand,
    ResponseCommitCommand,
    ResponseDeltaCommand,
    ResponseReplaceTextCommand,
    ResponseStartCommand,
    SessionUpdateCommand,
    UnknownCommand,
)
from vox.operations.conversation_runtime import ConversationRuntime
from vox.operations.errors import InvalidConfigError, SessionAlreadyConfiguredError


class RuntimeOrchestratorSpy:
    def __init__(self, *, configured: bool = True) -> None:
        self.config = object() if configured else None
        self.calls: list[tuple[str, tuple, dict]] = []
        self.events_queue: asyncio.Queue = asyncio.Queue()
        self.end_count = 0
        self.close_count = 0

    async def start_session(self, config) -> None:
        if self.config is not None:
            raise SessionAlreadyConfiguredError()
        self.config = config
        self.calls.append(("start_session", (config,), {}))

    async def ingest_pcm16(self, pcm16: bytes, sample_rate: int | None = None) -> None:
        self.calls.append(("ingest_pcm16", (pcm16,), {"sample_rate": sample_rate}))

    async def start_response(
        self,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
        supersedes_generation_id: str | None = None,
        output: ResponseOutputOptions | None = None,
    ) -> None:
        self.calls.append(
            (
                "start_response",
                (),
                {
                    "allow_interruptions": allow_interruptions,
                    "generation_id": generation_id,
                    "supersedes_generation_id": supersedes_generation_id,
                    "output": output,
                },
            )
        )

    async def append_response_text(
        self,
        text: str,
        *,
        allow_interruptions: bool = True,
        generation_id: str | None = None,
    ) -> None:
        self.calls.append(
            (
                "append_response_text",
                (text,),
                {
                    "allow_interruptions": allow_interruptions,
                    "generation_id": generation_id,
                },
            )
        )

    async def replace_response_text(self, text: str, *, allow_interruptions: bool = True) -> None:
        self.calls.append(
            (
                "replace_response_text",
                (text,),
                {"allow_interruptions": allow_interruptions},
            )
        )

    async def commit_response(self, *, generation_id: str | None = None) -> None:
        self.calls.append(("commit_response", (), {"generation_id": generation_id}))

    async def cancel_response(self, *, generation_id: str | None = None) -> None:
        self.calls.append(("cancel_response", (), {"generation_id": generation_id}))

    async def end_of_stream(self, *, flush_response: bool = True) -> None:
        self.end_count += 1
        self.calls.append(("end_of_stream", (), {"flush_response": flush_response}))
        await self.events_queue.put(ConvDoneEvent())

    async def close(self) -> None:
        self.close_count += 1

    async def events(self):
        while True:
            event = await self.events_queue.get()
            yield event
            if isinstance(event, ConvDoneEvent):
                return


@pytest.mark.asyncio
async def test_runtime_dispatches_typed_commands_and_preserves_pcm_identity():
    orchestrator = RuntimeOrchestratorSpy(configured=False)
    runtime = ConversationRuntime(orchestrator)
    config = ConversationSessionConfig(stt_model="stt:1", tts_model="tts:1")

    await runtime.dispatch(SessionUpdateCommand(config=config))
    await runtime.dispatch(AudioAppendCommand(pcm16=b"\x01\x02\x03\x04", sample_rate=16_000))
    await runtime.dispatch(ResponseStartCommand(allow_interruptions=False, generation_id="generation-7"))
    await runtime.dispatch(
        ResponseDeltaCommand(
            text="hello",
            allow_interruptions=False,
            generation_id="generation-7",
        )
    )
    await runtime.dispatch(ResponseCommitCommand(generation_id="generation-7"))
    await runtime.dispatch(ResponseReplaceTextCommand(text="replacement"))

    assert orchestrator.calls[:6] == [
        ("start_session", (config,), {}),
        ("ingest_pcm16", (b"\x01\x02\x03\x04",), {"sample_rate": 16_000}),
        (
            "start_response",
            (),
            {
                "allow_interruptions": False,
                "generation_id": "generation-7",
                "supersedes_generation_id": None,
                "output": None,
            },
        ),
        (
            "append_response_text",
            ("hello",),
            {"allow_interruptions": False, "generation_id": "generation-7"},
        ),
        ("commit_response", (), {"generation_id": "generation-7"}),
        ("replace_response_text", ("replacement",), {"allow_interruptions": True}),
    ]


@pytest.mark.asyncio
async def test_runtime_requires_configuration_before_conversation_commands():
    runtime = ConversationRuntime(RuntimeOrchestratorSpy(configured=False))

    with pytest.raises(InvalidConfigError, match="send session.update first"):
        await runtime.dispatch(ResponseStartCommand())


@pytest.mark.asyncio
async def test_runtime_routes_client_events_before_configuration():
    received = []
    runtime = ConversationRuntime(
        RuntimeOrchestratorSpy(configured=False),
        client_event_handler=lambda event, payload: received.append((event, payload)),
    )

    await runtime.dispatch(ClientEventCommand(event="ui.toast", payload={"message": "hi"}))

    assert received == [("ui.toast", {"message": "hi"})]


@pytest.mark.asyncio
async def test_runtime_rejects_audio_on_control_only_transport():
    runtime = ConversationRuntime(
        RuntimeOrchestratorSpy(),
        allow_input_audio=False,
        unknown_message_label="unknown RTC control message type",
    )

    with pytest.raises(
        InvalidConfigError,
        match="unknown RTC control message type: 'input_audio_buffer.append'",
    ):
        await runtime.dispatch(AudioAppendCommand(pcm16=b"pcm"))


@pytest.mark.asyncio
async def test_runtime_preserves_transport_specific_unknown_error_label():
    runtime = ConversationRuntime(
        RuntimeOrchestratorSpy(),
        unknown_message_label="unknown conversation message type",
    )

    with pytest.raises(
        InvalidConfigError,
        match="unknown conversation message type: 'bogus'",
    ):
        await runtime.dispatch(UnknownCommand(name="bogus"))


@pytest.mark.asyncio
async def test_runtime_rejects_empty_response_text_after_configuration():
    runtime = ConversationRuntime(RuntimeOrchestratorSpy())

    with pytest.raises(InvalidConfigError, match="response.delta requires 'delta' text"):
        await runtime.dispatch(ResponseDeltaCommand(text=""))
    with pytest.raises(InvalidConfigError, match="response.replace_text requires 'text'"):
        await runtime.dispatch(ResponseReplaceTextCommand(text=""))


@pytest.mark.asyncio
async def test_runtime_pumps_events_and_closes_every_resource_exactly_once():
    orchestrator = RuntimeOrchestratorSpy()
    runtime = ConversationRuntime(orchestrator, flush_response_on_close=False)
    received = []
    event_task = runtime.start_event_pump(lambda event: _append_event(received, event))
    background_started = asyncio.Event()

    async def background() -> None:
        background_started.set()
        await asyncio.Event().wait()

    background_task = runtime.start_background_task(background())
    await background_started.wait()
    await orchestrator.events_queue.put(ConvResponseCreatedEvent(response_id="resp_1"))

    await asyncio.gather(runtime.close(), runtime.close(), runtime.end_input())

    assert received == [ConvResponseCreatedEvent(response_id="resp_1"), ConvDoneEvent()]
    assert orchestrator.end_count == 1
    assert orchestrator.close_count == 1
    assert event_task.done()
    assert background_task.done()
    assert ("end_of_stream", (), {"flush_response": False}) in orchestrator.calls


@pytest.mark.asyncio
async def test_runtime_cleans_tasks_and_orchestrator_when_end_input_fails():
    class FailingEndOrchestrator(RuntimeOrchestratorSpy):
        async def end_of_stream(self, *, flush_response: bool = True) -> None:
            self.end_count += 1
            raise RuntimeError("end failed")

    orchestrator = FailingEndOrchestrator()
    runtime = ConversationRuntime(orchestrator)
    event_task = runtime.start_event_pump(lambda event: _append_event([], event))
    background_task = runtime.start_background_task(asyncio.Event().wait())

    with pytest.raises(RuntimeError, match="end failed"):
        await runtime.close()

    assert orchestrator.end_count == 1
    assert orchestrator.close_count == 1
    assert event_task.done()
    assert background_task.done()


@pytest.mark.asyncio
async def test_runtime_close_caller_cancellation_does_not_cancel_owned_cleanup():
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class BlockingCloseOrchestrator(RuntimeOrchestratorSpy):
        async def close(self) -> None:
            close_started.set()
            await release_close.wait()
            await super().close()

    orchestrator = BlockingCloseOrchestrator()
    runtime = ConversationRuntime(orchestrator)
    runtime.start_event_pump(lambda event: _append_event([], event))
    close_task = asyncio.create_task(runtime.close())

    await close_started.wait()
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(close_task, timeout=0.05)
    assert runtime._close_task is not None
    assert runtime._close_task.done() is False

    release_close.set()
    await asyncio.wait_for(runtime._close_task, timeout=1)

    assert orchestrator.close_count == 1
    assert runtime._close_task.done()


@pytest.mark.asyncio
async def test_runtime_releases_completed_background_task_references():
    runtime = ConversationRuntime(RuntimeOrchestratorSpy())

    tasks = [runtime.start_background_task(asyncio.sleep(0)) for _ in range(100)]
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)

    assert runtime._background_tasks == set()


@pytest.mark.asyncio
async def test_runtime_close_waits_for_cancellation_resistant_background_tasks(monkeypatch):
    import vox.operations.conversation_runtime as runtime_module

    original_reap = runtime_module.reap_task
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def quick_reap(task):
        await original_reap(task, timeout=0.01)

    async def resistant() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    monkeypatch.setattr(runtime_module, "reap_task", quick_reap)
    runtime = ConversationRuntime(RuntimeOrchestratorSpy())
    runtime.start_event_pump(lambda event: _append_event([], event))
    task = runtime.start_background_task(resistant())
    close_task = asyncio.create_task(runtime.close())

    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
    await asyncio.sleep(0.02)

    assert close_task.done() is False
    assert task in runtime._background_tasks

    release.set()
    await asyncio.wait_for(close_task, timeout=1)

    assert task.done()
    assert runtime._background_tasks == set()


async def _append_event(received: list, event) -> None:
    received.append(event)
