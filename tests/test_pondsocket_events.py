from __future__ import annotations

import pytest

from vox.operations.conversation import ConvDoneEvent, ConvResponseCreatedEvent
from vox.operations.conversation_commands import ClientEventCommand, UnknownCommand
from vox.operations.errors import SessionNotConfiguredError
from vox.operations.rtc_runtime import RtcOfferCommand
from vox.server.pondsocket_events import (
    broadcast_conversation_event_to_user,
    broadcast_rtc_client_events_to_user,
    broadcast_rtc_control_event_to_user,
    broadcast_wire_to_user,
    decline_if_channel_attached,
    handle_pondsocket_control_event,
    pondsocket_route_or_channel_session_id,
    reply_pondsocket_error,
    try_broadcast_wire_to_user,
)
from vox.server.rtc_timeline import RtcTurnTimeline


class FakeChannel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.broadcasts: list[tuple[str, dict, str]] = []

    async def broadcast_to(self, event_name: str, payload: dict, user_id: str) -> None:
        if self.fail:
            raise RuntimeError("broadcast failed")
        self.broadcasts.append((event_name, payload, user_id))


class FakeContext:
    def __init__(
        self,
        *,
        replied: bool = False,
        fail: bool = False,
        event_name: str = "response.cancel",
        payload: dict | object | None = None,
    ) -> None:
        self._replied = replied
        self.fail = fail
        self.event_name = event_name
        self.payload = {} if payload is None else payload
        self.replies: list[tuple[str, dict]] = []

    def has_replied(self) -> bool:
        return self._replied

    def get_payload(self):
        return self.payload

    async def reply(self, event_name: str, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("reply failed")
        self._replied = True
        self.replies.append((event_name, payload))


class FakeRoute:
    def __init__(self, params: dict[str, str] | None = None) -> None:
        self.params = params or {}


class FakeJoinChannel:
    def __init__(self, *, name: str, users: int) -> None:
        self.name = name
        self.users = users

    async def user_count(self) -> int:
        return self.users


class FakeJoinContext:
    def __init__(self, *, channel_name: str = "/rtc/rtc_123", users: int = 0, params: dict[str, str] | None = None):
        self.channel = FakeJoinChannel(name=channel_name, users=users)
        self.route = FakeRoute(params)
        self.declines: list[tuple[int, str]] = []

    async def decline(self, code: int, reason: str) -> None:
        self.declines.append((code, reason))


class FakeRuntime:
    def __init__(self, error: Exception | None = None) -> None:
        self.conversation = self
        self.error = error
        self.commands = []

    async def dispatch(self, command) -> None:
        if self.error is not None:
            raise self.error
        self.commands.append(command)


class FakeRtcRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.conversation = FakeRuntime(RuntimeError("RTC command bypassed its runtime"))


class FakeLogger:
    def __init__(self) -> None:
        self.exceptions: list[str] = []

    def exception(self, message: str) -> None:
        self.exceptions.append(message)


class FakeOrchestrator:
    def __init__(self, *events) -> None:
        self._events = events

    async def events(self):
        for event in self._events:
            yield event


class FakePreparedEvent:
    def __init__(self, wire: dict | None, *, done: bool = False) -> None:
        self.wire = wire
        self.done = done


class FakeControlQueue:
    def __init__(self, *events) -> None:
        self._events = list(events)

    async def get(self):
        if self._events:
            return self._events.pop(0)
        return None


class FakeRecord:
    def __init__(self, control_events) -> None:
        self.control_events = control_events


def test_pondsocket_route_or_channel_session_id_prefers_route_params():
    ctx = FakeJoinContext(channel_name="/rtc/from-channel", params={"session_id": "from-param"})

    assert pondsocket_route_or_channel_session_id(ctx) == "from-param"


def test_pondsocket_route_or_channel_session_id_falls_back_to_channel_suffix():
    ctx = FakeJoinContext(channel_name="/conversation/demo", params={})

    assert pondsocket_route_or_channel_session_id(ctx) == "demo"


@pytest.mark.asyncio
async def test_decline_if_channel_attached_declines_existing_user():
    ctx = FakeJoinContext(users=1)

    declined = await decline_if_channel_attached(ctx, "already attached")

    assert declined is True
    assert ctx.declines == [(409, "already attached")]


@pytest.mark.asyncio
async def test_decline_if_channel_attached_allows_empty_channel():
    ctx = FakeJoinContext(users=0)

    declined = await decline_if_channel_attached(ctx, "already attached")

    assert declined is False
    assert ctx.declines == []


@pytest.mark.asyncio
async def test_broadcast_wire_to_user_uses_conversation_wire_contract():
    channel = FakeChannel()

    await broadcast_wire_to_user(
        channel,
        "user-1",
        {"type": "response.created", "response_id": "resp_1", "session_id": "rtc_1"},
    )

    assert channel.broadcasts == [
        (
            "response.created",
            {"response_id": "resp_1", "session_id": "rtc_1"},
            "user-1",
        )
    ]


@pytest.mark.asyncio
async def test_try_broadcast_wire_to_user_reports_suppressed_failure():
    channel = FakeChannel(fail=True)

    sent = await try_broadcast_wire_to_user(channel, "user-1", {"type": "response.done"})

    assert sent is False
    assert channel.broadcasts == []


@pytest.mark.asyncio
async def test_broadcast_conversation_event_to_user_serializes_operation_event():
    channel = FakeChannel()
    events = (
        ConvResponseCreatedEvent(response_id="resp_1"),
        ConvDoneEvent(),
    )

    for event in events:
        await broadcast_conversation_event_to_user(
            event,
            channel=channel,
            user_id="user-1",
        )

    assert channel.broadcasts == [("response.created", {"response_id": "resp_1"}, "user-1")]


@pytest.mark.asyncio
async def test_broadcast_conversation_event_to_user_suppresses_channel_failures():
    channel = FakeChannel(fail=True)

    await broadcast_conversation_event_to_user(
        ConvResponseCreatedEvent(response_id="resp_1"),
        channel=channel,
        user_id="user-1",
    )

    assert channel.broadcasts == []


@pytest.mark.asyncio
async def test_broadcast_rtc_control_event_to_user_uses_prepared_wire_and_reports_done():
    channel = FakeChannel()
    seen = []

    def prepare_event(*, record, session_id, event):
        seen.append((record, session_id, event))
        return FakePreparedEvent(
            {"type": "rtc.event", "value": event},
            done=event == "done",
        )

    record = object()
    timeline = RtcTurnTimeline(session_id="rtc_1")
    done_values = []
    for event in ("first", "done"):
        done_values.append(
            await broadcast_rtc_control_event_to_user(
                event,
                channel=channel,
                user_id="user-1",
                record=record,
                session_id="rtc_1",
                prepare_event=prepare_event,
                timeline=timeline,
            )
        )

    assert seen == [(record, "rtc_1", "first"), (record, "rtc_1", "done")]
    assert done_values == [False, True]
    assert channel.broadcasts == [
        ("rtc.event", {"value": "first"}, "user-1"),
        ("rtc.event", {"value": "done"}, "user-1"),
    ]


@pytest.mark.asyncio
async def test_broadcast_rtc_control_event_includes_turn_timing_diagnostics():
    channel = FakeChannel()

    def prepare_event(*, record, session_id, event):
        wires = {
            "started": {"type": "input_audio_buffer.speech_started", "timestamp_ms": 10},
            "interrupted": {
                "type": "interruption.detected",
                "response_id": "resp_1",
                "reason": "partial_keyword",
            },
            "done": {"type": "response.done", "response_id": "resp_1"},
        }
        return FakePreparedEvent(wires[event], done=event == "done")

    record = object()
    timeline = RtcTurnTimeline(session_id="rtc_1")
    for event in ("started", "interrupted", "done"):
        await broadcast_rtc_control_event_to_user(
            event,
            channel=channel,
            user_id="user-1",
            record=record,
            session_id="rtc_1",
            prepare_event=prepare_event,
            timeline=timeline,
        )

    event_names = [name for name, _, _ in channel.broadcasts]
    assert event_names == [
        "input_audio_buffer.speech_started",
        "rtc.turn_timing",
        "interruption.detected",
        "rtc.turn_timing",
        "response.done",
        "rtc.turn_timing",
    ]
    interruption_timing = channel.broadcasts[3][1]["data"]
    assert interruption_timing["source_event"] == "interruption.detected"
    assert interruption_timing["response_id"] == "resp_1"
    assert interruption_timing["ms_since_speech_started"] >= 0


@pytest.mark.asyncio
async def test_broadcast_rtc_client_events_to_user_drains_control_queue_until_sentinel():
    channel = FakeChannel()
    record = FakeRecord(
        FakeControlQueue(
            {"type": "browser.event", "event": "ui.select", "payload": {"id": "a"}},
            None,
            {"type": "browser.event", "event": "ignored", "payload": {}},
        )
    )

    await broadcast_rtc_client_events_to_user(
        record=record,
        channel=channel,
        user_id="user-1",
    )

    assert channel.broadcasts == [
        (
            "browser.event",
            {"event": "ui.select", "payload": {"id": "a"}},
            "user-1",
        )
    ]


@pytest.mark.asyncio
async def test_reply_pondsocket_error_skips_already_replied_context():
    ctx = FakeContext(replied=True)

    await reply_pondsocket_error(ctx, "boom")

    assert ctx.replies == []


@pytest.mark.asyncio
async def test_reply_pondsocket_error_suppresses_reply_failures():
    ctx = FakeContext(fail=True)

    await reply_pondsocket_error(ctx, "boom")

    assert ctx.replies == []


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_replies_when_runtime_missing():
    ctx = FakeContext()

    await handle_pondsocket_control_event(
        ctx,
        runtime=None,
        missing_message="session not attached",
        error_log_message="unexpected",
        logger=FakeLogger(),
    )

    assert ctx.replies == [("error", {"message": "session not attached"})]


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_executes_conversation_command():
    ctx = FakeContext(event_name="response.create", payload={"response_id": "resp_1"})
    runtime = FakeRuntime()

    await handle_pondsocket_control_event(
        ctx,
        runtime=runtime,
        missing_message="session not attached",
        error_log_message="unexpected",
        logger=FakeLogger(),
    )

    assert ctx.replies == []
    assert runtime.commands == [UnknownCommand(name="response.create")]


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_dispatches_rtc_offer_through_rtc_runtime():
    ctx = FakeContext(
        event_name="rtc.offer",
        payload={"offer": {"type": "offer", "sdp": "offer-sdp"}},
    )
    runtime = FakeRtcRuntime()

    await handle_pondsocket_control_event(
        ctx,
        runtime=runtime,
        missing_message="RTC control session not attached",
        error_log_message="unexpected RTC control error",
        logger=FakeLogger(),
    )

    assert ctx.replies == []
    assert runtime.commands == [
        RtcOfferCommand(offer_type="offer", sdp="offer-sdp"),
    ]
    assert runtime.conversation.commands == []


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_passes_client_event_handler():
    ctx = FakeContext(event_name="client.event", payload={"event": "ping", "payload": {"n": 1}})
    runtime = FakeRuntime()

    await handle_pondsocket_control_event(
        ctx,
        runtime=runtime,
        missing_message="session not attached",
        error_log_message="unexpected",
        logger=FakeLogger(),
    )

    assert runtime.commands == [ClientEventCommand(event="ping", payload={"n": 1})]


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_replies_with_operation_error():
    ctx = FakeContext()
    runtime = FakeRuntime(SessionNotConfiguredError())

    await handle_pondsocket_control_event(
        ctx,
        runtime=runtime,
        missing_message="session not attached",
        error_log_message="unexpected",
        logger=FakeLogger(),
    )

    assert ctx.replies == [("error", {"message": "Session not configured"})]


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_logs_unexpected_errors():
    ctx = FakeContext()
    runtime = FakeRuntime(RuntimeError("boom"))
    logger = FakeLogger()

    await handle_pondsocket_control_event(
        ctx,
        runtime=runtime,
        missing_message="session not attached",
        error_log_message="unexpected control error",
        logger=logger,
    )

    assert logger.exceptions == ["unexpected control error"]
    assert ctx.replies == [("error", {"message": "boom"})]
