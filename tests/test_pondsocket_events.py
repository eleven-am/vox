from __future__ import annotations

import logging

import pytest

from vox.conversation.response_output import ResponseOutputOptions
from vox.operations.conversation import ConvDoneEvent, ConvResponseCreatedEvent
from vox.operations.conversation_commands import (
    ClientEventCommand,
    ResponseStartCommand,
    UnknownCommand,
)
from vox.operations.errors import (
    ConversationCommandError,
    InvalidConfigError,
    SessionNotConfiguredError,
)
from vox.operations.rtc_runtime import RtcCandidateCommand, RtcOfferCommand
from vox.server.pondsocket_events import (
    broadcast_conversation_event_to_user,
    broadcast_wire_to_user,
    decline_if_channel_attached,
    decode_pondsocket_command,
    deliver_wire_to_user,
    handle_pondsocket_control_event,
    pondsocket_route_or_channel_session_id,
    reply_pondsocket_error,
    try_broadcast_wire_to_user,
)


def test_decode_rtc_offer_carries_generation():
    command = decode_pondsocket_command(
        "rtc.offer",
        {"offer": {"type": "offer", "sdp": "offer-sdp"}, "generation": 4},
    )
    assert command == RtcOfferCommand(offer_type="offer", sdp="offer-sdp", generation=4)


def test_decode_rtc_offer_without_generation_is_none():
    command = decode_pondsocket_command("rtc.offer", {"offer": {"type": "offer", "sdp": "offer-sdp"}})
    assert command.generation is None


def test_decode_rtc_candidate_and_completion_carry_generation():
    candidate = decode_pondsocket_command(
        "rtc.ice_candidate",
        {
            "candidate": {
                "candidate": "candidate:1",
                "sdpMid": "0",
            },
            "generation": 7,
        },
    )
    complete = decode_pondsocket_command(
        "rtc.ice_candidate",
        {"candidate": None, "generation": 7},
    )

    assert candidate == RtcCandidateCommand(
        candidate="candidate:1",
        sdp_mid="0",
        generation=7,
    )
    assert complete == RtcCandidateCommand(candidate=None, generation=7)


def test_decode_response_start_preserves_typed_output_override():
    command = decode_pondsocket_command(
        "response.start",
        {
            "generation_id": "gen-1",
            "output": {
                "model": "qwen3-tts:0.6b-clone",
                "language": "fr",
                "speed": 0.9,
                "params": {"temperature": 0.7},
            },
        },
    )

    assert command == ResponseStartCommand(
        generation_id="gen-1",
        output=ResponseOutputOptions(
            model="qwen3-tts:0.6b-clone",
            language="fr",
            speed=0.9,
            params={"temperature": 0.7},
        ),
    )


def test_decode_response_start_rejects_unknown_output_fields():
    with pytest.raises(InvalidConfigError, match="unsupported field"):
        decode_pondsocket_command(
            "response.start",
            {"output": {"model": "kokoro-tts:v1.0", "future_field": True}},
        )


class FakeChannel:
    def __init__(self, *, fail: bool = False, fail_times: int = 0) -> None:
        self.fail = fail
        self.fail_times = fail_times
        self.attempts: list[str] = []
        self.broadcasts: list[tuple[str, dict, str]] = []

    async def broadcast_to(self, event_name: str, payload: dict, user_id: str) -> None:
        self.attempts.append(event_name)
        if self.fail:
            raise RuntimeError("broadcast failed")
        if self.fail_times > 0:
            self.fail_times -= 1
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
async def test_try_broadcast_wire_to_user_logs_warning_with_event_type(caplog):
    channel = FakeChannel(fail=True)

    with caplog.at_level(logging.WARNING, logger="vox.server.pondsocket_events"):
        sent = await try_broadcast_wire_to_user(
            channel,
            "user-1",
            {"type": "response.cancelled", "response_id": "resp_1"},
            session_id="sess_1",
        )

    assert sent is False
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "response.cancelled" in warnings[0].getMessage()
    assert "sess_1" in warnings[0].getMessage()
    assert "user-1" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_deliver_wire_to_user_retries_critical_event_once_and_delivers():
    channel = FakeChannel(fail_times=1)

    delivered = await deliver_wire_to_user(
        channel,
        "user-1",
        {"type": "response.audio.clear", "response_id": "resp_1"},
        session_id="sess_1",
    )

    assert delivered is True
    assert channel.attempts == ["response.audio.clear", "response.audio.clear"]
    assert channel.broadcasts == [("response.audio.clear", {"response_id": "resp_1"}, "user-1")]


@pytest.mark.asyncio
async def test_deliver_wire_to_user_logs_error_when_critical_event_lost(caplog):
    channel = FakeChannel(fail=True)

    with caplog.at_level(logging.WARNING, logger="vox.server.pondsocket_events"):
        delivered = await deliver_wire_to_user(
            channel,
            "user-1",
            {"type": "interruption.detected", "response_id": "resp_1"},
            session_id="sess_1",
        )

    assert delivered is False
    assert channel.attempts == ["interruption.detected", "interruption.detected"]
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "interruption.detected" in errors[0].getMessage()
    assert "sess_1" in errors[0].getMessage()


@pytest.mark.asyncio
async def test_deliver_wire_to_user_does_not_retry_non_critical_events(caplog):
    channel = FakeChannel(fail=True)

    with caplog.at_level(logging.WARNING, logger="vox.server.pondsocket_events"):
        delivered = await deliver_wire_to_user(
            channel,
            "user-1",
            {"type": "response.audio.delta", "response_id": "resp_1", "delta": "AAAA"},
            session_id="sess_1",
        )

    assert delivered is False
    assert channel.attempts == ["response.audio.delta"]
    assert [record for record in caplog.records if record.levelno == logging.ERROR] == []


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

    assert ctx.replies == [
        (
            "error",
            {"message": "session not attached", "code": "command_invalid", "recoverable": True},
        )
    ]


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

    assert ctx.replies == [
        (
            "error",
            {"message": "Session not configured", "code": "command_invalid", "recoverable": True},
        )
    ]


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_preserves_offer_generation_on_error():
    ctx = FakeContext(
        event_name="rtc.offer",
        payload={
            "offer": {"type": "offer", "sdp": "offer-sdp"},
            "restart": True,
            "generation": 17,
        },
    )
    runtime = FakeRuntime(SessionNotConfiguredError())

    await handle_pondsocket_control_event(
        ctx,
        runtime=runtime,
        missing_message="session not attached",
        error_log_message="unexpected",
        logger=FakeLogger(),
    )

    assert ctx.replies == [
        (
            "error",
            {
                "message": "Session not configured",
                "code": "command_invalid",
                "recoverable": True,
                "generation": 17,
            },
        )
    ]


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_replies_with_typed_conversation_error():
    ctx = FakeContext(event_name="response.delta", payload={"delta": "late"})
    runtime = FakeRuntime(
        ConversationCommandError(
            "stale response generation",
            code="response_stale_generation",
            recoverable=True,
            generation_id="gen-a",
        )
    )

    await handle_pondsocket_control_event(
        ctx,
        runtime=runtime,
        missing_message="session not attached",
        error_log_message="unexpected",
        logger=FakeLogger(),
    )

    assert ctx.replies == [
        (
            "error",
            {
                "message": "stale response generation",
                "code": "response_stale_generation",
                "recoverable": True,
                "generation_id": "gen-a",
            },
        )
    ]


@pytest.mark.asyncio
async def test_handle_pondsocket_control_event_replies_with_fatal_error_fields():
    ctx = FakeContext()
    runtime = FakeRuntime(
        ConversationCommandError(
            "session dead",
            code="session_failed",
            recoverable=False,
        )
    )

    await handle_pondsocket_control_event(
        ctx,
        runtime=runtime,
        missing_message="session not attached",
        error_log_message="unexpected",
        logger=FakeLogger(),
    )

    assert ctx.replies == [
        (
            "error",
            {"message": "session dead", "code": "session_failed", "recoverable": False},
        )
    ]


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
    assert ctx.replies == [("error", {"message": "boom", "code": "command_invalid", "recoverable": True})]
