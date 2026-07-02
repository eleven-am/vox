from __future__ import annotations

import pytest

from vox.operations.errors import NoDefaultModelError, SessionNotConfiguredError
from vox.operations.streaming_reporting import (
    StreamingOperationErrorReporter,
    streaming_operation_error_message,
)


class FakeStreamingSession(StreamingOperationErrorReporter):
    def __init__(self) -> None:
        self.errors: list[str] = []

    async def report_error(self, message: str) -> None:
        self.errors.append(message)


class CustomOperationErrorSession(StreamingOperationErrorReporter):
    def __init__(self) -> None:
        self.errors: list[str] = []

    async def report_operation_error(self, exc) -> None:
        self.errors.append(f"custom: {exc}")


@pytest.mark.asyncio
async def test_streaming_operation_error_reporter_returns_true_for_success():
    session = FakeStreamingSession()
    called = False

    async def action() -> None:
        nonlocal called
        called = True

    ok = await session.run_or_report_operation_error(action)

    assert ok is True
    assert called is True
    assert session.errors == []


@pytest.mark.asyncio
async def test_streaming_operation_error_reporter_reports_operation_errors():
    session = FakeStreamingSession()

    async def action() -> None:
        raise SessionNotConfiguredError()

    ok = await session.run_or_report_operation_error(action)

    assert ok is False
    assert session.errors == ["Session not configured"]


def test_streaming_operation_error_message_names_missing_default_model_type():
    assert (
        streaming_operation_error_message(NoDefaultModelError("tts"))
        == "No TTS model specified and no default TTS model available"
    )


@pytest.mark.asyncio
async def test_streaming_operation_error_reporter_allows_custom_error_policy():
    session = CustomOperationErrorSession()

    async def action() -> None:
        raise SessionNotConfiguredError()

    ok = await session.run_or_report_operation_error(action)

    assert ok is False
    assert session.errors == ["custom: Session not configured"]
