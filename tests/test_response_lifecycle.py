from __future__ import annotations

from vox.conversation.response_lifecycle import ConversationResponseLifecycle, TerminalRecord
from vox.conversation.response_output import ResponseOutputConfig
from vox.conversation.response_stream import AppendResult, ResponseStream

OUTPUT = ResponseOutputConfig(model="tts:1", voice=None, language="en")


def test_response_lifecycle_creates_sequential_streams_and_stamps_generation():
    lifecycle = ConversationResponseLifecycle()

    first = lifecycle.start_stream(
        output=OUTPUT,
        allow_interruptions=False,
        generation_id="gen-1",
    )
    second = lifecycle.start_stream(output=OUTPUT, allow_interruptions=True)

    assert first.response_id == "resp_1"
    assert first.allow_interruptions is False
    assert first.generation_id == "gen-1"
    assert second.response_id == "resp_2"
    assert second.generation_id is None
    assert lifecycle.stream is second
    assert lifecycle.response_active


def test_response_lifecycle_reuses_open_uncommitted_stream_only():
    lifecycle = ConversationResponseLifecycle()
    stream = lifecycle.start_stream(output=OUTPUT)

    assert lifecycle.open_uncommitted_stream() is stream

    stream.mark_committed()

    assert lifecycle.open_uncommitted_stream() is None


def test_terminalize_writes_frozen_record_closes_and_clears_the_stream():
    lifecycle = ConversationResponseLifecycle()
    stream = lifecycle.start_stream(output=OUTPUT, generation_id="gen-1")

    record = lifecycle.terminalize(stream, "cancelled")

    assert record == TerminalRecord(response_id="resp_1", generation_id="gen-1", reason="cancelled")
    assert stream.closed
    assert lifecycle.stream is None
    assert lifecycle.terminal is record
    assert not lifecycle.response_active


def test_double_terminalize_returns_none_and_keeps_the_first_record():
    lifecycle = ConversationResponseLifecycle()
    stream = lifecycle.start_stream(output=OUTPUT, generation_id="gen-1")

    first = lifecycle.terminalize(stream, "cancelled")
    second = lifecycle.terminalize(stream, "failed")

    assert first is not None
    assert second is None
    assert lifecycle.terminal is first
    assert lifecycle.terminal.reason == "cancelled"


def test_terminalize_ignores_stale_stream_and_cannot_kill_a_successor():
    lifecycle = ConversationResponseLifecycle()
    stale = lifecycle.start_stream(output=OUTPUT, generation_id="gen-old")
    lifecycle.terminalize(stale, "cancelled")
    live = lifecycle.start_stream(output=OUTPUT, generation_id="gen-new")

    record = lifecycle.terminalize(stale, "failed")

    assert record is None
    assert lifecycle.stream is live
    assert not live.closed
    assert lifecycle.terminal is None
    assert lifecycle.response_active


def test_detached_stream_object_cannot_terminalize_the_live_stream():
    lifecycle = ConversationResponseLifecycle()
    live = lifecycle.start_stream(output=OUTPUT)
    detached = ResponseStream.create(response_id=live.response_id, output=OUTPUT)

    record = lifecycle.terminalize(detached, "done")

    assert record is None
    assert lifecycle.stream is live
    assert not live.closed
    assert lifecycle.terminal is None


def test_terminal_record_is_cleared_only_by_the_next_successful_start():
    lifecycle = ConversationResponseLifecycle()
    stream = lifecycle.start_stream(output=OUTPUT, generation_id="gen-1")
    record = lifecycle.terminalize(stream, "done")

    assert lifecycle.terminal is record

    lifecycle.terminalize(stream, "cancelled")
    assert lifecycle.terminal is record

    lifecycle.start_stream(output=OUTPUT, generation_id="gen-2")
    assert lifecycle.terminal is None


def test_appendable_stream_reasons_derive_from_stream_state():
    lifecycle = ConversationResponseLifecycle()

    assert lifecycle.appendable_stream("resp_1") is AppendResult.NO_ACTIVE_RESPONSE

    stream = lifecycle.start_stream(output=OUTPUT)
    assert lifecycle.appendable_stream(stream.response_id) is stream
    assert lifecycle.appendable_stream("resp_other") is AppendResult.RESPONSE_MISMATCH

    stream.mark_committed()
    assert lifecycle.appendable_stream(stream.response_id) is AppendResult.RESPONSE_COMMITTED

    lifecycle.terminalize(stream, "done")
    assert lifecycle.appendable_stream(stream.response_id) is AppendResult.NO_ACTIVE_RESPONSE
