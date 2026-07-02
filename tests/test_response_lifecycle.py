from __future__ import annotations

from vox.conversation.response_lifecycle import ConversationResponseLifecycle
from vox.conversation.response_stream import ResponseStream


def test_response_lifecycle_creates_sequential_streams_and_tracks_active_id():
    lifecycle = ConversationResponseLifecycle()

    first = lifecycle.start_stream(allow_interruptions=False)
    second = lifecycle.start_stream(allow_interruptions=True)

    assert first.response_id == "resp_1"
    assert first.allow_interruptions is False
    assert second.response_id == "resp_2"
    assert lifecycle.stream is second
    assert lifecycle.active_response_id == "resp_2"
    assert lifecycle.last_cancelled_response_id is None


def test_response_lifecycle_reuses_open_uncommitted_stream_only():
    lifecycle = ConversationResponseLifecycle()
    stream = lifecycle.start_stream()

    assert lifecycle.open_uncommitted_stream() is stream

    stream.mark_committed()

    assert lifecycle.open_uncommitted_stream() is None


def test_response_lifecycle_preserves_pending_done_stream_after_task_finish():
    lifecycle = ConversationResponseLifecycle()
    stream = lifecycle.start_stream()
    stream.pending_done = True

    lifecycle.clear_finished_stream_if_current(stream)

    assert lifecycle.stream is stream
    assert lifecycle.active_response_id == stream.response_id


def test_response_lifecycle_captures_cancelled_id_and_clears_active_stream():
    lifecycle = ConversationResponseLifecycle()
    stream = lifecycle.start_stream()

    assert lifecycle.remember_cancelled_response() == stream.response_id
    lifecycle.clear_active_response(stream)

    assert lifecycle.stream is None
    assert lifecycle.active_response_id is None
    assert lifecycle.last_cancelled_response_id == "resp_1"
    assert lifecycle.active_or_cancelled_response_id() == "resp_1"


def test_response_lifecycle_ignores_stale_stream_completion():
    lifecycle = ConversationResponseLifecycle()
    active = lifecycle.start_stream()
    stale = ResponseStream.create(response_id="resp_old")

    lifecycle.finish_stream_if_current(stale)

    assert lifecycle.stream is active
    assert lifecycle.active_response_id == active.response_id
