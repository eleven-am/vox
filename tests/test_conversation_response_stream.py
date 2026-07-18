from __future__ import annotations

import asyncio

import pytest

from vox.conversation.response_stream import RESPONSE_STREAM_QUEUE_MAX, ResponseStream


def test_response_stream_create_uses_bounded_queue_and_response_metadata():
    stream = ResponseStream.create(response_id="resp_1", allow_interruptions=False)

    assert stream.response_id == "resp_1"
    assert stream.allow_interruptions is False
    assert stream.queue.maxsize == RESPONSE_STREAM_QUEUE_MAX
    assert stream.committed is False


@pytest.mark.asyncio
async def test_response_stream_append_and_next_text_records_text_parts():
    stream = ResponseStream.create(response_id="resp_1")

    await stream.append_text("hello")
    await stream.append_text(" there")

    assert await stream.next_text() == "hello"
    assert await stream.next_text() == " there"
    assert stream.text_parts == ["hello", " there"]


@pytest.mark.asyncio
async def test_response_stream_commit_marker_ends_text_iteration():
    stream = ResponseStream.create(response_id="resp_1")

    assert stream.mark_committed() is True
    assert stream.mark_committed() is False
    await stream.enqueue_end()

    assert await stream.next_text() is None


@pytest.mark.asyncio
async def test_response_stream_bounded_queue_blocks_when_full():
    stream = ResponseStream.create(response_id="resp_1")
    stream.queue = asyncio.Queue(maxsize=1)

    await stream.append_text("first")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stream.append_text("second"), timeout=0.05)


@pytest.mark.asyncio
async def test_closed_response_stream_rejects_append_and_end():
    stream = ResponseStream.create(response_id="resp_1")

    stream.close()

    assert stream.closed is True
    assert await stream.append_text("late") is False
    assert await stream.enqueue_end() is False
    assert stream.queue.empty()


@pytest.mark.asyncio
async def test_open_response_stream_reports_successful_append_and_end():
    stream = ResponseStream.create(response_id="resp_1")

    assert await stream.append_text("hello") is True
    assert await stream.enqueue_end() is True


def test_response_stream_assistant_context_prefers_heard_text():
    stream = ResponseStream.create(response_id="resp_1")
    stream.text_parts.extend(["unheard ", "text"])

    assert stream.assistant_context_text() == "unheard text"

    stream.add_heard_text("heard ")
    stream.add_heard_text("text")

    assert stream.assistant_context_text() == "heard text"
