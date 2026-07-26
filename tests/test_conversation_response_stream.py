from __future__ import annotations

import asyncio

import pytest

from vox.conversation.response_output import ResponseOutputConfig
from vox.conversation.response_stream import RESPONSE_STREAM_QUEUE_MAX, AppendResult, ResponseStream

OUTPUT = ResponseOutputConfig(model="tts:1", voice=None, language="en")


def test_response_stream_create_uses_bounded_queue_and_response_metadata():
    stream = ResponseStream.create(
        response_id="resp_1",
        output=OUTPUT,
        allow_interruptions=False,
    )

    assert stream.response_id == "resp_1"
    assert stream.allow_interruptions is False
    assert stream.queue.maxsize == RESPONSE_STREAM_QUEUE_MAX
    assert stream.committed is False


@pytest.mark.asyncio
async def test_response_stream_append_and_next_text_records_text_parts():
    stream = ResponseStream.create(response_id="resp_1", output=OUTPUT)

    await stream.append_text("hello")
    await stream.append_text(" there")

    assert await stream.next_text() == "hello"
    assert await stream.next_text() == " there"
    assert stream.text_parts == ["hello", " there"]


@pytest.mark.asyncio
async def test_response_stream_commit_marker_ends_text_iteration():
    stream = ResponseStream.create(response_id="resp_1", output=OUTPUT)

    assert stream.mark_committed() is True
    assert stream.mark_committed() is False
    await stream.enqueue_end()

    assert await stream.next_text() is None


@pytest.mark.asyncio
async def test_response_stream_bounded_queue_blocks_when_full():
    stream = ResponseStream.create(response_id="resp_1", output=OUTPUT)
    stream.queue = asyncio.Queue(maxsize=1)

    await stream.append_text("first")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stream.append_text("second"), timeout=0.05)


@pytest.mark.asyncio
async def test_closing_full_response_stream_wakes_blocked_append_and_commit():
    stream = ResponseStream.create(response_id="resp_1", output=OUTPUT)
    stream.queue = asyncio.Queue(maxsize=1)
    await stream.append_text("first")

    append = asyncio.create_task(stream.append_text("second"))
    commit = asyncio.create_task(stream.enqueue_end())
    await asyncio.sleep(0)
    assert not append.done()
    assert not commit.done()

    stream.close()

    results = await asyncio.wait_for(asyncio.gather(append, commit), timeout=0.1)
    assert results == [AppendResult.STREAM_ENDED, AppendResult.STREAM_ENDED]
    assert stream.queue.empty()


@pytest.mark.asyncio
async def test_closed_response_stream_rejects_append_and_end():
    stream = ResponseStream.create(response_id="resp_1", output=OUTPUT)

    stream.close()

    assert stream.closed is True
    assert await stream.append_text("late") is AppendResult.STREAM_ENDED
    assert await stream.enqueue_end() is AppendResult.STREAM_ENDED
    assert stream.queue.empty()


@pytest.mark.asyncio
async def test_open_response_stream_reports_successful_append_and_end():
    stream = ResponseStream.create(response_id="resp_1", output=OUTPUT)

    assert await stream.append_text("hello") is AppendResult.ACCEPTED
    assert await stream.enqueue_end() is AppendResult.ACCEPTED
    assert AppendResult.ACCEPTED.is_accepted
    assert not AppendResult.STREAM_ENDED.is_accepted


def test_response_stream_assistant_context_prefers_heard_text():
    stream = ResponseStream.create(response_id="resp_1", output=OUTPUT)
    stream.text_parts.extend(["unheard ", "text"])

    assert stream.assistant_context_text() == "unheard text"

    stream.add_heard_text("heard ")
    stream.add_heard_text("text")

    assert stream.assistant_context_text() == "heard text"
