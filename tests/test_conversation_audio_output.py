from __future__ import annotations

import asyncio

import pytest

from vox.conversation.audio_output import PendingAudio, ResponseAudioOutput


def test_audio_output_sequences_reset_per_response():
    output = ResponseAudioOutput()

    assert output.next_sequence() == 1
    assert output.next_sequence() == 2

    output.reset_for_response()

    assert output.next_sequence() == 1


@pytest.mark.parametrize("with_pending", [False, True])
def test_audio_output_rejects_response_reset_while_suspension_is_owned(with_pending):
    output = ResponseAudioOutput()
    output.pause(17)
    if with_pending:
        output.hold(b"held", 16_000, 1)

    with pytest.raises(RuntimeError, match="suspension owner=17"):
        output.reset_for_response()

    assert output.paused
    assert output.pause_owner == 17
    assert output.pending_count == int(with_pending)


def test_audio_output_rejects_response_reset_with_unowned_pending_audio():
    output = ResponseAudioOutput()
    output.hold(b"held", 16_000, 1)

    with pytest.raises(RuntimeError, match="pending audio=1"):
        output.reset_for_response()

    assert output.pending_count == 1


def test_audio_output_holds_and_pops_pending_batches_in_order():
    output = ResponseAudioOutput()

    output.pause()

    assert output.paused
    assert output.hold_if_paused(b"a", 16_000, 1)
    assert output.hold_if_paused(b"b", 24_000, 2)

    assert output.pending_count == 2
    assert output.pop_pending_batch() == [
        PendingAudio(b"a", 16_000, 1),
        PendingAudio(b"b", 24_000, 2),
    ]
    assert output.pending_count == 0


def test_audio_output_clear_pending_and_clear_all():
    output = ResponseAudioOutput(pace_to_playout=True)
    output.hold(b"a", 16_000, 1)
    output.mark_playout(b"\0" * 3200, 16_000)
    assert output.pending_count == 1
    assert output.playout_delay_s() > 0

    output.clear_pending()
    assert output.pending_count == 0
    assert output.playout_delay_s() > 0

    output.clear_all()
    assert output.pending_count == 0
    assert output.playout_delay_s() == 0


def test_audio_output_ignores_playout_when_pacing_disabled_or_audio_invalid():
    output = ResponseAudioOutput(pace_to_playout=False)
    output.mark_playout(b"\0" * 3200, 16_000)
    assert output.playout_delay_s() == 0

    paced = ResponseAudioOutput(pace_to_playout=True)
    paced.mark_playout(b"", 16_000)
    paced.mark_playout(b"\0" * 3200, 0)
    assert paced.playout_delay_s() == 0


def test_audio_output_resume_keeps_pause_active_until_finished():
    output = ResponseAudioOutput()

    output.pause()
    output.hold(b"a", 16_000, 1)

    assert output.pop_pending_batch() == [PendingAudio(b"a", 16_000, 1)]
    assert output.paused
    assert output.hold_if_paused(b"b", 16_000, 2)
    assert output.pop_pending_batch() == [PendingAudio(b"b", 16_000, 2)]

    output.finish_resume()

    assert not output.paused
    assert not output.hold_if_paused(b"c", 16_000, 3)


def test_audio_output_pending_resume_batches_keep_pause_active_until_finished():
    output = ResponseAudioOutput()
    output.pause()
    output.hold(b"a", 16_000, 1)

    batches = []
    for batch in output.pending_resume_batches():
        batches.append(batch)
        if batch[0].audio == b"a":
            assert output.paused
            assert output.hold_if_paused(b"b", 16_000, 2)

    assert batches == [
        [PendingAudio(b"a", 16_000, 1)],
        [PendingAudio(b"b", 16_000, 2)],
    ]
    assert output.paused

    output.finish_resume()

    assert not output.paused


def test_audio_output_flush_clears_pause_pending_and_playout():
    output = ResponseAudioOutput(pace_to_playout=True)
    output.pause()
    output.hold(b"a", 16_000, 1)
    output.mark_playout(b"\0" * 3200, 16_000)

    output.flush()

    assert not output.paused
    assert output.pending_count == 0
    assert output.playout_delay_s() == 0


def test_audio_output_terminal_release_does_not_depend_on_flush_action():
    output = ResponseAudioOutput()
    output.pause(7)
    output.hold(b"held", 16_000, 1)
    output.flush = lambda: (_ for _ in ()).throw(RuntimeError("broken flush"))

    output.release_all()

    assert not output.paused
    assert output.pause_owner is None
    assert output.pending_count == 0


def test_audio_output_pause_is_owned_by_the_interruption_candidate():
    output = ResponseAudioOutput()

    assert output.pause(11)
    assert output.paused
    assert output.pause_owner == 11
    assert not output.pause(11)
    assert not output.finish_resume(12)
    assert output.paused

    assert output.finish_resume(11)
    assert not output.paused
    assert output.pause_owner is None


@pytest.mark.asyncio
async def test_audio_output_backpressures_at_pending_byte_limit_until_resume():
    output = ResponseAudioOutput(max_pending_bytes=4)
    output.pause(11)
    assert await output.hold_or_wait_if_paused(b"1234", 16_000, 1)

    blocked = asyncio.create_task(output.hold_or_wait_if_paused(b"5", 16_000, 2))
    await asyncio.sleep(0)

    assert not blocked.done()
    assert output.pending_bytes == 4
    assert output.pop_pending_batch() == [PendingAudio(b"1234", 16_000, 1)]
    assert not blocked.done()

    output.finish_resume(11)

    assert await blocked is False
    assert output.pending_bytes == 0


@pytest.mark.asyncio
async def test_audio_output_terminal_release_unblocks_pending_producer():
    output = ResponseAudioOutput(max_pending_bytes=1)
    output.pause(12)
    assert await output.hold_or_wait_if_paused(b"a", 16_000, 1)
    blocked = asyncio.create_task(output.hold_or_wait_if_paused(b"b", 16_000, 2))
    await asyncio.sleep(0)

    output.release_all()

    assert await blocked is False
    assert not output.paused
    assert output.pending_bytes == 0
