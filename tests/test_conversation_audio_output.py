from __future__ import annotations

from vox.conversation.audio_output import PendingAudio, ResponseAudioOutput


def test_audio_output_sequences_reset_per_response():
    output = ResponseAudioOutput()

    assert output.next_sequence() == 1
    assert output.next_sequence() == 2

    output.reset_for_response()

    assert output.next_sequence() == 1


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
