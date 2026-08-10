import asyncio
from types import SimpleNamespace

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError

from vox.server.rtc_media import (
    RtcAudioOutputTrack,
    RtcAudioSenderTrack,
    audio_frame_to_pcm16,
    create_rtc_audio_queue,
    emit_media_event,
    pump_input_audio,
)


@pytest.mark.asyncio
async def test_rtc_audio_sender_deactivation_preserves_owner_cancellation():
    class Source:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def recv(self):
            self.started.set()
            await self.release.wait()

    source = Source()
    track = RtcAudioSenderTrack(source, active=True)
    recv_task = asyncio.create_task(track.recv())
    await source.started.wait()

    track.deactivate()
    recv_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(recv_task, timeout=0.1)


@pytest.mark.asyncio
async def test_rtc_audio_sender_handoff_preserves_inflight_frame():
    queue = asyncio.Queue()
    source = RtcAudioOutputTrack(queue)
    await source.enqueue(np.arange(960, dtype=np.int16).tobytes(), 16_000)
    pace_started = asyncio.Event()
    pace_calls = 0

    async def pace(_samples: int) -> None:
        nonlocal pace_calls
        pace_calls += 1
        if pace_calls == 2:
            pace_started.set()
            await asyncio.Event().wait()

    source._pace = pace
    current = RtcAudioSenderTrack(source, active=True)
    replacement = RtcAudioSenderTrack(source, active=False)

    first = await current.recv()
    current_recv = asyncio.create_task(current.recv())
    await pace_started.wait()
    current.deactivate()
    replacement.activate()
    second = await asyncio.wait_for(replacement.recv(), timeout=0.1)

    assert np.array_equal(first.to_ndarray().reshape(-1), np.arange(320, dtype=np.int16))
    assert np.array_equal(
        second.to_ndarray().reshape(-1),
        np.arange(320, 640, dtype=np.int16),
    )

    current_recv.cancel()
    with pytest.raises(asyncio.CancelledError):
        await current_recv


@pytest.mark.asyncio
async def test_rtc_audio_clear_invalidates_frame_waiting_for_pacing():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await track.enqueue(np.full(640, 1000, dtype=np.int16).tobytes(), 16_000)
    pace_started = asyncio.Event()
    release_pace = asyncio.Event()
    pace_calls = 0

    async def pace(_samples: int) -> None:
        nonlocal pace_calls
        pace_calls += 1
        if pace_calls == 1:
            pace_started.set()
            await release_pace.wait()

    track._pace = pace
    recv_task = asyncio.create_task(track.recv())
    await pace_started.wait()
    track.clear()
    release_pace.set()
    frame = await asyncio.wait_for(recv_task, timeout=0.1)

    assert np.all(frame.to_ndarray() == 0)


@pytest.mark.asyncio
async def test_rtc_audio_output_track_packetizes_large_tts_chunk():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await queue.put((bytes(3200), 16_000))

    first = await track.recv()
    second = await track.recv()
    third = await track.recv()

    assert first.samples == 320
    assert first.sample_rate == 16_000
    assert first.pts == 0
    assert second.samples == 320
    assert second.pts == 320
    assert third.samples == 320
    assert third.pts == 640


@pytest.mark.asyncio
async def test_rtc_audio_output_track_reports_only_played_audio_frames():
    queue = asyncio.Queue()
    played: list[tuple[bytes, int]] = []
    track = RtcAudioOutputTrack(
        queue,
        on_playout=lambda pcm16, sample_rate: played.append((pcm16, sample_rate)),
    )
    await track.enqueue(np.full(640, 1200, dtype=np.int16).tobytes(), 16_000)

    assert played == []
    await track.recv()

    assert len(played) == 1
    assert played[0][1] == 16_000
    assert np.frombuffer(played[0][0], dtype=np.int16).size == 320


@pytest.mark.asyncio
async def test_rtc_audio_output_track_stops_on_sentinel():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await queue.put(None)

    with pytest.raises(MediaStreamError):
        await track.recv()


@pytest.mark.asyncio
async def test_rtc_audio_output_track_resyncs_clock_on_sample_rate_switch():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await track.enqueue(np.full(480, 100, dtype=np.int16).tobytes(), 24_000)

    first = await track.recv()
    assert first.sample_rate == 24_000
    start_before_switch = track._start

    await track.enqueue(np.full(320, 200, dtype=np.int16).tobytes(), 16_000)
    second = await track.recv()

    assert second.sample_rate == 16_000
    assert track._start != start_before_switch


@pytest.mark.asyncio
async def test_rtc_audio_output_clear_drains_queue_and_resets_pacing():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await track.enqueue(np.full(1600, 1000, dtype=np.int16).tobytes(), 16_000)

    first = await track.recv()
    assert first.samples == 320
    assert np.any(first.to_ndarray() != 0)

    track._start = 0.0
    track.clear()
    silence = await track.recv()
    assert silence.samples == 320
    assert np.all(silence.to_ndarray() == 0)
    assert queue.empty()

    await track.enqueue(np.full(320, 2000, dtype=np.int16).tobytes(), 16_000)
    resumed = await track.recv()
    assert resumed.pts > silence.pts
    assert np.all(resumed.to_ndarray() == 2000)


@pytest.mark.asyncio
async def test_rtc_audio_suspend_outputs_silence_without_consuming_buffered_audio():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await track.enqueue(np.arange(640, dtype=np.int16).tobytes(), 16_000)

    assert track.suspend(7)
    assert track.stats()["last_suspend_to_silence_ms"] is None
    suspended = await track.recv()
    assert np.all(suspended.to_ndarray() == 0)
    assert track.stats()["last_suspend_to_silence_ms"] >= 0.0
    assert track.buffered_audio_ms == 40.0
    assert not track.resume(8)

    still_suspended = await track.recv()
    assert np.all(still_suspended.to_ndarray() == 0)
    assert track.buffered_audio_ms == 40.0

    assert track.resume(7)
    resumed = await track.recv()
    assert np.array_equal(resumed.to_ndarray().reshape(-1), np.arange(320, dtype=np.int16))
    assert track.buffered_audio_ms == 20.0


@pytest.mark.asyncio
async def test_rtc_audio_suspend_invalidates_an_inflight_playout_frame_without_dropping_it():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await track.enqueue(np.full(320, 1700, dtype=np.int16).tobytes(), 16_000)
    pace_started = asyncio.Event()
    release_pace = asyncio.Event()

    async def pace(_samples: int) -> None:
        pace_started.set()
        await release_pace.wait()

    track._pace = pace
    recv_task = asyncio.create_task(track.recv())
    await pace_started.wait()
    track.suspend(9)
    release_pace.set()
    suspended = await asyncio.wait_for(recv_task, timeout=0.1)

    assert np.all(suspended.to_ndarray() == 0)
    assert track.buffered_audio_ms == 20.0
    assert track.resume(9)
    resumed = await track.recv()
    assert np.all(resumed.to_ndarray() == 1700)


@pytest.mark.asyncio
async def test_rtc_audio_suspend_wins_when_recv_is_waiting_for_its_first_chunk():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    recv_task = asyncio.create_task(track.recv())
    await asyncio.sleep(0)

    track.suspend(21)
    await track.enqueue(np.full(320, 2300, dtype=np.int16).tobytes(), 16_000)
    suspended = await asyncio.wait_for(recv_task, timeout=0.1)

    assert np.all(suspended.to_ndarray() == 0)
    assert track.buffered_audio_ms == 20.0
    assert track.resume(21)
    resumed = await track.recv()
    assert np.all(resumed.to_ndarray() == 2300)


@pytest.mark.asyncio
async def test_rtc_audio_output_stop_releases_suspension_and_buffered_audio():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await track.enqueue(np.full(640, 2300, dtype=np.int16).tobytes(), 16_000)
    track.suspend(21)

    track.stop()

    stats = track.stats()
    assert not stats["suspended"]
    assert stats["suspend_owner"] is None
    assert stats["buffered_audio_ms"] == 0.0


@pytest.mark.asyncio
async def test_rtc_audio_enqueue_started_before_clear_cannot_play_after_clear():
    queue = create_rtc_audio_queue(max_buffered_audio_ms=100, chunk_ms=50)
    track = RtcAudioOutputTrack(queue, enqueue_chunk_ms=50)
    stale_audio = np.full(2_400, 1000, dtype=np.int16).tobytes()
    producer = asyncio.create_task(track.enqueue(stale_audio, 16_000))
    await asyncio.sleep(0)
    assert not producer.done()

    track.clear()
    cleared = await track.recv()
    await asyncio.wait_for(producer, timeout=0.1)
    after_clear = await track.recv()

    assert np.all(cleared.to_ndarray() == 0)
    assert np.all(after_clear.to_ndarray() == 0)


@pytest.mark.asyncio
async def test_rtc_audio_output_track_fills_tts_gaps_with_paced_silence():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await track.enqueue(np.full(320, 1000, dtype=np.int16).tobytes(), 16_000)

    first = await track.recv()
    assert np.all(first.to_ndarray() == 1000)

    silence = await asyncio.wait_for(track.recv(), timeout=0.1)
    assert silence.samples == 320
    assert silence.pts == 320
    assert np.all(silence.to_ndarray() == 0)

    await track.enqueue(np.full(320, 2000, dtype=np.int16).tobytes(), 16_000)
    resumed = await asyncio.wait_for(track.recv(), timeout=0.1)
    assert resumed.pts == 640
    assert np.all(resumed.to_ndarray() == 2000)


@pytest.mark.asyncio
async def test_rtc_audio_output_track_reports_buffer_and_clear_stats():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await track.enqueue(np.full(1600, 1000, dtype=np.int16).tobytes(), 16_000)

    queued = track.stats()
    assert queued["buffered_audio_ms"] == 100.0
    assert queued["max_buffered_audio_ms"] == 100.0
    assert queued["queued_items"] == 1
    assert queued["enqueued_chunks"] == 1

    frame = await track.recv()
    assert frame.samples == 320
    after_frame = track.stats()
    assert after_frame["buffered_audio_ms"] == 80.0
    assert after_frame["pending_samples"] == 1280

    track.clear()
    after_clear = track.stats()
    assert after_clear["buffered_audio_ms"] == 0.0
    assert after_clear["clear_count"] == 1

    silence = await track.recv()
    assert np.all(silence.to_ndarray() == 0)
    assert track.stats()["silence_frames"] == 1


@pytest.mark.asyncio
async def test_rtc_audio_output_queue_applies_backpressure_at_buffer_limit():
    queue = create_rtc_audio_queue(max_buffered_audio_ms=100, chunk_ms=50)
    track = RtcAudioOutputTrack(queue, enqueue_chunk_ms=50)
    producer = asyncio.create_task(track.enqueue(np.full(3_200, 1000, dtype=np.int16).tobytes(), 16_000))

    await asyncio.sleep(0)
    assert queue.full()
    assert not producer.done()
    assert track.stats()["max_buffered_audio_ms_limit"] == 100

    await track.recv()
    await asyncio.sleep(0)
    assert not producer.done()
    while not producer.done():
        await track.recv()
    await asyncio.wait_for(producer, timeout=0.1)

    assert track.stats()["buffered_audio_ms"] <= 100


@pytest.mark.asyncio
async def test_pump_input_audio_treats_media_stream_error_as_eof():
    class ClosedTrack:
        async def recv(self):
            raise MediaStreamError

    calls = []

    async def ingest(pcm16: bytes, sample_rate: int | None) -> None:
        calls.append((pcm16, sample_rate))

    await pump_input_audio(ClosedTrack(), ingest)

    assert calls == []


def test_audio_frame_to_pcm16_uses_pyav_resampler_api():
    samples = np.arange(480, dtype=np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    frame.sample_rate = 24_000

    pcm16, sample_rate = audio_frame_to_pcm16(frame)

    assert sample_rate == 24_000
    assert np.frombuffer(pcm16, dtype=np.int16).tolist() == samples.reshape(-1).tolist()


@pytest.mark.asyncio
async def test_pump_input_audio_ingests_real_audio_frame():
    samples = np.arange(320, dtype=np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    frame.sample_rate = 16_000

    class OneFrameTrack:
        def __init__(self) -> None:
            self._sent = False

        async def recv(self):
            if self._sent:
                raise MediaStreamError
            self._sent = True
            return frame

    calls = []

    async def ingest(pcm16: bytes, sample_rate: int | None) -> None:
        calls.append((pcm16, sample_rate))

    await pump_input_audio(OneFrameTrack(), ingest)

    assert calls == [(samples.reshape(-1).tobytes(), 16_000)]


@pytest.mark.asyncio
async def test_emit_media_event_is_noop_without_queue():
    record = SimpleNamespace(media_events=None)

    await emit_media_event(record, {"type": "rtc.connection_state"})


@pytest.mark.asyncio
async def test_emit_media_event_queues_when_available():
    record = SimpleNamespace(media_events=asyncio.Queue())

    event = {"type": "rtc.connection_state", "state": "connected"}
    await emit_media_event(record, event)

    assert await record.media_events.get() == event


@pytest.mark.asyncio
async def test_emit_media_event_binds_generation_at_enqueue_not_drain():
    record = SimpleNamespace(media_events=asyncio.Queue(), negotiation_generation=3)

    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": {"candidate": "pre"}})
    record.negotiation_generation = 4
    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": {"candidate": "post"}})

    pre = record.media_events.get_nowait()
    post = record.media_events.get_nowait()
    assert pre["generation"] == 3
    assert post["generation"] == 4


@pytest.mark.asyncio
async def test_emit_media_event_leaves_untracked_event_types_and_absent_generation_unstamped():
    record = SimpleNamespace(media_events=asyncio.Queue(), negotiation_generation=5)
    await emit_media_event(record, {"type": "rtc.connection_state", "state": "connected"})
    assert "generation" not in record.media_events.get_nowait()

    record.negotiation_generation = None
    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": {"candidate": "c"}})
    assert "generation" not in record.media_events.get_nowait()
