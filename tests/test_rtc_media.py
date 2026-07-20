import asyncio
from types import SimpleNamespace

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError

from vox.server.rtc_media import (
    RtcAudioOutputTrack,
    audio_frame_to_pcm16,
    create_rtc_audio_queue,
    emit_media_event,
    pump_input_audio,
)


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
