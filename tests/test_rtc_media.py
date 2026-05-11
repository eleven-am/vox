import asyncio

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError

from vox.server.rtc_media import RtcAudioOutputTrack, audio_frame_to_pcm16, pump_input_audio


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
async def test_rtc_audio_output_track_stops_on_sentinel():
    queue = asyncio.Queue()
    track = RtcAudioOutputTrack(queue)
    await queue.put(None)

    with pytest.raises(MediaStreamError):
        await track.recv()


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
