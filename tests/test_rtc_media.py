import asyncio

import pytest
from aiortc.mediastreams import MediaStreamError

from vox.server.rtc_media import RtcAudioOutputTrack


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
