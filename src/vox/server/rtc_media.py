from __future__ import annotations

import asyncio
import fractions
from collections.abc import Awaitable, Callable

import av
import numpy as np
from aiortc import MediaStreamTrack


class RtcAudioOutputTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, queue: asyncio.Queue[tuple[bytes, int] | None]) -> None:
        super().__init__()
        self._queue = queue
        self._timestamp = 0

    async def recv(self) -> av.AudioFrame:
        item = await self._queue.get()
        if item is None:
            raise asyncio.CancelledError
        pcm16, sample_rate = item
        samples = np.frombuffer(pcm16, dtype=np.int16)
        if samples.size == 0:
            samples = np.zeros(1, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, sample_rate)
        self._timestamp += samples.size
        return frame


async def pump_input_audio(
    track: MediaStreamTrack,
    ingest_pcm16: Callable[[bytes, int | None], Awaitable[None]],
) -> None:
    while True:
        frame = await track.recv()
        pcm16, sample_rate = audio_frame_to_pcm16(frame)
        if pcm16:
            await ingest_pcm16(pcm16, sample_rate)


def audio_frame_to_pcm16(frame: av.AudioFrame) -> tuple[bytes, int]:
    mono = frame.reformat(format="s16", layout="mono")
    chunks = [bytes(plane) for plane in mono.planes]
    return b"".join(chunks), int(mono.sample_rate)
