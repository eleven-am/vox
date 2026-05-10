from __future__ import annotations

import asyncio
import fractions
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av.audio.resampler import AudioResampler


@dataclass
class RtcAudioDrain:
    future: asyncio.Future[None]


RtcAudioQueueItem = tuple[bytes, int] | RtcAudioDrain | None


class RtcAudioOutputTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, queue: asyncio.Queue[RtcAudioQueueItem]) -> None:
        super().__init__()
        self._queue = queue
        self._timestamp = 0
        self._start: float | None = None
        self._pending = np.empty(0, dtype=np.int16)
        self._sample_rate = 48_000

    async def enqueue(self, pcm16: bytes, sample_rate: int) -> None:
        await self._queue.put((pcm16, sample_rate))

    async def wait_until_drained(self) -> None:
        loop = asyncio.get_running_loop()
        marker = RtcAudioDrain(loop.create_future())
        await self._queue.put(marker)
        await marker.future

    def clear(self) -> None:
        self._pending = np.empty(0, dtype=np.int16)
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, RtcAudioDrain) and not item.future.done():
                item.future.set_result(None)

    async def recv(self) -> av.AudioFrame:
        while self._pending.size == 0:
            item = await self._queue.get()
            if item is None:
                raise MediaStreamError
            if isinstance(item, RtcAudioDrain):
                if not item.future.done():
                    item.future.set_result(None)
                continue
            pcm16, sample_rate = item
            self._sample_rate = int(sample_rate) or self._sample_rate
            self._pending = np.frombuffer(pcm16, dtype=np.int16)

        samples_per_frame = max(1, self._sample_rate // 50)
        samples = self._pending[:samples_per_frame]
        self._pending = self._pending[samples_per_frame:]
        if samples.size == 0:
            samples = np.zeros(1, dtype=np.int16)

        await self._pace(samples.size)

        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += samples.size
        return frame

    async def _pace(self, samples: int) -> None:
        if self._start is None:
            self._start = time.time()
            return
        target_time = self._start + (self._timestamp / self._sample_rate)
        wait = target_time - time.time()
        if wait > 0:
            await asyncio.sleep(wait)


async def pump_input_audio(
    track: MediaStreamTrack,
    ingest_pcm16: Callable[[bytes, int | None], Awaitable[None]],
) -> None:
    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            return
        pcm16, sample_rate = audio_frame_to_pcm16(frame)
        if pcm16:
            await ingest_pcm16(pcm16, sample_rate)


def audio_frame_to_pcm16(frame: av.AudioFrame) -> tuple[bytes, int]:
    sample_rate = int(frame.sample_rate or 48_000)
    resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: list[bytes] = []
    for mono in resampler.resample(frame):
        samples = mono.to_ndarray()
        chunks.append(np.ascontiguousarray(samples.reshape(-1), dtype=np.int16).tobytes())
    return b"".join(chunks), sample_rate
