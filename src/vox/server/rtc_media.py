from __future__ import annotations

import asyncio
import fractions
import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av.audio.resampler import AudioResampler


@dataclass
class RtcAudioDrain:
    future: asyncio.Future[None]


@dataclass(frozen=True)
class RtcAudioClear:
    pass


RtcAudioQueueItem = tuple[bytes, int] | RtcAudioDrain | RtcAudioClear | None


class RtcAudioOutputTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, queue: asyncio.Queue[RtcAudioQueueItem]) -> None:
        super().__init__()
        self._queue = queue
        self._timestamp = 0
        self._start: float | None = None
        self._pending = np.empty(0, dtype=np.int16)
        self._sample_rate = 48_000
        self._silenced = False
        self._queued_audio_ms = 0.0
        self._max_buffered_audio_ms = 0.0
        self._enqueued_chunks = 0
        self._silence_frames = 0
        self._clear_count = 0

    async def enqueue(self, pcm16: bytes, sample_rate: int) -> None:
        self._silenced = False
        self._enqueued_chunks += 1
        self._queued_audio_ms += _pcm16_duration_ms(pcm16, sample_rate)
        self._update_max_buffered_audio_ms()
        await self._queue.put((pcm16, sample_rate))

    async def wait_until_drained(self) -> None:
        loop = asyncio.get_running_loop()
        marker = RtcAudioDrain(loop.create_future())
        await self._queue.put(marker)
        await marker.future

    def clear(self) -> None:
        self._pending = np.empty(0, dtype=np.int16)
        self._queued_audio_ms = 0.0
        self._clear_count += 1
        self._silenced = True
        self._sync_clock()
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, RtcAudioDrain) and not item.future.done():
                item.future.set_result(None)
        self._queue.put_nowait(RtcAudioClear())

    async def recv(self) -> av.AudioFrame:
        while self._pending.size == 0:
            if self._start is None and not self._silenced:
                item = await self._queue.get()
            else:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    return await self._silence_frame()
            if item is None:
                raise MediaStreamError
            if isinstance(item, RtcAudioDrain):
                if not item.future.done():
                    item.future.set_result(None)
                continue
            if isinstance(item, RtcAudioClear):
                self._pending = np.empty(0, dtype=np.int16)
                self._queued_audio_ms = 0.0
                self._silenced = True
                self._sync_clock()
                return await self._silence_frame()
            pcm16, sample_rate = item
            self._silenced = False
            self._queued_audio_ms = max(0.0, self._queued_audio_ms - _pcm16_duration_ms(pcm16, sample_rate))
            new_rate = int(sample_rate) or self._sample_rate
            if new_rate != self._sample_rate:
                self._sample_rate = new_rate
                self._sync_clock()
            self._pending = np.frombuffer(pcm16, dtype=np.int16)
            self._update_max_buffered_audio_ms()

        samples_per_frame = max(1, self._sample_rate // 50)
        samples = self._pending[:samples_per_frame]
        self._pending = self._pending[samples_per_frame:]
        if samples.size == 0:
            samples = np.zeros(1, dtype=np.int16)

        return await self._frame(samples)

    async def _silence_frame(self) -> av.AudioFrame:
        self._silence_frames += 1
        samples_per_frame = max(1, self._sample_rate // 50)
        return await self._frame(np.zeros(samples_per_frame, dtype=np.int16))

    async def _frame(self, samples: np.ndarray) -> av.AudioFrame:
        await self._pace(samples.size)

        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += samples.size
        return frame

    def _sync_clock(self) -> None:
        self._start = time.time() - (self._timestamp / self._sample_rate)

    async def _pace(self, samples: int) -> None:
        if self._start is None:
            self._start = time.time()
            return
        target_time = self._start + (self._timestamp / self._sample_rate)
        wait = target_time - time.time()
        if wait < -0.1:
            self._sync_clock()
            return
        if wait > 0:
            await asyncio.sleep(wait)

    @property
    def buffered_audio_ms(self) -> float:
        pending_ms = float(self._pending.size) / max(1, self._sample_rate) * 1000.0
        return max(0.0, self._queued_audio_ms) + pending_ms

    def stats(self) -> dict[str, Any]:
        buffered_ms = self.buffered_audio_ms
        self._max_buffered_audio_ms = max(self._max_buffered_audio_ms, buffered_ms)
        return {
            "buffered_audio_ms": round(buffered_ms, 2),
            "max_buffered_audio_ms": round(self._max_buffered_audio_ms, 2),
            "queued_items": self._queue.qsize(),
            "pending_samples": int(self._pending.size),
            "sample_rate": int(self._sample_rate),
            "enqueued_chunks": self._enqueued_chunks,
            "silence_frames": self._silence_frames,
            "clear_count": self._clear_count,
        }

    def _update_max_buffered_audio_ms(self) -> None:
        self._max_buffered_audio_ms = max(self._max_buffered_audio_ms, self.buffered_audio_ms)


def _pcm16_duration_ms(pcm16: bytes, sample_rate: int) -> float:
    rate = int(sample_rate) or 48_000
    return (len(pcm16) // np.dtype(np.int16).itemsize) / max(1, rate) * 1000.0


def cancel_media_tasks(record: Any) -> list[asyncio.Task]:
    tasks = list(getattr(record, "media_tasks", ()))
    for task in tasks:
        task.cancel()
    media_tasks = getattr(record, "media_tasks", None)
    if media_tasks is not None:
        media_tasks.clear()
    return tasks


async def cancel_and_drain_media_tasks(record: Any) -> None:
    tasks = cancel_media_tasks(record)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


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
    resampler = _resampler_for(sample_rate)
    chunks: list[bytes] = []
    for mono in resampler.resample(frame):
        samples = mono.to_ndarray()
        chunks.append(np.ascontiguousarray(samples.reshape(-1), dtype=np.int16).tobytes())
    return b"".join(chunks), sample_rate


@functools.lru_cache(maxsize=8)
def _resampler_for(sample_rate: int) -> AudioResampler:
    return AudioResampler(format="s16", layout="mono", rate=sample_rate)
