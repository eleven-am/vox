from __future__ import annotations

import asyncio
import fractions
import functools
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av.audio.resampler import AudioResampler

logger = logging.getLogger(__name__)


@dataclass
class RtcAudioDrain:
    future: asyncio.Future[None]
    epoch: int


@dataclass(frozen=True)
class RtcAudioClear:
    epoch: int


@dataclass(frozen=True)
class RtcAudioChunk:
    pcm16: bytes
    sample_rate: int
    epoch: int


RtcAudioQueueItem = tuple[bytes, int] | RtcAudioChunk | RtcAudioDrain | RtcAudioClear | None

DEFAULT_RTC_OUTPUT_BUFFER_MS = 2_000
DEFAULT_RTC_OUTPUT_CHUNK_MS = 100


def create_rtc_audio_queue(
    *,
    max_buffered_audio_ms: int | None = None,
    chunk_ms: int = DEFAULT_RTC_OUTPUT_CHUNK_MS,
) -> asyncio.Queue[RtcAudioQueueItem]:
    """Create a bounded playout queue sized by an audio-duration budget."""
    if max_buffered_audio_ms is None:
        raw_limit = os.environ.get("VOX_RTC_MAX_OUTPUT_BUFFER_MS", "")
        try:
            max_buffered_audio_ms = int(raw_limit) if raw_limit else DEFAULT_RTC_OUTPUT_BUFFER_MS
        except ValueError:
            max_buffered_audio_ms = DEFAULT_RTC_OUTPUT_BUFFER_MS
    chunk_ms = max(1, int(chunk_ms))
    # One chunk may already be held in ``RtcAudioOutputTrack._pending`` while
    # the queue fills, so reserve that chunk inside the total duration budget.
    max_items = max(1, math.ceil(max(1, int(max_buffered_audio_ms)) / chunk_ms) - 1)
    return asyncio.Queue(maxsize=max_items)


class RtcAudioOutputTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self,
        queue: asyncio.Queue[RtcAudioQueueItem],
        *,
        on_playout: Callable[[bytes, int], None] | None = None,
        enqueue_chunk_ms: int = DEFAULT_RTC_OUTPUT_CHUNK_MS,
    ) -> None:
        super().__init__()
        self._queue = queue
        self._on_playout = on_playout
        self._enqueue_chunk_ms = max(1, int(enqueue_chunk_ms))
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
        self._playout_callback_errors = 0
        self._epoch = 0
        self._playout_revision = 0
        self._suspend_owner: int | None = None
        self._suspend_requested_at: float | None = None
        self._last_suspend_to_silence_ms: float | None = None
        self._recv_lock = asyncio.Lock()

    def set_playout_observer(self, on_playout: Callable[[bytes, int], None] | None) -> None:
        self._on_playout = on_playout

    async def enqueue(self, pcm16: bytes, sample_rate: int) -> None:
        epoch = self._epoch
        self._silenced = False
        self._enqueued_chunks += 1
        rate = max(1, int(sample_rate))
        bytes_per_chunk = max(2, rate * 2 * self._enqueue_chunk_ms // 1000)
        bytes_per_chunk -= bytes_per_chunk % 2
        for offset in range(0, len(pcm16), bytes_per_chunk):
            chunk = pcm16[offset : offset + bytes_per_chunk]
            if not chunk:
                continue
            if epoch != self._epoch:
                return
            await self._queue.put(RtcAudioChunk(pcm16=chunk, sample_rate=rate, epoch=epoch))
            if epoch != self._epoch:
                return
            self._queued_audio_ms += _pcm16_duration_ms(chunk, rate)
            self._update_max_buffered_audio_ms()

    async def wait_until_drained(self) -> None:
        loop = asyncio.get_running_loop()
        epoch = self._epoch
        marker = RtcAudioDrain(loop.create_future(), epoch)
        await self._queue.put(marker)
        if epoch != self._epoch:
            if not marker.future.done():
                marker.future.set_result(None)
            return
        await marker.future

    def clear(self) -> None:
        self._epoch += 1
        self._playout_revision += 1
        self._suspend_owner = None
        self._suspend_requested_at = None
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
        self._queue.put_nowait(RtcAudioClear(self._epoch))

    def suspend(self, owner: int) -> bool:
        if self._suspend_owner == owner:
            return False
        self._suspend_owner = owner
        self._suspend_requested_at = time.monotonic()
        self._playout_revision += 1
        self._sync_clock()
        return True

    def resume(self, owner: int) -> bool:
        if self._suspend_owner != owner:
            return False
        self._suspend_owner = None
        self._suspend_requested_at = None
        self._playout_revision += 1
        self._sync_clock()
        return True

    async def recv(self) -> av.AudioFrame:
        async with self._recv_lock:
            while True:
                if self._suspend_owner is not None:
                    return await self._suspended_silence_frame()
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
                        if item.epoch != self._epoch:
                            continue
                        self._pending = np.empty(0, dtype=np.int16)
                        self._queued_audio_ms = 0.0
                        self._silenced = True
                        self._sync_clock()
                        return await self._silence_frame()
                    if isinstance(item, RtcAudioChunk):
                        if item.epoch != self._epoch:
                            continue
                        pcm16 = item.pcm16
                        sample_rate = item.sample_rate
                    else:
                        pcm16, sample_rate = item
                    self._silenced = False
                    self._queued_audio_ms = max(
                        0.0,
                        self._queued_audio_ms - _pcm16_duration_ms(pcm16, sample_rate),
                    )
                    new_rate = int(sample_rate) or self._sample_rate
                    if new_rate != self._sample_rate:
                        self._sample_rate = new_rate
                        self._sync_clock()
                    self._pending = np.frombuffer(pcm16, dtype=np.int16)
                    self._update_max_buffered_audio_ms()

                if self._suspend_owner is not None:
                    return await self._suspended_silence_frame()
                samples_per_frame = max(1, self._sample_rate // 50)
                samples = self._pending[:samples_per_frame]
                if samples.size == 0:
                    samples = np.zeros(1, dtype=np.int16)
                epoch = self._epoch
                revision = self._playout_revision
                await self._pace(samples.size)
                if epoch != self._epoch or revision != self._playout_revision:
                    continue
                self._pending = self._pending[samples_per_frame:]
                frame = self._build_frame(samples)
                if self._on_playout is not None:
                    try:
                        self._on_playout(np.ascontiguousarray(samples).tobytes(), self._sample_rate)
                    except Exception:
                        self._playout_callback_errors += 1
                return frame

    async def _silence_frame(self) -> av.AudioFrame:
        self._silence_frames += 1
        samples_per_frame = max(1, self._sample_rate // 50)
        return await self._frame(np.zeros(samples_per_frame, dtype=np.int16))

    async def _suspended_silence_frame(self) -> av.AudioFrame:
        requested_at = self._suspend_requested_at
        owner = self._suspend_owner
        frame = await self._silence_frame()
        if requested_at is not None:
            latency_ms = max(0.0, (time.monotonic() - requested_at) * 1000.0)
            self._last_suspend_to_silence_ms = latency_ms
            if self._suspend_requested_at == requested_at:
                self._suspend_requested_at = None
            logger.info(
                "rtc output suspension reached silence owner=%s suspend_to_silence_ms=%.2f",
                owner,
                latency_ms,
            )
        return frame

    async def _frame(self, samples: np.ndarray) -> av.AudioFrame:
        await self._pace(samples.size)
        return self._build_frame(samples)

    def _build_frame(self, samples: np.ndarray) -> av.AudioFrame:
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
            "max_buffered_audio_ms_limit": (
                (self._queue.maxsize + 1) * self._enqueue_chunk_ms if self._queue.maxsize > 0 else None
            ),
            "queued_items": self._queue.qsize(),
            "pending_samples": int(self._pending.size),
            "sample_rate": int(self._sample_rate),
            "enqueued_chunks": self._enqueued_chunks,
            "silence_frames": self._silence_frames,
            "clear_count": self._clear_count,
            "suspended": self._suspend_owner is not None,
            "suspend_owner": self._suspend_owner,
            "last_suspend_to_silence_ms": (
                round(self._last_suspend_to_silence_ms, 2)
                if self._last_suspend_to_silence_ms is not None
                else None
            ),
            "playout_callback_errors": self._playout_callback_errors,
        }

    def stop(self) -> None:
        self.clear()
        super().stop()

    def _update_max_buffered_audio_ms(self) -> None:
        self._max_buffered_audio_ms = max(self._max_buffered_audio_ms, self.buffered_audio_ms)


class RtcAudioSenderTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, source: RtcAudioOutputTrack, *, active: bool = False) -> None:
        super().__init__()
        self._source = source
        self._active = asyncio.Event()
        self._recv_task: asyncio.Task[av.AudioFrame] | None = None
        if active:
            self._active.set()

    def activate(self) -> None:
        self._active.set()

    def deactivate(self) -> None:
        self._active.clear()
        task = self._recv_task
        if task is not None:
            task.cancel()

    async def recv(self) -> av.AudioFrame:
        while True:
            await self._active.wait()
            task = asyncio.create_task(self._source.recv())
            self._recv_task = task
            try:
                frame = await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if self._active.is_set() or (current is not None and current.cancelling()):
                    raise
                continue
            finally:
                if self._recv_task is task:
                    self._recv_task = None
            if self._active.is_set():
                return frame

    def stop(self) -> None:
        self.deactivate()
        super().stop()


def _pcm16_duration_ms(pcm16: bytes, sample_rate: int) -> float:
    rate = int(sample_rate) or 48_000
    return (len(pcm16) // np.dtype(np.int16).itemsize) / max(1, rate) * 1000.0


GENERATION_STAMPED_EVENT_TYPES = frozenset({"rtc.answer", "rtc.ice_candidate", "rtc.signaling_error"})
_GENERATION_UNSET = object()


def stamp_negotiation_generation(
    record: Any,
    event: dict,
    *,
    generation: int | None | object = _GENERATION_UNSET,
) -> dict:
    if generation is _GENERATION_UNSET:
        generation = getattr(record, "negotiation_generation", None)
    if generation is not None and event.get("type") in GENERATION_STAMPED_EVENT_TYPES:
        event["generation"] = generation
    return event


async def emit_media_event(
    record: Any,
    event: dict,
    *,
    generation: int | None | object = _GENERATION_UNSET,
) -> None:
    if getattr(record, "media_events", None) is not None:
        stamp_negotiation_generation(record, event, generation=generation)
        await record.media_events.put(event)


async def pump_input_audio(
    track: MediaStreamTrack,
    ingest_pcm16: Callable[[bytes, int | None], Awaitable[None]],
) -> None:
    while True:
        try:
            frame = cast(av.AudioFrame, await track.recv())
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
