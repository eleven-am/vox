from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
import threading
import time
import wave
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from vox.audio.pipeline import AudioChunk
from vox.core.worker_host import WorkerHost
from vox.speech_context.reducer import offset_context_spans
from vox.speech_context.runtime import RUNTIME_SPECS, RuntimeSpec, create_worker_host
from vox.speech_context.types import (
    SpeechContext,
    SpeechContextSpan,
    SpeechContextTrack,
    spans_from_payload,
)
from vox.streaming.codecs import float32_to_pcm16

logger = logging.getLogger(__name__)

DEFAULT_MAX_ADMITTED_ANALYSES = 2
DEFAULT_MAX_ADMITTED_AUDIO_BYTES = 256 * 1024 * 1024


class SpeechContextCapacityError(RuntimeError):
    pass


@dataclass(eq=False)
class _AnalysisRun:
    cancelled: threading.Event = field(default_factory=threading.Event)
    _hosts: dict[str, WorkerHost] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def bind_host(self, key: str, host: WorkerHost) -> bool:
        with self._lock:
            if self.cancelled.is_set():
                return False
            self._hosts[key] = host
            return True

    def cancel(self) -> tuple[tuple[str, WorkerHost], ...]:
        with self._lock:
            self.cancelled.set()
            return tuple(self._hosts.items())

    def ensure_active(self) -> None:
        if self.cancelled.is_set():
            raise RuntimeError("speech context analysis cancelled")


async def cancel_speech_context_task(
    task: asyncio.Task[SpeechContext] | None,
) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


class SpeechContextService:
    def __init__(
        self,
        *,
        home: Path | None = None,
        timeout: float = 300.0,
        host_factory: Callable[[RuntimeSpec], WorkerHost] | None = None,
        max_admitted_analyses: int = DEFAULT_MAX_ADMITTED_ANALYSES,
        max_admitted_audio_bytes: int = DEFAULT_MAX_ADMITTED_AUDIO_BYTES,
    ) -> None:
        self._home = home
        self._timeout = timeout
        self._host_factory = host_factory
        self._hosts: dict[str, WorkerHost] = {}
        self._host_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._analysis_slots = asyncio.Semaphore(1)
        self._analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-context-analysis")
        self._track_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="speech-context")
        self._max_admitted_analyses = max(1, int(max_admitted_analyses))
        self._max_admitted_audio_bytes = max(1, int(max_admitted_audio_bytes))
        self._admitted_analyses = 0
        self._admitted_audio_bytes = 0
        self._analysis_tasks: set[asyncio.Task[Any]] = set()
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._active_runs: set[_AnalysisRun] = set()
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def analyze_chunks(
        self,
        chunks: Iterable[AudioChunk],
        *,
        timeline_offset_ms: int = 0,
    ) -> SpeechContext:
        frozen = tuple(chunks)
        audio_bytes = sum(int(chunk.data.nbytes) for chunk in frozen)
        return await self._run_analysis(
            self._analyze_chunks_sync,
            frozen,
            timeline_offset_ms,
            audio_bytes=audio_bytes,
        )

    async def analyze_wave_path(
        self,
        audio_path: Path,
        *,
        timeline_offset_ms: int = 0,
    ) -> SpeechContext:
        try:
            audio_bytes = audio_path.stat().st_size
        except OSError:
            audio_bytes = 0
        return await self._run_analysis(
            self._analyze_wave_path_sync,
            audio_path,
            timeline_offset_ms,
            audio_bytes=audio_bytes,
        )

    async def close(self) -> None:
        async with self._close_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                tasks = tuple(self._analysis_tasks)
                runs = tuple(self._active_runs)
            for run in runs:
                run.cancel()
            for task in tasks:
                task.cancel()
            hosts = self._detach_all_hosts()
            await asyncio.to_thread(self._close_hosts, hosts)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            while self._cleanup_tasks:
                await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
            await asyncio.to_thread(self._shutdown_executors)

    async def _run_analysis(
        self,
        function: Callable[..., SpeechContext],
        *args: Any,
        audio_bytes: int,
    ) -> SpeechContext:
        self._admit(audio_bytes)
        current = asyncio.current_task()
        if current is None:
            self._release_admission(audio_bytes)
            raise RuntimeError("speech context analysis requires an asyncio task")
        self._analysis_tasks.add(current)
        acquired = False
        run: _AnalysisRun | None = None
        work: asyncio.Future[SpeechContext] | None = None
        try:
            await self._analysis_slots.acquire()
            acquired = True
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("speech context service is closed")
                run = _AnalysisRun()
                self._active_runs.add(run)
            loop = asyncio.get_running_loop()
            work = loop.run_in_executor(
                self._analysis_executor,
                partial(function, *args, run),
            )
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(self._cancel_and_drain(run, work))
                self._cleanup_tasks.add(cleanup)
                cleanup.add_done_callback(self._cleanup_tasks.discard)
                while not cleanup.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(cleanup)
                with contextlib.suppress(Exception):
                    cleanup.result()
                raise
        finally:
            if run is not None:
                with self._state_lock:
                    self._active_runs.discard(run)
            if acquired:
                self._analysis_slots.release()
            self._analysis_tasks.discard(current)
            self._release_admission(audio_bytes)

    async def _cancel_and_drain(
        self,
        run: _AnalysisRun,
        work: asyncio.Future[SpeechContext],
    ) -> None:
        hosts = run.cancel()
        await asyncio.to_thread(self._retire_hosts, hosts)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.shield(work)

    def _admit(self, audio_bytes: int) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("speech context service is closed")
            count_full = self._admitted_analyses >= self._max_admitted_analyses
            bytes_full = audio_bytes > self._max_admitted_audio_bytes - self._admitted_audio_bytes
            if count_full or bytes_full:
                raise SpeechContextCapacityError("speech context analysis capacity exceeded")
            self._admitted_analyses += 1
            self._admitted_audio_bytes += audio_bytes

    def _release_admission(self, audio_bytes: int) -> None:
        with self._state_lock:
            self._admitted_analyses -= 1
            self._admitted_audio_bytes -= audio_bytes

    def _analyze_chunks_sync(
        self,
        chunks: tuple[AudioChunk, ...],
        timeline_offset_ms: int,
        run: _AnalysisRun,
    ) -> SpeechContext:
        run.ensure_active()
        if not chunks:
            return SpeechContext(
                status="failed",
                unavailable=("speaker", "sounds"),
            )
        with tempfile.TemporaryDirectory(prefix="vox-speech-context-") as directory:
            audio_path = Path(directory) / "input.wav"
            self._write_wave(audio_path, chunks)
            run.ensure_active()
            return self._analyze_wave_path_sync(audio_path, timeline_offset_ms, run)

    def _analyze_wave_path_sync(
        self,
        audio_path: Path,
        timeline_offset_ms: int,
        run: _AnalysisRun,
    ) -> SpeechContext:
        run.ensure_active()
        futures = {
            key: self._track_executor.submit(self._request_track, spec, audio_path, run)
            for key, spec in RUNTIME_SPECS.items()
        }
        results: dict[str, dict[str, Any] | None] = {}
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except Exception as error:
                logger.warning("speech context %s unavailable: %s", key, error)
                results[key] = None
        run.ensure_active()
        return self._assemble(results, timeline_offset_ms=timeline_offset_ms)

    @staticmethod
    def _write_wave(audio_path: Path, chunks: tuple[AudioChunk, ...]) -> None:
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            origin_ms = chunks[0].offset_ms
            cursor_samples = 0
            for chunk in chunks:
                if chunk.sample_rate != 16_000:
                    raise ValueError("speech context requires mono 16 kHz audio chunks")
                start_samples = round((chunk.offset_ms - origin_ms) * 16_000 / 1000)
                if start_samples > cursor_samples:
                    SpeechContextService._write_silence(handle, start_samples - cursor_samples)
                    cursor_samples = start_samples
                trim_samples = max(0, cursor_samples - start_samples)
                audio = chunk.data[trim_samples:]
                handle.writeframes(float32_to_pcm16(audio))
                cursor_samples += len(audio)

    @staticmethod
    def _write_silence(handle: wave.Wave_write, samples: int) -> None:
        block_samples = 16_000 * 30
        remaining = samples
        block = b"\0\0" * min(block_samples, remaining)
        while remaining:
            count = min(block_samples, remaining)
            handle.writeframes(block[: count * 2])
            remaining -= count

    def _request_track(
        self,
        spec: RuntimeSpec,
        audio_path: Path,
        run: _AnalysisRun,
    ) -> dict[str, Any]:
        run.ensure_active()
        host = self._host_for(spec)
        if not run.bind_host(spec.key, host):
            self._retire_hosts(((spec.key, host),))
            run.ensure_active()
        started_at = time.perf_counter()
        response = host.request(
            {"op": "analyze_compact", "audio_path": str(audio_path)},
            timeout=self._timeout,
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"{spec.key} worker returned an invalid compact result")
        result = dict(result)
        result.pop("_pre_reduction", None)
        logger.info(
            "speech context %s complete analysis_ms=%d",
            "SenseVoice" if spec.key == "speaker" else "YAMNet",
            round((time.perf_counter() - started_at) * 1000),
        )
        return result

    def _host_for(self, spec: RuntimeSpec) -> WorkerHost:
        with self._host_lock:
            if self._closed:
                raise RuntimeError("speech context service is closed")
            current = self._hosts.get(spec.key)
            if current is not None and current.alive:
                return current
            if current is not None:
                with contextlib.suppress(Exception):
                    current.close()
            host = (
                self._host_factory(spec)
                if self._host_factory is not None
                else create_worker_host(
                    spec,
                    home=self._home,
                    startup_timeout=self._timeout,
                )
            )
            self._hosts[spec.key] = host
            return host

    def _detach_all_hosts(self) -> tuple[WorkerHost, ...]:
        with self._host_lock:
            hosts = tuple(self._hosts.values())
            self._hosts = {}
            return hosts

    def _retire_hosts(self, hosts: tuple[tuple[str, WorkerHost], ...]) -> None:
        for key, host in hosts:
            should_close = False
            with self._host_lock:
                if self._hosts.get(key) is host:
                    self._hosts.pop(key, None)
                    should_close = True
            if should_close:
                with contextlib.suppress(Exception):
                    host.close()

    @staticmethod
    def _close_hosts(hosts: tuple[WorkerHost, ...]) -> None:
        for host in hosts:
            with contextlib.suppress(Exception):
                host.close()

    def _shutdown_executors(self) -> None:
        self._analysis_executor.shutdown(wait=True, cancel_futures=True)
        self._track_executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _assemble(
        results: dict[str, dict[str, Any] | None],
        *,
        timeline_offset_ms: int,
    ) -> SpeechContext:
        unavailable: list[SpeechContextTrack] = []
        emotions: tuple[SpeechContextSpan, ...] | None = None
        vocal: tuple[SpeechContextSpan, ...] | None = None
        sounds: tuple[SpeechContextSpan, ...] | None = None
        try:
            raw_speaker = results.get("speaker")
            if raw_speaker is not None:
                shifted = offset_context_spans(
                    raw_speaker,
                    offset_ms=timeline_offset_ms,
                )
                parsed_emotions = spans_from_payload(shifted, "emotions")
                parsed_vocal = spans_from_payload(shifted, "vocal")
                emotions = parsed_emotions
                vocal = parsed_vocal
            else:
                unavailable.append("speaker")
        except (TypeError, ValueError):
            unavailable.append("speaker")
        try:
            raw_sounds = results.get("sounds")
            if raw_sounds is not None:
                shifted = offset_context_spans(
                    raw_sounds,
                    offset_ms=timeline_offset_ms,
                )
                sounds = spans_from_payload(shifted, "sounds")
            else:
                unavailable.append("sounds")
        except (TypeError, ValueError):
            unavailable.append("sounds")
        if not unavailable:
            status = "complete"
        elif len(unavailable) == 2:
            status = "failed"
        else:
            status = "partial"
        return SpeechContext(
            status=status,
            emotions=emotions,
            vocal=vocal,
            sounds=sounds,
            unavailable=tuple(unavailable),
        )
