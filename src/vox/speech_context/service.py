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
    ) -> None:
        self._home = home
        self._timeout = timeout
        self._host_factory = host_factory
        self._hosts: dict[str, WorkerHost] = {}
        self._host_lock = threading.Lock()
        self._track_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="speech-context")
        self._closed = False

    async def analyze_chunks(
        self,
        chunks: Iterable[AudioChunk],
        *,
        timeline_offset_ms: int = 0,
    ) -> SpeechContext:
        frozen = tuple(chunks)
        return await asyncio.to_thread(
            self._analyze_chunks_sync,
            frozen,
            timeline_offset_ms,
        )

    async def analyze_wave_path(
        self,
        audio_path: Path,
        *,
        timeline_offset_ms: int = 0,
    ) -> SpeechContext:
        return await asyncio.to_thread(
            self._analyze_wave_path_sync,
            audio_path,
            timeline_offset_ms,
        )

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _analyze_chunks_sync(
        self,
        chunks: tuple[AudioChunk, ...],
        timeline_offset_ms: int,
    ) -> SpeechContext:
        if not chunks:
            return SpeechContext(
                status="failed",
                unavailable=("speaker", "sounds"),
            )
        with tempfile.TemporaryDirectory(prefix="vox-speech-context-") as directory:
            audio_path = Path(directory) / "input.wav"
            self._write_wave(audio_path, chunks)
            return self._analyze_wave_path_sync(audio_path, timeline_offset_ms)

    def _analyze_wave_path_sync(
        self,
        audio_path: Path,
        timeline_offset_ms: int,
    ) -> SpeechContext:
        futures = {
            key: self._track_executor.submit(self._request_track, spec, audio_path)
            for key, spec in RUNTIME_SPECS.items()
        }
        results: dict[str, dict[str, Any] | None] = {}
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except Exception as error:
                logger.warning("speech context %s unavailable: %s", key, error)
                results[key] = None
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

    def _request_track(self, spec: RuntimeSpec, audio_path: Path) -> dict[str, Any]:
        host = self._host_for(spec)
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

    def _close_sync(self) -> None:
        with self._host_lock:
            if self._closed:
                return
            self._closed = True
            hosts = tuple(self._hosts.values())
            self._hosts = {}
        self._track_executor.shutdown(wait=True, cancel_futures=False)
        for host in hosts:
            with contextlib.suppress(Exception):
                host.close()
