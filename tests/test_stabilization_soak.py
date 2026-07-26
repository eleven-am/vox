from __future__ import annotations

import asyncio
import gc
import json
import os
import subprocess
import sys
import threading
import tracemalloc
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aiortc import RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

import vox.server.rtc_signaling as rtc_signaling_module
from vox.conversation import TurnPolicy, TurnState
from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.scheduler import Scheduler
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelInfo,
    ModelType,
    SynthesizeChunk,
    TranscribeResult,
    VoiceInfo,
)
from vox.core.worker_host import WorkerHost
from vox.operations.conversation import ConversationSessionConfig
from vox.operations.conversation_commands import (
    ResponseCommitCommand,
    ResponseDeltaCommand,
    ResponseStartCommand,
    SessionUpdateCommand,
)
from vox.operations.rtc_runtime import RtcOfferCommand, RtcRuntime
from vox.server.rtc_registry import RtcSessionRegistry
from vox.streaming.types import SpeechStarted, StreamTranscript

_WORKER_SOURCE = """
import time
from vox.core.worker_host import worker_main

def handle(request):
    time.sleep(float(request.get("delay", 0)))
    return {"text": request.get("text", "ok")}

raise SystemExit(worker_main(handle))
"""


@dataclass(frozen=True)
class _ResourceSnapshot:
    rss_bytes: int
    python_bytes: int
    file_descriptors: int
    child_processes: int
    threads: int
    owned_threads: tuple[str, ...]
    tasks: int
    accelerator_bytes: int | None


def _rss_bytes() -> int:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip()) * 1024


def _file_descriptor_count() -> int:
    for path in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(path):
            return len(os.listdir(path))
    return 0


def _child_process_count() -> int:
    process = subprocess.Popen(
        ["ps", "-axo", "pid=,ppid="],
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout, _stderr = process.communicate()
    assert process.returncode == 0
    parent = os.getpid()
    return sum(
        1
        for line in stdout.splitlines()
        if line.split() and int(line.split()[0]) != process.pid and int(line.split()[1]) == parent
    )


def _accelerator_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "current_allocated_memory"):
        try:
            return int(mps.current_allocated_memory())
        except RuntimeError:
            return None
    return None


def _owned_thread_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith("vox-") or thread.name.endswith("-worker-stderr")
        )
    )


def _snapshot() -> _ResourceSnapshot:
    gc.collect()
    python_bytes, _peak = tracemalloc.get_traced_memory()
    return _ResourceSnapshot(
        rss_bytes=_rss_bytes(),
        python_bytes=python_bytes,
        file_descriptors=_file_descriptor_count(),
        child_processes=_child_process_count(),
        threads=threading.active_count(),
        owned_threads=_owned_thread_names(),
        tasks=len(asyncio.all_tasks()),
        accelerator_bytes=_accelerator_bytes(),
    )


class _SoakTTS(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="soak-tts",
            type=ModelType.TTS,
            architectures=("soak",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *_args: Any, **_kwargs: Any) -> None: ...

    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text: str, **_kwargs: Any):
        yield SynthesizeChunk(
            audio=np.zeros(480, dtype=np.float32).tobytes(),
            sample_rate=24_000,
            is_final="interrupted" not in text,
        )
        if "interrupted" in text:
            await asyncio.Future()


class _SingleAdapterScheduler:
    def __init__(self) -> None:
        self.adapter = _SoakTTS()

    @asynccontextmanager
    async def acquire(self, _model: str):
        yield self.adapter


class _Peer:
    connectionState = "connected"
    iceConnectionState = "connected"
    iceGatheringState = "complete"

    def __init__(self, **_kwargs: Any) -> None:
        self.closed = 0
        self.handlers: dict[str, Any] = {}
        self.tracks: list[Any] = []

    def addTrack(self, track: Any) -> None:
        self.tracks.append(track)

    def on(self, name: str):
        def register(handler):
            self.handlers[name] = handler
            return handler

        return register

    async def setRemoteDescription(self, _description: Any) -> None: ...

    async def createAnswer(self) -> RTCSessionDescription:
        return RTCSessionDescription(type="answer", sdp="answer-sdp")

    async def setLocalDescription(self, _description: Any) -> None: ...

    async def close(self) -> None:
        self.closed += 1


async def _wait_for_state(session: Any, state: TurnState) -> None:
    for _ in range(100):
        if session.state is state:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"session did not reach {state.value}; current={session.state.value}")


async def _wait_for_event(
    events: list[dict[str, Any]],
    event_type: str,
    changed: asyncio.Event,
    *,
    start: int = 0,
) -> dict[str, Any]:
    async with asyncio.timeout(2.0):
        while True:
            for event in events[start:]:
                if event.get("type") == event_type:
                    return event
            changed.clear()
            for event in events[start:]:
                if event.get("type") == event_type:
                    return event
            await changed.wait()


async def _consume_playout(track: Any, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await track.recv()
        except MediaStreamError:
            return


async def _stop_playout(task: asyncio.Task[None], stop: asyncio.Event) -> None:
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)


async def _rtc_cycle(registry: RtcSessionRegistry, peers: deque[_Peer], cycle: int) -> None:
    record = registry.create_session(control_transport="pondsocket")
    events: list[dict[str, Any]] = []
    events_changed = asyncio.Event()

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)
        events_changed.set()

    runtime = RtcRuntime(
        scheduler=_SingleAdapterScheduler(),
        registry=registry,
        session_id=record.session_id,
        transport="pondsocket",
        emit=collect,
    )
    await runtime.start()
    await runtime.dispatch(
        SessionUpdateCommand(
            config=ConversationSessionConfig(
                stt_model="soak-stt:1",
                tts_model="soak-tts:1",
                voice="default",
                language="en",
                policy=TurnPolicy(
                    min_interrupt_duration_ms=50,
                    false_interruption_timeout_ms=200,
                    max_endpointing_delay_ms=200,
                    speaking_interrupt_min_duration_ms=50,
                    speaking_interrupt_min_words=2,
                    aec_warmup_ms=0,
                ),
            )
        )
    )
    await runtime.dispatch(
        RtcOfferCommand(
            offer_type="offer",
            sdp="initial",
            generation=cycle * 2,
        )
    )
    old_peer = peers[-1]
    output_track = record.audio_output_track
    assert output_track is not None
    session = runtime.conversation.orchestrator._session
    assert session is not None

    complete_generation = f"complete-{cycle}"
    complete_start = len(events)
    complete_stop = asyncio.Event()
    complete_playout = asyncio.create_task(_consume_playout(output_track, complete_stop))
    await runtime.dispatch(ResponseStartCommand(generation_id=complete_generation))
    await runtime.dispatch(
        ResponseDeltaCommand(
            text="This response should drain completely.",
            generation_id=complete_generation,
        )
    )
    await runtime.dispatch(ResponseCommitCommand(generation_id=complete_generation))
    await _wait_for_event(events, "response.done", events_changed, start=complete_start)
    await _stop_playout(complete_playout, complete_stop)
    assert output_track.buffered_audio_ms == 0
    await _wait_for_state(session, TurnState.IDLE)

    interrupted_generation = f"interrupted-{cycle}"
    interrupted_start = len(events)
    interrupted_stop = asyncio.Event()
    interrupted_playout = asyncio.create_task(_consume_playout(output_track, interrupted_stop))
    await runtime.dispatch(ResponseStartCommand(generation_id=interrupted_generation))
    await runtime.dispatch(
        ResponseDeltaCommand(
            text="This response should be interrupted.",
            generation_id=interrupted_generation,
        )
    )
    await runtime.dispatch(ResponseCommitCommand(generation_id=interrupted_generation))
    await _wait_for_state(session, TurnState.SPEAKING)
    await _wait_for_event(events, "response.created", events_changed, start=interrupted_start)
    for _ in range(100):
        if output_track.stats()["enqueued_chunks"] >= 2:
            break
        await asyncio.sleep(0)
    assert output_track.stats()["enqueued_chunks"] >= 2
    clear_count = output_track.stats()["clear_count"]
    utterance_id = cycle + 1
    await session._forward_stream_event(SpeechStarted(timestamp_ms=1_000, utterance_id=utterance_id))
    await session._forward_stream_event(
        StreamTranscript(
            text="please stop now",
            is_partial=True,
            start_ms=1_000,
            end_ms=1_600,
            audio_duration_ms=600,
            utterance_id=utterance_id,
        )
    )
    await _wait_for_state(session, TurnState.INTERRUPTED)
    await _wait_for_event(events, "interruption.detected", events_changed, start=interrupted_start)
    await _wait_for_event(events, "response.audio.clear", events_changed, start=interrupted_start)
    await _wait_for_event(events, "response.cancelled", events_changed, start=interrupted_start)
    assert output_track.stats()["clear_count"] == clear_count + 1
    assert output_track.buffered_audio_ms == 0
    await _stop_playout(interrupted_playout, interrupted_stop)

    await runtime.dispatch(
        RtcOfferCommand(
            offer_type="offer",
            sdp="restart",
            restart=True,
            generation=cycle * 2 + 1,
        )
    )
    current_peer = peers[-1]
    assert current_peer is not old_peer
    queued_before = record.media_events.qsize()
    old_peer.connectionState = "failed"
    await old_peer.handlers["connectionstatechange"]()
    assert registry.get(record.session_id) is record
    assert record.media_events.qsize() == queued_before

    await runtime.close(reason="soak_complete")
    await registry.drain_teardowns()
    assert registry.get(record.session_id) is None
    assert runtime.conversation._background_tasks == set()
    assert record.media_tasks == set()
    assert record.audio_output.qsize() <= record.audio_output.maxsize
    assert old_peer.closed == 1
    assert current_peer.closed == 1
    assert session._owned_tts_tasks == set()
    assert session._tts_reaper_tasks == set()


@pytest.mark.asyncio
async def test_rtc_lifecycle_soak_100_cycles(monkeypatch) -> None:
    peers: deque[_Peer] = deque(maxlen=2)

    def create_peer(**kwargs: Any) -> _Peer:
        peer = _Peer(**kwargs)
        peers.append(peer)
        return peer

    monkeypatch.setattr(rtc_signaling_module, "TrickleRTCPeerConnection", create_peer)
    tracemalloc.start()
    registry = RtcSessionRegistry()
    for cycle in range(5):
        await _rtc_cycle(registry, peers, cycle)
    baseline = _snapshot()
    samples = [baseline]

    for cycle in range(100):
        await _rtc_cycle(registry, peers, cycle + 5)
        if (cycle + 1) % 10 == 0:
            samples.append(_snapshot())

    await registry.close_all()
    final = _snapshot()
    tracemalloc.stop()
    assert registry._sessions == {}
    assert registry._teardown_tasks == set()
    assert final.child_processes == baseline.child_processes
    assert final.owned_threads == baseline.owned_threads
    assert all(sample.owned_threads == baseline.owned_threads for sample in samples)
    assert final.tasks <= baseline.tasks
    assert final.file_descriptors <= baseline.file_descriptors + 2
    assert final.python_bytes - baseline.python_bytes < 2 * 1024 * 1024
    assert final.rss_bytes - baseline.rss_bytes < 16 * 1024 * 1024
    assert max(sample.rss_bytes for sample in samples) - baseline.rss_bytes < 24 * 1024 * 1024
    print(json.dumps({"rtc_cycles": 100, "baseline": asdict(baseline), "final": asdict(final)}, sort_keys=True))


class _WorkerAdapter(STTAdapter):
    live_pids: set[int] = set()

    def __init__(self) -> None:
        self._host: WorkerHost | None = None
        self._cache = bytearray()
        self.request_started = threading.Event()
        self.request_delay = 0.0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="soak-worker",
            type=ModelType.STT,
            architectures=("soak",),
            default_sample_rate=16_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, _model_path: str, _device: str, **_kwargs: Any) -> None:
        self._host = WorkerHost(
            [sys.executable, "-c", _WORKER_SOURCE],
            env=dict(os.environ),
            name="soak-worker",
            startup_timeout=10.0,
        )
        self.live_pids.add(self._host._proc.pid)
        self._cache = bytearray(256 * 1024)

    def unload(self) -> None:
        host = self._host
        self._host = None
        self._cache = bytearray()
        if host is not None:
            pid = host._proc.pid
            host.close(grace=1.0)
            self.live_pids.discard(pid)

    def trim(self) -> None:
        host = self._host
        assert host is not None
        assert host.request({"text": "trim"}, timeout=5.0) == {"text": "trim"}
        self._cache = bytearray()

    @property
    def is_loaded(self) -> bool:
        return self._host is not None and self._host.alive

    def transcribe(
        self,
        _audio: np.ndarray,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> TranscribeResult:
        self.request_started.set()
        host = self._host
        assert host is not None
        result = host.request(
            {
                "text": "ok",
                "delay": self.request_delay,
            },
            timeout=5.0,
        )
        if not self._cache:
            self._cache = bytearray(256 * 1024)
        return TranscribeResult(text=result["text"])

    def memory_status(self) -> dict[str, Any]:
        return {
            "cache_bytes": len(self._cache),
            "worker_pid": self._host._proc.pid if self._host is not None else None,
        }


class _WorkerRegistry:
    def resolve(self, _name: str, _tag: str) -> tuple[ModelInfo, Path]:
        return (
            ModelInfo(
                name="soak-stt",
                tag="1",
                type=ModelType.STT,
                format=ModelFormat.ONNX,
                architecture="soak",
                adapter="soak-worker",
            ),
            Path("/tmp/vox/soak-model"),
        )

    def resolve_model_ref(
        self,
        name: str,
        tag: str = "latest",
        *,
        explicit_tag: bool = False,
    ) -> tuple[str, str]:
        return name, tag

    def get_adapter_class(self, _adapter_name: str) -> type[_WorkerAdapter]:
        return _WorkerAdapter


async def _model_cycle(scheduler: Scheduler) -> None:
    await scheduler.preload("soak-stt:1")
    async with scheduler.acquire("soak-stt:1") as adapter:
        assert isinstance(adapter, _WorkerAdapter)
        assert (await adapter.execute_sync(lambda: adapter.transcribe(np.zeros(160, dtype=np.float32)))).text == "ok"
        adapter.request_started.clear()
        adapter.request_delay = 0.01
        pending = asyncio.create_task(adapter.execute_sync(lambda: adapter.transcribe(np.zeros(160, dtype=np.float32))))
        assert await asyncio.to_thread(adapter.request_started.wait, 1.0)
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await adapter.wait_execution_idle(timeout=1.0)
    assert await scheduler.trim("soak-stt:1") is True
    async with scheduler.acquire("soak-stt:1") as adapter:
        assert isinstance(adapter, _WorkerAdapter)
        adapter.request_delay = 0.0
        assert (await adapter.execute_sync(lambda: adapter.transcribe(np.zeros(160, dtype=np.float32)))).text == "ok"
    assert await scheduler.unload("soak-stt:1") is True
    assert scheduler.list_loaded() == []
    assert _WorkerAdapter.live_pids == set()


@pytest.mark.asyncio
async def test_model_lifecycle_soak_50_cycles() -> None:
    tracemalloc.start()
    scheduler = Scheduler(
        _WorkerRegistry(),
        default_device="cpu",
        max_loaded=1,
        ttl_seconds=0,
    )
    await _model_cycle(scheduler)
    baseline = _snapshot()
    samples = [baseline]

    for cycle in range(50):
        await _model_cycle(scheduler)
        if (cycle + 1) % 5 == 0:
            samples.append(_snapshot())

    await scheduler.stop()
    final = _snapshot()
    tracemalloc.stop()
    assert scheduler.list_loaded() == []
    assert scheduler._load_tasks == {}
    assert scheduler._maintenance_tasks == set()
    assert _WorkerAdapter.live_pids == set()
    assert final.child_processes == baseline.child_processes
    assert final.owned_threads == baseline.owned_threads
    assert all(sample.owned_threads == baseline.owned_threads for sample in samples)
    assert final.tasks <= baseline.tasks
    assert final.file_descriptors <= baseline.file_descriptors + 2
    assert final.python_bytes - baseline.python_bytes < 2 * 1024 * 1024
    assert final.rss_bytes - baseline.rss_bytes < 16 * 1024 * 1024
    assert max(sample.rss_bytes for sample in samples) - baseline.rss_bytes < 24 * 1024 * 1024
    print(json.dumps({"model_cycles": 50, "baseline": asdict(baseline), "final": asdict(final)}, sort_keys=True))
