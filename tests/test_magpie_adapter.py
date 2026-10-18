from __future__ import annotations

import asyncio
import base64
import ctypes
import io
import json
import socket
import tarfile
import threading
import time
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType
from vox.operations.errors import InvalidConfigError


class FakeHost:
    instance: FakeHost | None = None

    def __init__(self, argv, *, env, name, startup_timeout):
        self.argv = argv
        self.env = env
        self.name = name
        self.startup_timeout = startup_timeout
        self.alive = True
        self.pid = 123456
        self.ready_frame = {
            "ready": True,
            "sample_rate": 22050,
            "voices": ["Aria", "Jason", "John", "Leo", "Sofia"],
            "version": "nemo-speech-tts 0.2.0",
            "thread_budget": 3,
            "generator_threads": 2,
            "codec_threads": 1,
            "memory": {"rss_bytes": 100},
        }
        self.frames: list[dict] = []
        self.payload = None
        self.controls: list[dict] = []
        self.cancelled = threading.Event()
        FakeHost.instance = self

    def stream_request(self, payload, timeout):
        self.payload = payload
        yield from self.frames

    def send_control(self, payload):
        self.controls.append(payload)
        self.cancelled.set()

    def close(self):
        self.alive = False


def _write_bundle(root: Path) -> None:
    (root / "magpie_tts_multilingual_357m.v2607.f16.gguf").write_bytes(b"magpie")
    codec = root / "codec"
    codec.mkdir()
    (codec / "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf").write_bytes(b"codec")


def _load_adapter(tmp_path: Path):
    from vox_magpie.adapter import MagpieTTSAdapter

    _write_bundle(tmp_path)
    with (
        patch("vox_magpie.adapter.ensure_runtime", return_value=tmp_path / "runtime"),
        patch("vox_magpie.adapter.WorkerHost", FakeHost),
        patch("vox_magpie.adapter._worker_env", return_value={}),
        patch("vox_magpie.adapter.native_library", return_value=tmp_path / "lib.so"),
        patch("vox_magpie.adapter.tokenizer_dir", return_value=tmp_path / "tokenizer"),
        patch("vox_magpie.adapter._resolve_threads", return_value=3),
    ):
        adapter = MagpieTTSAdapter()
        adapter.load(str(tmp_path), "cpu")
    return adapter


def test_magpie_metadata_and_voices():
    from vox_magpie.adapter import MagpieTTSAdapter

    adapter = MagpieTTSAdapter()
    info = adapter.info()

    assert info.name == "magpie-tts-gguf"
    assert info.type == ModelType.TTS
    assert info.supported_formats == (ModelFormat.GGUF,)
    assert info.supports_streaming is True
    assert info.supports_voice_cloning is False
    assert info.supported_languages == ("en", "es", "de", "fr", "it", "vi", "hi", "ar", "ko", "pt")
    assert [voice.id for voice in adapter.list_voices()] == ["aria", "jason", "john", "leo", "sofia"]
    assert adapter.estimate_vram_bytes() == 0


def test_default_threads_reserves_one_cgroup_cpu():
    with patch("vox_magpie.adapter._available_cpus", return_value=4):
        from vox_magpie.adapter import _default_threads

        assert _default_threads() == 3


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(1, (1, 1)), (2, (1, 1)), (3, (2, 1)), (4, (2, 2)), (8, (4, 4))],
)
def test_native_thread_pools_share_the_configured_budget(budget: int, expected: tuple[int, int]):
    from vox_magpie.worker import _thread_allocation

    assert _thread_allocation(budget) == expected


def test_load_starts_cpu_worker_with_model_and_codec(tmp_path: Path):
    adapter = _load_adapter(tmp_path)
    host = FakeHost.instance

    assert host is not None
    assert adapter.is_loaded is True
    assert host.argv[host.argv.index("--threads") + 1] == "3"
    assert host.argv[host.argv.index("--magpie-model") + 1].endswith(
        "magpie_tts_multilingual_357m.v2607.f16.gguf"
    )
    assert host.argv[host.argv.index("--codec-model") + 1].endswith(
        "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"
    )


def test_load_rejects_cuda(tmp_path: Path):
    from vox_magpie.adapter import MagpieTTSAdapter

    with pytest.raises(Exception, match="CPU-only"):
        MagpieTTSAdapter().load(str(tmp_path), "cuda")


@pytest.mark.asyncio
async def test_synthesize_streams_each_pcm_frame_without_buffering(tmp_path: Path):
    adapter = _load_adapter(tmp_path)
    host = FakeHost.instance
    assert host is not None
    pcm = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    host.frames = [
        {"request_id": "placeholder", "audio": base64.b64encode(pcm).decode(), "sample_rate": 22050},
        {
            "request_id": "placeholder",
            "done": True,
            "stats": {"ttfa_ms": 12.0},
            "memory": {"rss_bytes": 200},
        },
    ]

    original_stream = host.stream_request

    def stream(payload, timeout):
        for frame in original_stream(payload, timeout):
            yield {**frame, "request_id": payload["request_id"]}

    host.stream_request = stream
    chunks = [chunk async for chunk in adapter.synthesize("Hola", voice="sofia", language="es")]

    assert chunks[-1].is_final is True
    samples = np.frombuffer(chunks[0].audio, dtype=np.float32)
    assert np.allclose(samples[:3], [0.0, 0.5, -0.5])
    assert host.payload["voice"] == "Sofia"
    assert host.payload["language"] == "es-ES"
    assert adapter.memory_status()["last_synthesis"]["ttfa_ms"] == 12.0


@pytest.mark.asyncio
async def test_closing_stream_cancels_and_drains_worker(tmp_path: Path):
    adapter = _load_adapter(tmp_path)
    host = FakeHost.instance
    assert host is not None
    pcm = np.array([0, 1], dtype="<i2").tobytes()

    def stream(payload, timeout):
        host.payload = payload
        yield {
            "request_id": payload["request_id"],
            "audio": base64.b64encode(pcm).decode(),
            "sample_rate": 22050,
        }
        assert host.cancelled.wait(timeout=2)
        yield {"request_id": payload["request_id"], "done": True, "cancelled": True}

    host.stream_request = stream
    synthesis = adapter.synthesize("Please stop me")
    first = await anext(synthesis)
    assert first.audio
    await asyncio.wait_for(synthesis.aclose(), timeout=2)

    assert host.controls == [{"op": "cancel", "request_id": host.payload["request_id"]}]
    assert host.alive is True


@pytest.mark.asyncio
async def test_cancellation_before_first_audio_cancels_and_drains_worker(tmp_path: Path):
    adapter = _load_adapter(tmp_path)
    host = FakeHost.instance
    assert host is not None
    started = threading.Event()

    def stream(payload, timeout):
        host.payload = payload
        started.set()
        assert host.cancelled.wait(timeout=2)
        yield {"request_id": payload["request_id"], "done": True, "cancelled": True}

    host.stream_request = stream
    synthesis = adapter.synthesize("Cancel before audio")
    task = asyncio.create_task(anext(synthesis))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)

    assert host.controls == [{"op": "cancel", "request_id": host.payload["request_id"]}]
    assert host.alive is True


@pytest.mark.parametrize("language", ["ja", "zh", "xx"])
def test_rejects_languages_missing_from_official_cpu_runtime(language: str):
    from vox_magpie.adapter import MagpieTTSAdapter

    with pytest.raises(InvalidConfigError, match="does not support"):
        MagpieTTSAdapter().validate_synthesis_request(language=language)


def test_rejects_unknown_voice_and_reference_cloning():
    from vox_magpie.adapter import MagpieTTSAdapter

    adapter = MagpieTTSAdapter()
    with pytest.raises(InvalidConfigError, match="Unknown Magpie voice"):
        adapter.validate_synthesis_request(voice="unknown")
    with pytest.raises(InvalidConfigError, match="does not support reference"):
        adapter.validate_synthesis_request(reference_text="hello")


def _tar_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _tar_gz_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_tokenizer_range_extractor_stops_before_model_weights(tmp_path: Path):
    from vox_magpie.runtime import _extract_tar_segment

    first = _tar_bytes([("tokenizer.txt", b"tokenizer"), ("model_weights.ckpt", b"secret")])
    second = _tar_bytes([("model_config.yaml", b"config")])
    stream = io.BytesIO(first + second)
    destination = tmp_path / "tokenizer"
    destination.mkdir()

    _extract_tar_segment(stream, destination, len(first), "model_weights.ckpt")
    _extract_tar_segment(stream, destination, len(second), None)

    assert (destination / "tokenizer.txt").read_bytes() == b"tokenizer"
    assert (destination / "model_config.yaml").read_bytes() == b"config"
    assert not (destination / "model_weights.ckpt").exists()


def test_runtime_install_is_atomic_and_versioned(tmp_path: Path, monkeypatch):
    import vox_magpie.runtime as runtime

    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime, "_REQUIRED_NATIVE_LIBRARIES", ("libnemo_speech_tts.so",))
    fixture_hash = __import__("hashlib").sha256(b"fixture").hexdigest()
    monkeypatch.setattr(runtime, "TOKENIZER_MEMBERS", {"model_config.yaml": (7, fixture_hash)})
    archive_payload = _tar_gz_bytes([("release/lib/libnemo_speech_tts.so", b"native")])
    monkeypatch.setattr(runtime, "NEMO_SPEECH_SHA256", __import__("hashlib").sha256(archive_payload).hexdigest())

    def download(url, destination, *, byte_range=None):
        assert byte_range is None
        destination.write_bytes(archive_payload)

    def install_tokenizer(stage):
        target = stage / "tokenizer"
        target.mkdir()
        for name in runtime.TOKENIZER_MEMBERS:
            (target / name).write_bytes(b"fixture")

    monkeypatch.setattr(runtime, "_download", download)
    monkeypatch.setattr(runtime, "_install_tokenizer", install_tokenizer)

    result = runtime.ensure_runtime()

    assert (result / "native" / "lib" / "libnemo_speech_tts.so").read_bytes() == b"native"
    assert json.loads((result / "runtime.json").read_text())["nemo_speech_version"] == "0.2.0"


def test_runtime_checksum_failure_preserves_existing_directory(tmp_path: Path, monkeypatch):
    import vox_magpie.runtime as runtime

    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    target = runtime.runtime_dir()
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("stable")

    def download(url, destination, *, byte_range=None):
        destination.write_bytes(b"corrupt")

    monkeypatch.setattr(runtime, "_download", download)
    with pytest.raises(RuntimeError, match="checksum"):
        runtime.ensure_runtime()

    assert (target / "keep.txt").read_text() == "stable"


def test_worker_cancellation_stops_native_callback_and_keeps_protocol_aligned(monkeypatch):
    import vox_magpie.worker as worker

    class FakeNative:
        sample_rate = 22050
        voices = ["Aria", "Jason", "John", "Leo", "Sofia"]
        version = "fixture"
        thread_budget = 3
        generator_threads = 2
        codec_threads = 1

        def __init__(self, **kwargs):
            self.closed = threading.Event()

        def synthesize(self, request, callback):
            pcm = (ctypes.c_uint8 * 4)(0, 0, 1, 0)
            for _ in range(100):
                if not callback(pcm, len(pcm), None):
                    return 3, worker.SynthesisStats()
                time.sleep(0.005)
            return worker.STATUS_OK, worker.SynthesisStats()

        def last_error(self):
            return ""

        def close(self):
            self.closed.set()

    engine = FakeNative()
    monkeypatch.setattr(worker, "NativeMagpie", lambda **kwargs: engine)
    parent, child = socket.socketpair()
    monkeypatch.setenv(worker.WORKER_FD_ENV, str(child.fileno()))
    monkeypatch.delenv("VOX_WORKER_PARENT_PID", raising=False)
    outcomes = []
    args = Namespace(
        library="lib.so",
        magpie_model="magpie.gguf",
        codec_model="codec.gguf",
        tokenizer_dir="tokenizer",
        threads=3,
    )

    def run():
        try:
            outcomes.append(worker.run_worker(args))
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    stream = parent.makefile("rwb")
    ready = json.loads(stream.readline())
    child.close()
    assert ready["ready"] is True
    request = {"op": "synthesize", "request_id": "one", "text": "hello", "voice": "John", "language": "en-US"}
    stream.write(json.dumps(request).encode() + b"\n")
    stream.flush()
    first = json.loads(stream.readline())
    assert first["request_id"] == "one"
    assert "audio" in first
    stream.write(b'{"op":"cancel","request_id":"one"}\n')
    stream.flush()
    frames = []
    while True:
        frame = json.loads(stream.readline())
        frames.append(frame)
        if frame.get("done") is True:
            break
    stream.close()
    parent.close()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert frames[-1]["cancelled"] is True
    assert frames[-1]["request_id"] == "one"
    assert engine.closed.is_set()
    assert outcomes == [0]


def test_worker_parent_disconnect_cancels_active_native_synthesis(monkeypatch):
    import vox_magpie.worker as worker

    class FakeNative:
        sample_rate = 22050
        voices = ["Aria", "Jason", "John", "Leo", "Sofia"]
        version = "fixture"
        thread_budget = 3
        generator_threads = 2
        codec_threads = 1

        def __init__(self, **kwargs):
            self.closed = threading.Event()

        def synthesize(self, request, callback):
            pcm = (ctypes.c_uint8 * 4)(0, 0, 1, 0)
            for _ in range(100):
                if not callback(pcm, len(pcm), None):
                    return worker.STATUS_CANCELLED, worker.SynthesisStats()
                time.sleep(0.005)
            return worker.STATUS_OK, worker.SynthesisStats()

        def last_error(self):
            return ""

        def close(self):
            self.closed.set()

    engine = FakeNative()
    monkeypatch.setattr(worker, "NativeMagpie", lambda **kwargs: engine)
    parent, child = socket.socketpair()
    monkeypatch.setenv(worker.WORKER_FD_ENV, str(child.fileno()))
    monkeypatch.delenv("VOX_WORKER_PARENT_PID", raising=False)
    outcomes = []
    args = Namespace(
        library="lib.so",
        magpie_model="magpie.gguf",
        codec_model="codec.gguf",
        tokenizer_dir="tokenizer",
        threads=3,
    )

    def run():
        try:
            outcomes.append(worker.run_worker(args))
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    stream = parent.makefile("rwb")
    assert json.loads(stream.readline())["ready"] is True
    child.close()
    request = {"op": "synthesize", "request_id": "one", "text": "hello", "voice": "John", "language": "en-US"}
    stream.write(json.dumps(request).encode() + b"\n")
    stream.flush()
    assert "audio" in json.loads(stream.readline())
    stream.close()
    parent.close()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert engine.closed.is_set()
    assert outcomes == [0]


def test_worker_does_not_destroy_native_engine_while_synthesis_thread_is_stuck(monkeypatch):
    import vox_magpie.worker as worker

    class FakeNative:
        sample_rate = 22050
        voices = ["Aria", "Jason", "John", "Leo", "Sofia"]
        version = "fixture"
        thread_budget = 3
        generator_threads = 2
        codec_threads = 1

        def __init__(self, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    class StuckThread:
        def __init__(self, **kwargs):
            self.join_timeout = None

        def start(self):
            return None

        def join(self, timeout=None):
            self.join_timeout = timeout

        def is_alive(self):
            return True

    engine = FakeNative()
    monkeypatch.setattr(worker, "NativeMagpie", lambda **kwargs: engine)
    monkeypatch.setattr(worker.threading, "Thread", StuckThread)
    parent, child = socket.socketpair()
    monkeypatch.setenv(worker.WORKER_FD_ENV, str(child.fileno()))
    monkeypatch.delenv("VOX_WORKER_PARENT_PID", raising=False)
    request = {"op": "synthesize", "request_id": "one", "text": "hello", "voice": "John", "language": "en-US"}
    parent.sendall(json.dumps(request).encode() + b"\n")
    parent.shutdown(socket.SHUT_WR)
    args = Namespace(
        library="lib.so",
        magpie_model="magpie.gguf",
        codec_model="codec.gguf",
        tokenizer_dir="tokenizer",
        threads=3,
    )

    assert worker.run_worker(args) == 0

    parent.close()
    child.close()
    assert engine.closed is False
