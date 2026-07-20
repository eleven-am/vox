from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import vox.core.worker_host as worker_host_module
from vox.core.errors import ModelLoadError
from vox.core.worker_host import WorkerError, WorkerHost

SRC_DIR = str(Path(worker_host_module.__file__).resolve().parents[2])

WORKER_SOURCE = """
import json, os, socket, subprocess, sys, time
from vox.core.worker_host import WORKER_FD_ENV, worker_main

mode = sys.argv[1]

if mode == "no-ready":
    time.sleep(60)
elif mode == "ready-error":
    stream = socket.socket(fileno=int(os.environ[WORKER_FD_ENV])).makefile("rwb")
    stream.write(b'{"error": "model weights missing"}\\n')
    stream.flush()
    time.sleep(60)
elif mode == "garbage-frame":
    stream = socket.socket(fileno=int(os.environ[WORKER_FD_ENV])).makefile("rwb")
    stream.write(b'{"ready": true}\\n')
    stream.flush()
    stream.readline()
    stream.write(b"this is not json\\n")
    stream.flush()
    time.sleep(60)
else:
    def handler(request):
        op = request["op"]
        if op == "echo":
            return {"echo": request["value"]}
        if op == "noisy-echo":
            os.write(1, b"NATIVE GARBAGE 0.53 [chunk]\\n")
            return {"echo": request["value"]}
        if op == "stderr-flood":
            sys.stderr.write("flood-marker\\n")
            sys.stderr.write("x" * 262144 + "\\n")
            sys.stderr.flush()
            return {"flooded": True}
        if op == "sleep":
            time.sleep(request["seconds"])
            return {"slept": True}
        if op == "boom":
            raise RuntimeError("handler exploded")
        if op == "die":
            os._exit(3)
        if op == "spawn-grandchild":
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
            return {"grandchild_pid": proc.pid}
        raise RuntimeError("unknown op: " + op)

    sys.exit(worker_main(handler))
"""


def spawn(mode: str = "harness", *, startup_timeout: float = 30.0) -> WorkerHost:
    return WorkerHost(
        [sys.executable, "-c", WORKER_SOURCE, mode],
        env={"PYTHONPATH": SRC_DIR},
        name="synthetic",
        startup_timeout=startup_timeout,
    )


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_happy_path_round_trips():
    host = spawn()
    try:
        assert host.alive
        for value in ("one", "two", "three"):
            assert host.request({"op": "echo", "value": value}, timeout=10.0) == {"echo": value}
    finally:
        host.close(grace=2.0)
    assert not host.alive


def test_native_fd1_garbage_does_not_corrupt_protocol():
    host = spawn()
    try:
        for value in ("a", "b", "c"):
            assert host.request({"op": "noisy-echo", "value": value}, timeout=10.0) == {"echo": value}
        assert host.alive
    finally:
        host.close(grace=2.0)


def test_stderr_chatter_is_drained_without_blocking(caplog):
    with caplog.at_level(logging.INFO, logger="vox.core.worker_host"):
        host = spawn()
        try:
            assert host.request({"op": "stderr-flood"}, timeout=10.0) == {"flooded": True}
            assert host.request({"op": "echo", "value": "after-flood"}, timeout=10.0) == {"echo": "after-flood"}
            assert wait_until(lambda: "flood-marker" in caplog.text)
        finally:
            host.close(grace=2.0)


def test_request_timeout_kills_process_group():
    host = spawn()
    pid = host._proc.pid
    with pytest.raises(WorkerError, match="timed out"):
        host.request({"op": "sleep", "seconds": 30}, timeout=0.4)
    assert not host.alive
    assert host._proc.returncode is not None
    assert process_gone(pid)
    with pytest.raises(WorkerError, match="not alive"):
        host.request({"op": "echo", "value": "late"}, timeout=1.0)


def test_garbage_protocol_frame_kills_worker():
    host = spawn("garbage-frame")
    with pytest.raises(WorkerError, match="garbage frame"):
        host.request({"op": "echo", "value": "x"}, timeout=10.0)
    assert not host.alive
    assert process_gone(host._proc.pid)


def test_nonzero_exit_mid_request_fails_loudly():
    host = spawn()
    with pytest.raises(WorkerError, match="exit code 3"):
        host.request({"op": "die"}, timeout=10.0)
    assert not host.alive
    assert host._proc.returncode == 3


def test_close_kills_whole_process_group_including_grandchild():
    host = spawn()
    response = host.request({"op": "spawn-grandchild"}, timeout=10.0)
    grandchild_pid = response["grandchild_pid"]
    assert not process_gone(grandchild_pid)
    host.close(grace=2.0)
    assert not host.alive
    assert process_gone(host._proc.pid)
    assert wait_until(lambda: process_gone(grandchild_pid))


def test_startup_timeout_raises_model_load_error(monkeypatch):
    spawned_pids: list[int] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned_pids.append(proc.pid)
        return proc

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    with pytest.raises(ModelLoadError, match="timed out"):
        spawn("no-ready", startup_timeout=0.4)
    assert len(spawned_pids) == 1
    assert wait_until(lambda: process_gone(spawned_pids[0]))


def test_startup_error_frame_raises_model_load_error():
    with pytest.raises(ModelLoadError, match="model weights missing"):
        spawn("ready-error")


def test_close_during_inflight_request_does_not_hang():
    host = spawn()
    pid = host._proc.pid
    outcomes: list[BaseException] = []

    def run_request():
        try:
            host.request({"op": "sleep", "seconds": 30}, timeout=10.0)
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=run_request)
    thread.start()
    assert wait_until(lambda: host._request_lock.locked())
    time.sleep(0.5)
    host.close(grace=2.0)
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], WorkerError)
    assert "protocol stream closed" in str(outcomes[0])
    assert not host.alive
    assert process_gone(pid)


def count_open_fds() -> int:
    return len(os.listdir("/dev/fd"))


def test_spawn_close_cycles_keep_fd_count_flat():
    warmup = spawn()
    warmup.close(grace=2.0)
    baseline = count_open_fds()
    for cycle in range(5):
        host = spawn()
        assert host.request({"op": "echo", "value": f"fd-{cycle}"}, timeout=10.0) == {"echo": f"fd-{cycle}"}
        host.close(grace=2.0)
    assert wait_until(lambda: count_open_fds() <= baseline)


def test_parent_death_signal_noops_off_linux(monkeypatch):
    monkeypatch.setattr(worker_host_module.sys, "platform", "darwin")

    def fail_cdll(*_args, **_kwargs):
        raise AssertionError("ctypes.CDLL must not be used off Linux")

    monkeypatch.setattr("ctypes.CDLL", fail_cdll)

    assert worker_host_module._install_parent_death_signal() is None


def test_parent_death_signal_requests_sigkill_on_linux(monkeypatch):
    monkeypatch.setattr(worker_host_module.sys, "platform", "linux")
    calls: list[tuple[int, ...]] = []

    class _FakeLibc:
        def prctl(self, *args: int) -> int:
            calls.append(args)
            return 0

    monkeypatch.setattr("ctypes.CDLL", lambda *_a, **_k: _FakeLibc())

    worker_host_module._install_parent_death_signal()

    assert calls == [(1, int(signal.SIGKILL))]


def test_handler_error_frame_fails_request_but_worker_survives():
    host = spawn()
    try:
        with pytest.raises(WorkerError, match="handler exploded"):
            host.request({"op": "boom"}, timeout=10.0)
        assert host.alive
        assert host.request({"op": "echo", "value": "still-here"}, timeout=10.0) == {"echo": "still-here"}
    finally:
        host.close(grace=2.0)
