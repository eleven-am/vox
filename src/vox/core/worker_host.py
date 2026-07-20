from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, NoReturn

from vox.core.errors import ModelLoadError, VoxError

logger = logging.getLogger(__name__)

WORKER_FD_ENV = "VOX_WORKER_PROTOCOL_FD"
WORKER_PARENT_PID_ENV = "VOX_WORKER_PARENT_PID"
_PR_SET_PDEATHSIG = 1


class WorkerError(VoxError):
    pass


def _drain_stderr(fd: int, name: str) -> None:
    with os.fdopen(fd, "rb") as stream:
        for raw in stream:
            logger.info("[worker %s] %s", name, raw.decode(errors="replace").rstrip())


class WorkerHost:
    def __init__(
        self,
        argv: list[str],
        *,
        env: dict[str, str],
        name: str,
        startup_timeout: float = 60.0,
    ) -> None:
        self._name = name
        self._alive = False
        self._buffer = bytearray()
        self._request_lock = threading.Lock()
        parent_sock, child_sock = socket.socketpair()
        stderr_read, stderr_write = os.pipe()
        try:
            self._proc = subprocess.Popen(
                argv,
                env={
                    **env,
                    WORKER_FD_ENV: str(child_sock.fileno()),
                    WORKER_PARENT_PID_ENV: str(os.getpid()),
                },
                stdin=subprocess.PIPE,
                stdout=stderr_write,
                stderr=stderr_write,
                pass_fds=(child_sock.fileno(),),
                start_new_session=True,
            )
        except Exception:
            parent_sock.close()
            os.close(stderr_read)
            raise
        finally:
            os.close(stderr_write)
            child_sock.close()
        self._sock = parent_sock
        self._alive = True
        threading.Thread(
            target=_drain_stderr,
            args=(stderr_read, name),
            name=f"{name}-worker-stderr",
            daemon=True,
        ).start()
        try:
            frame = self._read_frame(startup_timeout)
            if frame.get("ready") is not True:
                self._anomaly(f"expected ready frame, got {frame}")
        except WorkerError as error:
            raise ModelLoadError(f"{name} worker failed to start: {error}") from error

    @property
    def alive(self) -> bool:
        return self._alive and self._proc.poll() is None

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        with self._request_lock:
            if not self.alive:
                raise WorkerError(f"{self._name} worker is not alive")
            try:
                self._sock.sendall(json.dumps(payload).encode() + b"\n")
            except OSError as error:
                self._anomaly(f"protocol write failed: {error}")
            frame = self._read_frame(timeout)
            if "error" in frame:
                raise WorkerError(f"{self._name} worker request failed: {frame['error']}")
            return frame

    def close(self, grace: float = 5.0) -> None:
        self._alive = False
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        with self._request_lock:
            self._close_pipes()
        try:
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self._kill_group(signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=grace)
        self._kill_group(signal.SIGKILL)
        self._proc.wait()

    def _anomaly(self, reason: str) -> NoReturn:
        self._alive = False
        self._close_pipes()
        self._kill_group(signal.SIGKILL)
        self._proc.wait()
        raise WorkerError(f"{self._name} worker killed: {reason} (exit code {self._proc.returncode})")

    def _read_frame(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[: newline])
                del self._buffer[: newline + 1]
                try:
                    frame = json.loads(line)
                except ValueError:
                    self._anomaly(f"garbage frame: {line[:200]!r}")
                if not isinstance(frame, dict):
                    self._anomaly(f"garbage frame: {line[:200]!r}")
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._anomaly(f"timed out after {timeout}s")
            self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(65536)
            except TimeoutError:
                self._anomaly(f"timed out after {timeout}s")
            except OSError as error:
                self._anomaly(f"protocol read failed: {error}")
            if not chunk:
                self._anomaly("protocol stream closed")
            self._buffer += chunk

    def _close_pipes(self) -> None:
        self._sock.close()
        if self._proc.stdin is not None:
            with contextlib.suppress(OSError):
                self._proc.stdin.close()

    def _kill_group(self, sig: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self._proc.pid, sig)


def install_parent_death_signal() -> None:
    if sys.platform != "linux":
        return
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return
    libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL))


def worker_parent_lost() -> bool:
    expected = os.environ.get(WORKER_PARENT_PID_ENV)
    return expected is not None and os.getppid() != int(expected)


def worker_main(handler: Callable[[dict[str, Any]], dict[str, Any]]) -> int:
    install_parent_death_signal()
    sock = socket.socket(fileno=os.dup(int(os.environ[WORKER_FD_ENV])))
    os.dup2(2, 1)
    stream = sock.makefile("rwb")
    stream.write(b'{"ready": true}\n')
    stream.flush()
    for line in stream:
        request = json.loads(line)
        try:
            response = handler(request)
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        stream.write(json.dumps(response).encode() + b"\n")
        stream.flush()
    return 0
