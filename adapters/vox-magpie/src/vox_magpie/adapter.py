from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.errors import ModelLoadError
from vox.core.process_memory import process_memory_status
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesisParameterInfo,
    SynthesizeChunk,
    VoiceInfo,
)
from vox.core.worker_host import WorkerHost
from vox.operations.errors import InvalidConfigError
from vox_magpie.runtime import ensure_runtime, native_library, native_root, tokenizer_dir

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22_050
DEFAULT_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_STREAM_FRAME_TIMEOUT_SECONDS = 30.0
STARTUP_TIMEOUT_ENV = "VOX_MAGPIE_STARTUP_TIMEOUT_S"
STREAM_TIMEOUT_ENV = "VOX_MAGPIE_STREAM_TIMEOUT_S"
THREADS_ENV = "VOX_MAGPIE_THREADS"

_VOICES = (
    VoiceInfo(id="aria", name="Aria"),
    VoiceInfo(id="jason", name="Jason"),
    VoiceInfo(id="john", name="John"),
    VoiceInfo(id="leo", name="Leo"),
    VoiceInfo(id="sofia", name="Sofia"),
)
_LANGUAGE_CODES = {
    "en": "en-US",
    "en-us": "en-US",
    "es": "es-ES",
    "es-es": "es-ES",
    "de": "de-DE",
    "de-de": "de-DE",
    "fr": "fr-FR",
    "fr-fr": "fr-FR",
    "it": "it-IT",
    "it-it": "it-IT",
    "vi": "vi-VN",
    "vi-vn": "vi-VN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "ar": "ar-SA",
    "ar-sa": "ar-SA",
    "ar-ae": "ar-AE",
    "ar-msa": "ar-MSA",
    "ko": "ko-KR",
    "ko-kr": "ko-KR",
    "pt": "pt-BR",
    "pt-br": "pt-BR",
}
_EXPECTED_VOICES = {voice.name for voice in _VOICES}
_WORKER_ENV_NAMES = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number of seconds, got {raw!r}") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {raw!r}")
    return value


def _cgroup_cpu_limit() -> int | None:
    try:
        quota_raw, period_raw = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
    except (OSError, ValueError):
        return None
    if quota_raw == "max":
        return None
    try:
        quota = int(quota_raw)
        period = int(period_raw)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.floor(quota / period))


def _available_cpus() -> int:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = _cgroup_cpu_limit()
    return max(1, min(affinity, quota)) if quota is not None else max(1, affinity)


def _default_threads() -> int:
    available = _available_cpus()
    reserved = 1 if available > 1 else 0
    return max(1, min(4, available - reserved))


def _resolve_threads(configured: Any) -> int:
    raw = os.environ.get(THREADS_ENV)
    value = raw if raw is not None else configured
    if value is None:
        return _default_threads()
    try:
        threads = int(value)
    except (TypeError, ValueError) as error:
        raise ModelLoadError(f"Magpie threads must be an integer, got {value!r}") from error
    if not 1 <= threads <= 16:
        raise ModelLoadError("Magpie threads must be between 1 and 16")
    return threads


def _find_model_file(model_path: Path, filename: str) -> Path:
    matches = [path for path in model_path.rglob(filename) if path.is_file()]
    if len(matches) != 1:
        raise ModelLoadError(f"Magpie model bundle must contain exactly one {filename}; found {len(matches)}")
    return matches[0]


def _worker_python_paths() -> list[str]:
    import vox

    candidates = [
        str(Path(vox.__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[1]),
    ]
    return list(dict.fromkeys(candidates))


def _worker_env() -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name in _WORKER_ENV_NAMES}
    env["PYTHONPATH"] = os.pathsep.join(_worker_python_paths())
    library_path = str(native_root() / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (library_path, existing)))
    return env


def _worker_argv(magpie_model: Path, codec_model: Path, threads: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vox_magpie.worker",
        "--library",
        str(native_library()),
        "--magpie-model",
        str(magpie_model),
        "--codec-model",
        str(codec_model),
        "--tokenizer-dir",
        str(tokenizer_dir()),
        "--threads",
        str(threads),
    ]


def _next_frame(iterator: Iterator[dict[str, Any]]) -> tuple[bool, dict[str, Any] | None]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _drain_stream(iterator: Iterator[dict[str, Any]]) -> None:
    for frame in iterator:
        if frame.get("done") is True:
            return


def _normalize_language(language: str | None) -> str:
    if language is None or language.lower() in {"", "auto"}:
        return "en-US"
    normalized = language.lower().replace("_", "-")
    try:
        return _LANGUAGE_CODES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted({key for key in _LANGUAGE_CODES if "-" not in key}))
        raise InvalidConfigError(f"Magpie does not support language={language!r}; supported: {supported}") from error


def _normalize_voice(voice: str | None) -> str:
    if voice is None or voice.lower() == "default":
        return "John"
    match = next(
        (item.name for item in _VOICES if item.id == voice.lower() or item.name.lower() == voice.lower()),
        None,
    )
    if match is None:
        raise InvalidConfigError(f"Unknown Magpie voice {voice!r}; supported: aria, jason, john, leo, sofia")
    return match


class MagpieTTSAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._host: WorkerHost | None = None
        self._device = "cpu"
        self._threads = 0
        self._generator_threads = 0
        self._codec_threads = 0
        self._lock = threading.RLock()
        self._worker_memory: dict[str, Any] = {}
        self._last_stats: dict[str, Any] | None = None
        self._worker_started_at = 0.0
        self._request_count = 0

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="magpie-tts-gguf",
            type=ModelType.TTS,
            architectures=("magpie-tts-gguf", "magpie-tts", "magpietts"),
            default_sample_rate=SAMPLE_RATE,
            supported_formats=(ModelFormat.GGUF,),
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=("en", "es", "de", "fr", "it", "vi", "hi", "ar", "ko", "pt"),
        )

    def prepare_runtime(self) -> None:
        ensure_runtime()

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        with self._lock:
            if self._host is not None:
                if self._host.alive:
                    return
                self._host.close()
                self._host = None
            if device not in {"cpu", "auto"}:
                raise ModelLoadError("Magpie is CPU-only and requires device='cpu' or 'auto'")
            ensure_runtime()
            root = Path(model_path)
            magpie_model = _find_model_file(root, "magpie_tts_multilingual_357m.v2607.f16.gguf")
            codec_model = _find_model_file(root, "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf")
            threads = _resolve_threads(kwargs.get("threads"))
            host = WorkerHost(
                _worker_argv(magpie_model, codec_model, threads),
                env=_worker_env(),
                name="magpie-tts",
                startup_timeout=_positive_float_env(STARTUP_TIMEOUT_ENV, DEFAULT_STARTUP_TIMEOUT_SECONDS),
            )
            ready = host.ready_frame
            sample_rate = int(ready.get("sample_rate") or 0)
            voices = {str(item) for item in ready.get("voices") or ()}
            thread_budget = int(ready.get("thread_budget") or 0)
            generator_threads = int(ready.get("generator_threads") or 0)
            codec_threads = int(ready.get("codec_threads") or 0)
            valid_allocation = (
                thread_budget == threads
                and generator_threads >= 1
                and codec_threads >= 1
                and (threads == 1 or generator_threads + codec_threads == threads)
            )
            if sample_rate != SAMPLE_RATE or voices != _EXPECTED_VOICES or not valid_allocation:
                host.close()
                raise ModelLoadError(
                    "Magpie worker reported unexpected metadata: "
                    f"sample_rate={sample_rate}, voices={sorted(voices)}, "
                    f"thread_budget={thread_budget}, generator_threads={generator_threads}, "
                    f"codec_threads={codec_threads}"
                )
            self._host = host
            self._device = "cpu"
            self._threads = threads
            self._generator_threads = generator_threads
            self._codec_threads = codec_threads
            self._worker_memory = dict(ready.get("memory") or {})
            self._last_stats = None
            self._worker_started_at = time.monotonic()
            self._request_count = 0
            logger.info(
                "Loaded Magpie worker version=%s thread_budget=%d generator_threads=%d codec_threads=%d pid=%d",
                ready.get("version"),
                threads,
                self._generator_threads,
                self._codec_threads,
                host.pid,
            )

    def unload(self) -> None:
        with self._lock:
            host = self._host
            self._host = None
            self._threads = 0
            self._generator_threads = 0
            self._codec_threads = 0
            self._worker_memory = {}
            self._last_stats = None
            self._worker_started_at = 0.0
            self._request_count = 0
            if host is not None:
                host.close()
        logger.info("Magpie adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        host = self._host
        return host is not None and host.alive

    def list_voices(self) -> list[VoiceInfo]:
        return list(_VOICES)

    def synthesis_parameters(self) -> tuple[SynthesisParameterInfo, ...]:
        return (
            SynthesisParameterInfo(
                name="seed",
                type="integer",
                default=None,
                min_value=0,
                max_value=2**31 - 1,
                description="Deterministic native sampling seed.",
            ),
            SynthesisParameterInfo(
                name="temperature",
                type="number",
                default=None,
                min_value=0.0,
                max_value=2.0,
                description="Override the model sampling temperature.",
            ),
            SynthesisParameterInfo(
                name="cfg_scale",
                type="number",
                default=None,
                min_value=0.0,
                max_value=10.0,
                description="Override classifier-free guidance scale.",
            ),
        )

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if reference_audio is not None or reference_text is not None:
            raise InvalidConfigError("Magpie does not support reference_audio/reference_text voice cloning")
        _normalize_voice(voice)
        _normalize_language(language)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        host = self._host
        if host is None or not host.alive:
            raise RuntimeError("Magpie model is not loaded — call load() first")
        if not text or not text.strip():
            return
        if speed != 1.0:
            raise InvalidConfigError("Magpie does not support speed values other than 1.0")
        self.validate_synthesis_request(
            voice=voice,
            language=language,
            reference_audio=reference_audio,
            reference_text=reference_text,
            params=params,
        )
        options = dict(params or {})
        request_id = uuid.uuid4().hex
        payload = {
            "op": "synthesize",
            "request_id": request_id,
            "text": text,
            "voice": _normalize_voice(voice),
            "language": _normalize_language(language),
            "seed": options.get("seed", -1),
            "temperature": options.get("temperature"),
            "cfg_scale": options.get("cfg_scale"),
        }
        iterator = host.stream_request(
            payload,
            timeout=_positive_float_env(STREAM_TIMEOUT_ENV, DEFAULT_STREAM_FRAME_TIMEOUT_SECONDS),
        )
        terminal = False
        pending: asyncio.Task[tuple[bool, dict[str, Any] | None]] | None = None
        started = time.perf_counter()
        first_audio_ms: int | None = None
        try:
            while True:
                pending = asyncio.create_task(asyncio.to_thread(_next_frame, iterator))
                has_frame, frame = await asyncio.shield(pending)
                pending = None
                if not has_frame or frame is None:
                    break
                if str(frame.get("request_id") or request_id) != request_id:
                    raise RuntimeError("Magpie worker returned a frame for the wrong request")
                if frame.get("done") is True:
                    terminal = True
                    stats = frame.get("stats")
                    memory = frame.get("memory")
                    if isinstance(stats, dict):
                        self._last_stats = dict(stats)
                    if isinstance(memory, dict):
                        self._worker_memory = dict(memory)
                    break
                encoded = frame.get("audio")
                if not isinstance(encoded, str):
                    raise RuntimeError("Magpie worker returned an audio frame without PCM")
                pcm = base64.b64decode(encoded, validate=True)
                if len(pcm) % 2:
                    raise RuntimeError("Magpie worker returned an odd-sized PCM16 frame")
                audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                if first_audio_ms is None:
                    first_audio_ms = int((time.perf_counter() - started) * 1000)
                yield SynthesizeChunk(audio=audio.tobytes(), sample_rate=SAMPLE_RATE, is_final=False)
        finally:
            if not terminal:
                host.send_control({"op": "cancel", "request_id": request_id})
                if pending is not None:
                    with contextlib.suppress(BaseException):
                        await asyncio.shield(pending)
                with contextlib.suppress(BaseException):
                    await asyncio.to_thread(_drain_stream, iterator)
            iterator.close()

        self._request_count += 1
        logger.info(
            "Magpie synthesized chars=%d first_audio_ms=%s total_ms=%d native_stats=%s",
            len(text),
            first_audio_ms,
            int((time.perf_counter() - started) * 1000),
            self._last_stats,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=SAMPLE_RATE, is_final=True)

    def memory_status(self) -> dict[str, Any]:
        host = self._host
        if host is None or not host.alive:
            return {"worker_alive": False}
        current = process_memory_status(pid=host.pid)
        result: dict[str, Any] = {
            "worker_alive": True,
            "pid": current["pid"],
            "rss_bytes": current["rss_bytes"],
            "peak_rss_bytes": current["peak_rss_bytes"],
            "thread_budget": self._threads,
            "generator_threads": self._generator_threads,
            "codec_threads": self._codec_threads,
            "worker_age_seconds": max(0.0, time.monotonic() - self._worker_started_at),
            "worker_request_count": self._request_count,
        }
        if self._last_stats is not None:
            result["last_synthesis"] = dict(self._last_stats)
        return result

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 0
