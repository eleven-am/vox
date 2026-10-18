from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

from vox.core.process_memory import process_memory_status
from vox.core.worker_host import WORKER_FD_ENV, install_parent_death_signal, worker_parent_lost

STATUS_OK = 0
STATUS_CANCELLED = 4


def _thread_allocation(total: int) -> tuple[int, int]:
    if total <= 1:
        return 1, 1
    codec = max(1, total // 2)
    return total - codec, codec


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magpie_model", ctypes.c_char_p),
        ("codec_model", ctypes.c_char_p),
        ("tokenizer_model_dir", ctypes.c_char_p),
        ("text_normalizer_model_dir", ctypes.c_char_p),
    ]


class RuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("speaker", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("codec_threads", ctypes.c_int32),
        ("seed", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("chunk_frames", ctypes.c_int32),
        ("codec_queue_depth", ctypes.c_int32),
        ("codec_history_frames", ctypes.c_int32),
        ("codec_future_frames", ctypes.c_int32),
        ("window_ms", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("override_temperature", ctypes.c_bool),
        ("cfg_scale", ctypes.c_float),
        ("override_cfg_scale", ctypes.c_bool),
        ("use_cfg", ctypes.c_bool),
        ("use_local_transformer", ctypes.c_bool),
        ("use_kv_cache", ctypes.c_bool),
        ("use_stateful_codec", ctypes.c_bool),
        ("codec_cpu", ctypes.c_bool),
        ("flush_partial_chunk", ctypes.c_bool),
        ("verbose", ctypes.c_bool),
        ("lt_backend", ctypes.c_int),
        ("sampling_backend", ctypes.c_int),
        ("uma_mode", ctypes.c_int),
        ("longform_mode", ctypes.c_int),
        ("lt_fp32", ctypes.c_bool),
    ]


class SynthesizerConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model", ctypes.POINTER(ModelConfig)),
        ("runtime", ctypes.POINTER(RuntimeConfig)),
        ("default_language_code", ctypes.c_char_p),
        ("default_voice_name", ctypes.c_char_p),
    ]


class SynthesisOptions(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("request_id", ctypes.c_char_p),
        ("language_code", ctypes.c_char_p),
        ("speaker", ctypes.c_int32),
        ("seed", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("override_temperature", ctypes.c_bool),
        ("cfg_scale", ctypes.c_float),
        ("override_cfg_scale", ctypes.c_bool),
        ("voice_name", ctypes.c_char_p),
        ("output_sample_rate", ctypes.c_int32),
    ]


class SynthesisStats(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("sample_rate", ctypes.c_int32),
        ("generated_frames", ctypes.c_int32),
        ("chunks", ctypes.c_int32),
        ("e2e_chunks", ctypes.c_int32),
        ("samples_written", ctypes.c_uint64),
        *[(name, ctypes.c_double) for name in (
            "tokenizer_ms",
            "encoder_ms",
            "audio_s",
            "elapsed_s",
            "rtf",
            "rtfx",
            "ttfa_ms",
            "icl_avg_ms",
            "icl_min_ms",
            "icl_max_ms",
            "decoder_audio_s",
            "decoder_elapsed_s",
            "decoder_rtfx",
            "decoder_ttft_ms",
            "decoder_itl_avg_ms",
            "decoder_itl_min_ms",
            "decoder_itl_max_ms",
            "decoder_itl_p95_ms",
            "decoder_itl_p99_ms",
            "codec_audio_s",
            "codec_elapsed_s",
            "codec_rtfx",
            "codec_ttfa_ms",
            "codec_icl_avg_ms",
            "codec_icl_min_ms",
            "codec_icl_max_ms",
            "codec_icl_p95_ms",
            "codec_icl_p99_ms",
            "e2e_ttfa_ms",
            "e2e_icl_avg_ms",
            "e2e_icl_min_ms",
            "e2e_icl_max_ms",
            "e2e_icl_p95_ms",
            "e2e_icl_p99_ms",
            "e2e_rtfx",
        )],
    ]


PcmCallback = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_void_p,
)


class NativeMagpie:
    def __init__(
        self,
        *,
        library: Path,
        magpie_model: Path,
        codec_model: Path,
        tokenizer_dir: Path,
        threads: int,
    ) -> None:
        self._library = ctypes.CDLL(str(library))
        lib = self._library
        lib.nemo_speech_tts_runtime_config_default.restype = RuntimeConfig
        lib.nemo_speech_tts_synthesis_options_default.restype = SynthesisOptions
        lib.nemo_speech_tts_synthesis_stats_default.restype = SynthesisStats
        lib.nemo_speech_tts_create.argtypes = [ctypes.POINTER(SynthesizerConfig), ctypes.POINTER(ctypes.c_void_p)]
        lib.nemo_speech_tts_create.restype = ctypes.c_int
        lib.nemo_speech_tts_destroy.argtypes = [ctypes.c_void_p]
        lib.nemo_speech_tts_destroy.restype = None
        lib.nemo_speech_tts_sample_rate.argtypes = [ctypes.c_void_p]
        lib.nemo_speech_tts_sample_rate.restype = ctypes.c_int32
        lib.nemo_speech_tts_speaker_count.argtypes = [ctypes.c_void_p]
        lib.nemo_speech_tts_speaker_count.restype = ctypes.c_int32
        lib.nemo_speech_tts_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.nemo_speech_tts_speaker_name.restype = ctypes.c_char_p
        lib.nemo_speech_tts_synthesize_text.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(SynthesisOptions),
            ctypes.c_char_p,
            PcmCallback,
            ctypes.c_void_p,
            ctypes.POINTER(SynthesisStats),
        ]
        lib.nemo_speech_tts_synthesize_text.restype = ctypes.c_int
        lib.nemo_speech_tts_last_error.restype = ctypes.c_char_p
        lib.nemo_speech_tts_version.restype = ctypes.c_char_p

        model_bytes = str(magpie_model).encode()
        codec_bytes = str(codec_model).encode()
        tokenizer_bytes = str(tokenizer_dir).encode()
        model = ModelConfig(
            size=ctypes.sizeof(ModelConfig),
            magpie_model=model_bytes,
            codec_model=codec_bytes,
            tokenizer_model_dir=tokenizer_bytes,
            text_normalizer_model_dir=None,
        )
        runtime = lib.nemo_speech_tts_runtime_config_default()
        generator_threads, codec_threads = _thread_allocation(threads)
        runtime.threads = generator_threads
        runtime.codec_threads = codec_threads
        runtime.chunk_frames = 4
        runtime.codec_queue_depth = 2
        runtime.lt_backend = 1
        runtime.sampling_backend = 1
        config = SynthesizerConfig(
            size=ctypes.sizeof(SynthesizerConfig),
            model=ctypes.pointer(model),
            runtime=ctypes.pointer(runtime),
            default_language_code=b"en-US",
            default_voice_name=b"John",
        )
        handle = ctypes.c_void_p()
        status = lib.nemo_speech_tts_create(ctypes.byref(config), ctypes.byref(handle))
        if status != STATUS_OK or not handle.value:
            raise RuntimeError(self.last_error() or f"native Magpie load failed with status {status}")
        self._handle = handle
        self.thread_budget = threads
        self.generator_threads = generator_threads
        self.codec_threads = codec_threads

    @property
    def sample_rate(self) -> int:
        return int(self._library.nemo_speech_tts_sample_rate(self._handle))

    @property
    def voices(self) -> list[str]:
        count = int(self._library.nemo_speech_tts_speaker_count(self._handle))
        return [
            self._library.nemo_speech_tts_speaker_name(self._handle, index).decode()
            for index in range(count)
        ]

    @property
    def version(self) -> str:
        return self._library.nemo_speech_tts_version().decode()

    def last_error(self) -> str:
        value = self._library.nemo_speech_tts_last_error()
        return value.decode(errors="replace") if value else ""

    def synthesize(
        self,
        request: dict[str, Any],
        callback: PcmCallback,
    ) -> tuple[int, SynthesisStats]:
        options = self._library.nemo_speech_tts_synthesis_options_default()
        request_id = str(request["request_id"]).encode()
        language = str(request["language"]).encode()
        voice = str(request["voice"]).encode()
        options.request_id = request_id
        options.language_code = language
        options.voice_name = voice
        options.seed = int(request.get("seed", -1))
        temperature = request.get("temperature")
        if temperature is not None:
            options.temperature = float(temperature)
            options.override_temperature = True
        cfg_scale = request.get("cfg_scale")
        if cfg_scale is not None:
            options.cfg_scale = float(cfg_scale)
            options.override_cfg_scale = True
        stats = self._library.nemo_speech_tts_synthesis_stats_default()
        status = self._library.nemo_speech_tts_synthesize_text(
            self._handle,
            ctypes.byref(options),
            str(request["text"]).encode(),
            callback,
            None,
            ctypes.byref(stats),
        )
        return int(status), stats

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            self._library.nemo_speech_tts_destroy(handle)
            handle.value = None


def _stats_payload(stats: SynthesisStats) -> dict[str, Any]:
    return {
        "sample_rate": int(stats.sample_rate),
        "generated_frames": int(stats.generated_frames),
        "chunks": int(stats.chunks),
        "samples_written": int(stats.samples_written),
        "audio_s": float(stats.audio_s),
        "elapsed_s": float(stats.elapsed_s),
        "rtfx": float(stats.rtfx),
        "ttfa_ms": float(stats.e2e_ttfa_ms or stats.ttfa_ms),
        "inter_chunk_ms": float(stats.e2e_icl_avg_ms or stats.icl_avg_ms),
    }


def run_worker(args: argparse.Namespace) -> int:
    install_parent_death_signal()
    if worker_parent_lost():
        return 1
    engine = NativeMagpie(
        library=Path(args.library),
        magpie_model=Path(args.magpie_model),
        codec_model=Path(args.codec_model),
        tokenizer_dir=Path(args.tokenizer_dir),
        threads=args.threads,
    )
    sock = socket.socket(fileno=os.dup(int(os.environ[WORKER_FD_ENV])))
    stream = sock.makefile("rwb")
    write_lock = threading.Lock()
    state_lock = threading.Lock()
    cancel_event = threading.Event()
    active_request: str | None = None
    active_thread: threading.Thread | None = None

    def send(payload: dict[str, Any]) -> bool:
        try:
            encoded = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
            with write_lock:
                stream.write(encoded)
                stream.flush()
            return True
        except OSError:
            return False

    send(
        {
            "ready": True,
            "sample_rate": engine.sample_rate,
            "voices": engine.voices,
            "version": engine.version,
            "thread_budget": engine.thread_budget,
            "generator_threads": engine.generator_threads,
            "codec_threads": engine.codec_threads,
            "memory": process_memory_status(),
        }
    )

    def synthesize(request: dict[str, Any]) -> None:
        nonlocal active_request
        request_id = str(request["request_id"])

        @PcmCallback
        def emit(pcm: ctypes.POINTER(ctypes.c_uint8), size: int, _user_data: int) -> bool:
            if cancel_event.is_set() or worker_parent_lost():
                return False
            audio = ctypes.string_at(pcm, size)
            return send(
                {
                    "request_id": request_id,
                    "audio": base64.b64encode(audio).decode("ascii"),
                    "sample_rate": engine.sample_rate,
                }
            )

        try:
            status, stats = engine.synthesize(request, emit)
            locally_cancelled = cancel_event.is_set()
            cancelled = locally_cancelled or status == STATUS_CANCELLED
            with state_lock:
                active_request = None
            if not locally_cancelled and status not in (STATUS_OK, STATUS_CANCELLED):
                send({"request_id": request_id, "done": True, "error": engine.last_error()})
                return
            send(
                {
                    "request_id": request_id,
                    "done": True,
                    "cancelled": cancelled,
                    "stats": _stats_payload(stats),
                    "memory": process_memory_status(),
                }
            )
        except Exception as error:
            with state_lock:
                active_request = None
            send({"request_id": request_id, "done": True, "error": f"{type(error).__name__}: {error}"})

    try:
        for line in stream:
            request = json.loads(line)
            op = request.get("op")
            if op == "cancel":
                with state_lock:
                    if active_request == str(request.get("request_id")):
                        cancel_event.set()
                continue
            if op != "synthesize":
                send({"done": True, "error": f"unknown Magpie worker op: {op}"})
                continue
            with state_lock:
                if active_request is not None:
                    send({"request_id": request.get("request_id"), "done": True, "error": "worker is busy"})
                    continue
                cancel_event.clear()
                active_request = str(request["request_id"])
                active_thread = threading.Thread(target=synthesize, args=(request,), name="magpie-synthesis")
                active_thread.start()
    finally:
        cancel_event.set()
        if active_thread is not None:
            active_thread.join(timeout=5)
        if active_thread is None or not active_thread.is_alive():
            engine.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vox-magpie-worker")
    parser.add_argument("--library", required=True)
    parser.add_argument("--magpie-model", required=True)
    parser.add_argument("--codec-model", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--threads", type=int, required=True)
    return run_worker(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
