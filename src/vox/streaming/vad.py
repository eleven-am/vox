from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from vox.streaming.buffer import AudioRingBuffer
from vox.streaming.types import (
    MS_PER_SAMPLE,
    TARGET_SAMPLE_RATE,
    SpeechStarted,
    SpeechStopped,
)

logger = logging.getLogger(__name__)

VAD_WINDOW_SIZE_SAMPLES = 1000 * MS_PER_SAMPLE
MAX_BUFFER_SAMPLES = (15_000 + 3_000 + 1_000) * MS_PER_SAMPLE


@dataclass
class VADConfig:
    backend: str = "silero"
    start_threshold: float = 0.6
    continue_threshold: float = 0.4
    min_silence_duration_ms: int = 1000
    speech_pad_ms: int = 100
    speech_pre_roll_ms: int = 300
    min_speech_duration_ms: int = 250
    min_audio_duration_ms: int = 500
    max_utterance_ms: int = 15_000


@dataclass
class VADState:
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    active: bool = False
    carryover: bool = False
    utterance_id: int = 0


@dataclass
class SpeechSegment:
    audio: NDArray[np.float32]
    start_ms: int
    end_ms: int
    utterance_id: int = 0


class VADBackend(Protocol):
    """Consumes streamed audio once and carries model state across calls.

    ``process_frames`` returns ``(start_sample, end_sample, probability)`` in
    absolute stream coordinates, so the same audio yields the same frames however
    the caller chunks it. Grouping those frames into padded speech timestamps
    belongs to the processor, not the backend.
    """

    def process_frames(self, audio: NDArray[np.float32]) -> list[tuple[int, int, float]]: ...


def _frames_to_timestamps(
    speech_frames: list[tuple[int, int]],
    *,
    total_samples: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    min_speech_duration_ms: int,
) -> list[dict[str, int]]:
    """Group per-frame speech spans into padded Silero-style timestamp dicts."""
    if not speech_frames:
        return []

    min_silence_samples = int(min_silence_duration_ms * MS_PER_SAMPLE)
    pad_samples = int(speech_pad_ms * MS_PER_SAMPLE)
    min_speech_samples = int(min_speech_duration_ms * MS_PER_SAMPLE)

    raw_timestamps: list[tuple[int, int]] = []
    current_start, current_end = speech_frames[0]
    for start, end in speech_frames[1:]:
        if start - current_end <= min_silence_samples:
            current_end = end
            continue
        raw_timestamps.append((current_start, current_end))
        current_start, current_end = start, end

    raw_timestamps.append((current_start, current_end))
    return [
        {
            "start": max(0, start - pad_samples),
            "end": min(total_samples, end + pad_samples),
        }
        for start, end in raw_timestamps
        if end - start >= min_speech_samples
    ]


# Silero v5 at 16 kHz: 512-sample windows, each fed to the model with the
# previous window's trailing 64 samples of context prepended.
SILERO_ONNX_WINDOW_SAMPLES = 512
SILERO_ONNX_CONTEXT_SAMPLES = 64

# TEN's binding fixes its speech flag at construction time, so the streaming
# instance is built once at a neutral threshold and callers threshold the
# returned probability themselves.
TEN_VAD_STREAM_THRESHOLD = 0.5


class SileroOnnxVAD:
    """Default VAD backend: the Silero model run on onnxruntime (no torch).

    Uses the bundled ``silero_vad.onnx`` and replicates Silero's own
    preprocessing: 512-sample windows at 16 kHz, each prepended with the
    previous window's last 64 samples of context, carrying the recurrent
    state across windows. The onnxruntime session is a process-wide
    singleton. Batch timestamp calls use fresh state, while streaming frame
    calls preserve instance-owned context and state.
    """

    _session = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._stream_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._stream_context = np.zeros((1, SILERO_ONNX_CONTEXT_SAMPLES), dtype=np.float32)
        self._stream_pending = np.empty(0, dtype=np.float32)
        self._stream_processed_samples = 0

    @classmethod
    def _ensure_session(cls):
        if cls._session is not None:
            return cls._session
        with cls._lock:
            if cls._session is None:
                from importlib.resources import files

                import onnxruntime as ort

                model_path = files("vox.streaming.assets").joinpath("silero_vad.onnx")
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.log_severity_level = 3
                logger.info("Loading Silero VAD (onnxruntime)")
                with model_path.open("rb") as handle:
                    cls._session = ort.InferenceSession(
                        handle.read(),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
        return cls._session

    def process_frames(self, audio: NDArray[np.float32]) -> list[tuple[int, int, float]]:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size:
            self._stream_pending = np.concatenate([self._stream_pending, audio])

        session = self._ensure_session()
        sr = np.array(TARGET_SAMPLE_RATE, dtype=np.int64)
        frames: list[tuple[int, int, float]] = []
        window = SILERO_ONNX_WINDOW_SAMPLES
        while len(self._stream_pending) >= window:
            chunk = self._stream_pending[:window][None, :]
            self._stream_pending = self._stream_pending[window:]
            model_input = np.concatenate([self._stream_context, chunk], axis=1).astype(np.float32)
            output, self._stream_state = session.run(
                None,
                {"input": model_input, "state": self._stream_state, "sr": sr},
            )
            self._stream_context = model_input[:, -SILERO_ONNX_CONTEXT_SAMPLES:]
            start = self._stream_processed_samples
            self._stream_processed_samples += window
            frames.append((start, self._stream_processed_samples, float(output.reshape(-1)[0])))
        return frames


class TenVAD:
    """Optional TEN VAD backend.

    The TEN Python binding is intentionally loaded lazily so Vox can keep
    Silero as the default without forcing an experimental dependency on every
    installation. TEN operates on fixed 16 kHz frames and returns frame-level
    speech probabilities, which this adapter converts into Silero-style sample
    timestamps for the existing endpointing code.
    """

    def __init__(self, hop_size: int = 160) -> None:
        self._hop_size = hop_size
        self._instances: dict[float, object] = {}
        self._stream_pending = np.empty(0, dtype=np.float32)
        self._stream_processed_samples = 0

    def process_frames(self, audio: NDArray[np.float32]) -> list[tuple[int, int, float]]:
        if audio.size:
            self._stream_pending = np.concatenate(
                [self._stream_pending, np.ascontiguousarray(audio, dtype=np.float32)]
            )

        vad = self._instance_for(TEN_VAD_STREAM_THRESHOLD)
        frames: list[tuple[int, int, float]] = []
        while len(self._stream_pending) >= self._hop_size:
            chunk = self._stream_pending[: self._hop_size]
            self._stream_pending = self._stream_pending[self._hop_size :]
            pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            probability, flag = vad.process(pcm16)
            start = self._stream_processed_samples
            self._stream_processed_samples += self._hop_size
            frames.append((start, self._stream_processed_samples, 1.0 if int(flag) == 1 else float(probability)))
        return frames

    def _instance_for(self, threshold: float) -> object:
        threshold = float(threshold)
        if threshold not in self._instances:
            try:
                from ten_vad import TenVad as TenVadBinding
            except ImportError as exc:
                raise RuntimeError(
                    "TEN VAD backend requires the optional 'ten-vad' package. "
                    "Install it with the vox 'ten-vad' extra or pip install ten-vad."
                ) from exc
            self._instances[threshold] = TenVadBinding(self._hop_size, threshold)
        return self._instances[threshold]


def create_vad_backend(name: str) -> VADBackend:
    normalized = (name or "silero").strip().lower().replace("_", "-")
    if normalized in {"silero", "silero-onnx"}:
        return SileroOnnxVAD()
    if normalized in {"ten", "ten-vad", "tenvad"}:
        return TenVAD()
    raise ValueError(f"unknown VAD backend: {name}")


@dataclass
class VADProcessor:
    config: VADConfig = field(default_factory=VADConfig)
    state: VADState = field(default_factory=VADState)
    buffer: AudioRingBuffer = field(default_factory=lambda: AudioRingBuffer(MAX_BUFFER_SAMPLES))
    _vad_model: VADBackend | None = None
    _utterance_sequence: int = 0
    _stream_samples: int = 0
    _vad_probability_frames: list[tuple[int, int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self._vad_model is None:
            self._vad_model = create_vad_backend(self.config.backend)

    def _duration_ms(self) -> int:
        return len(self.buffer) // MS_PER_SAMPLE

    def _stream_duration_ms(self) -> int:
        return self._stream_samples // MS_PER_SAMPLE

    def _buffer_start_ms(self) -> int:
        return (self._stream_samples - len(self.buffer)) // MS_PER_SAMPLE

    def _buffer_sample_for_timestamp(self, timestamp_ms: int) -> int:
        buffer_start_sample = self._stream_samples - len(self.buffer)
        return (timestamp_ms * MS_PER_SAMPLE) - buffer_start_sample

    def append(self, audio: NDArray[np.float32]) -> tuple[SpeechStarted | SpeechStopped | None, SpeechSegment | None]:
        self._stream_samples += len(audio)
        self.buffer.append(audio)

        audio_window = self.buffer.get_last_n(VAD_WINDOW_SIZE_SAMPLES)

        threshold = (
            self.config.continue_threshold
            if self.state.active or self.state.carryover
            else self.config.start_threshold
        )

        window_start_sample = max(0, self._stream_samples - len(audio_window))
        self._vad_probability_frames.extend(self._vad_model.process_frames(audio))
        self._vad_probability_frames = [
            frame for frame in self._vad_probability_frames if frame[1] > window_start_sample
        ]
        speech_frames = [
            (max(start, window_start_sample), end)
            for start, end, probability in self._vad_probability_frames
            if probability >= threshold
        ]
        raw_timestamps = _frames_to_timestamps(
            speech_frames,
            total_samples=self._stream_samples,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
        )

        speech_ts = None
        if raw_timestamps:
            latest_end = max(ts["end"] for ts in raw_timestamps)
            silence_after_end_ms = (self._stream_samples - latest_end) // MS_PER_SAMPLE
            if silence_after_end_ms < self.config.min_silence_duration_ms:
                speech_ts = {
                    "start": min(ts["start"] for ts in raw_timestamps),
                    "end": latest_end,
                }

        if self.state.audio_start_ms is None:
            if speech_ts is None:
                self.state.carryover = False
                return None, None

            detected_start_ms = None
            if self.state.carryover:
                self.state.audio_start_ms = self._buffer_start_ms()
                self.state.carryover = False
            else:
                detected_start_ms = speech_ts["start"] // MS_PER_SAMPLE
                self.state.audio_start_ms = max(
                    self._buffer_start_ms(),
                    detected_start_ms - max(0, int(self.config.speech_pre_roll_ms)),
                )
            self.state.active = True
            self._utterance_sequence += 1
            self.state.utterance_id = self._utterance_sequence
            seed_start = max(0, self._buffer_sample_for_timestamp(self.state.audio_start_ms))
            seed_end = max(seed_start, len(self.buffer) - len(audio))
            initial_audio = self.buffer.get_slice(seed_start, seed_end)
            return SpeechStarted(
                timestamp_ms=self.state.audio_start_ms,
                utterance_id=self.state.utterance_id,
                audio=initial_audio,
                vad_segment_start_ms=detected_start_ms,
                observed_at_ms=self._stream_duration_ms(),
            ), None

        if speech_ts is None:
            self.state.audio_end_ms = self._stream_duration_ms() - self.config.speech_pad_ms
            segment = self._extract_segment()
            self._clear_buffer()
            if segment.end_ms - segment.start_ms < self.config.min_audio_duration_ms:
                logger.debug("Segment too short (%dms), skipping", segment.end_ms - segment.start_ms)
                return SpeechStopped(
                    timestamp_ms=segment.end_ms,
                    utterance_id=segment.utterance_id,
                ), None
            return SpeechStopped(
                timestamp_ms=segment.end_ms,
                utterance_id=segment.utterance_id,
            ), segment

        if self._stream_duration_ms() - self.state.audio_start_ms >= self.config.max_utterance_ms:
            self.state.audio_end_ms = self._stream_duration_ms() - self.config.speech_pad_ms
            segment = self._extract_segment()
            overflow_audio = self.buffer.get_slice(
                self._buffer_sample_for_timestamp(self.state.audio_end_ms),
                len(self.buffer),
            )
            self._clear_buffer()
            if len(overflow_audio) > 0:
                self.buffer.append(overflow_audio)
                self.state.carryover = True
            if segment.end_ms - segment.start_ms < self.config.min_audio_duration_ms:
                logger.debug(
                    "Segment too short after max_utterance cap (%dms), skipping",
                    segment.end_ms - segment.start_ms,
                )
                return SpeechStopped(
                    timestamp_ms=segment.end_ms,
                    utterance_id=segment.utterance_id,
                ), None
            return SpeechStopped(
                timestamp_ms=segment.end_ms,
                utterance_id=segment.utterance_id,
            ), segment

        return None, None

    def _extract_segment(self) -> SpeechSegment:
        if self.state.audio_start_ms is None or self.state.audio_end_ms is None:
            return SpeechSegment(
                audio=np.array([], dtype=np.float32),
                start_ms=0,
                end_ms=0,
                utterance_id=self.state.utterance_id,
            )

        start_sample = self._buffer_sample_for_timestamp(self.state.audio_start_ms)
        end_sample = self._buffer_sample_for_timestamp(self.state.audio_end_ms)

        return SpeechSegment(
            audio=self.buffer.get_slice(start_sample, end_sample),
            start_ms=self.state.audio_start_ms,
            end_ms=self.state.audio_end_ms,
            utterance_id=self.state.utterance_id,
        )

    def _clear_buffer(self) -> None:
        self.buffer.clear()
        self.state = VADState()

    def reset(self) -> None:
        self._clear_buffer()
        self._stream_samples = 0
