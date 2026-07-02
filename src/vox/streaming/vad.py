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


@dataclass
class SpeechSegment:
    audio: NDArray[np.float32]
    start_ms: int
    end_ms: int


class VADBackend(Protocol):
    def get_speech_timestamps(
        self,
        audio: NDArray[np.float32],
        threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        min_speech_duration_ms: int = 250,
    ) -> list[dict[str, int]]: ...


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

    timestamps: list[dict[str, int]] = []
    current_start, current_end = speech_frames[0]
    for start, end in speech_frames[1:]:
        if start - current_end <= min_silence_samples:
            current_end = end
            continue
        timestamps.append({
            "start": max(0, current_start - pad_samples),
            "end": min(total_samples, current_end + pad_samples),
        })
        current_start, current_end = start, end

    timestamps.append({
        "start": max(0, current_start - pad_samples),
        "end": min(total_samples, current_end + pad_samples),
    })
    return [ts for ts in timestamps if ts["end"] - ts["start"] >= min_speech_samples]


class SileroVAD:
    _instance: SileroVAD | None = None
    _model = None
    _get_speech_timestamps = None
    _lock = threading.Lock()

    def __new__(cls) -> SileroVAD:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def _ensure_model(self) -> None:
        if SileroVAD._model is not None:
            return
        with SileroVAD._lock:
            if SileroVAD._model is None:
                import torch

                logger.info("Loading Silero VAD model from torch.hub")
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad:master",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                )
                SileroVAD._model = model
                SileroVAD._get_speech_timestamps = utils[0]
                warmup_audio = torch.zeros(TARGET_SAMPLE_RATE)
                SileroVAD._get_speech_timestamps(
                    warmup_audio, SileroVAD._model, sampling_rate=TARGET_SAMPLE_RATE
                )
                logger.info("Silero VAD model loaded")

    def get_speech_timestamps(
        self,
        audio: NDArray[np.float32],
        threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        min_speech_duration_ms: int = 250,
    ) -> list[dict[str, int]]:
        import torch

        self._ensure_model()
        audio_tensor = torch.from_numpy(audio)
        return SileroVAD._get_speech_timestamps(
            audio_tensor,
            SileroVAD._model,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            min_speech_duration_ms=min_speech_duration_ms,
            sampling_rate=TARGET_SAMPLE_RATE,
        )


SILERO_ONNX_WINDOW_SAMPLES = 256


class SileroOnnxVAD:
    """Default VAD backend: the Silero model run on onnxruntime (no torch).

    Uses the bundled ``silero_vad.onnx`` — same model as the torch backend, so
    detection is equivalent — but avoids the ~2GB torch dependency and the
    runtime torch.hub download. The onnxruntime session is a process-wide
    singleton; each ``get_speech_timestamps`` call analyzes the passed window
    from a fresh recurrent state, matching the torch utility's per-call
    behavior.
    """

    _session = None
    _lock = threading.Lock()

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

    def get_speech_timestamps(
        self,
        audio: NDArray[np.float32],
        threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        min_speech_duration_ms: int = 250,
    ) -> list[dict[str, int]]:
        if audio.size == 0:
            return []

        session = self._ensure_session()
        window = SILERO_ONNX_WINDOW_SAMPLES
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(TARGET_SAMPLE_RATE, dtype=np.int64)

        speech_frames: list[tuple[int, int]] = []
        for start in range(0, len(audio) - window + 1, window):
            frame = audio[start:start + window]
            output, state = session.run(None, {"input": frame[None, :], "state": state, "sr": sr})
            if float(output.reshape(-1)[0]) >= threshold:
                speech_frames.append((start, start + window))

        return _frames_to_timestamps(
            speech_frames,
            total_samples=len(audio),
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            min_speech_duration_ms=min_speech_duration_ms,
        )


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

    def get_speech_timestamps(
        self,
        audio: NDArray[np.float32],
        threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        min_speech_duration_ms: int = 250,
    ) -> list[dict[str, int]]:
        if audio.size == 0:
            return []

        vad = self._instance_for(threshold)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        speech_frames: list[tuple[int, int]] = []

        for start in range(0, len(pcm16), self._hop_size):
            end = min(start + self._hop_size, len(pcm16))
            frame = pcm16[start:end]
            if len(frame) < self._hop_size:
                frame = np.pad(frame, (0, self._hop_size - len(frame)))

            probability, flag = vad.process(frame)
            is_speech = int(flag) == 1 or float(probability) >= threshold
            if is_speech:
                speech_frames.append((start, end))

        return _frames_to_timestamps(
            speech_frames,
            total_samples=len(pcm16),
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            min_speech_duration_ms=min_speech_duration_ms,
        )


def create_vad_backend(name: str) -> VADBackend:
    normalized = (name or "silero").strip().lower().replace("_", "-")
    if normalized in {"silero", "silero-onnx"}:
        return SileroOnnxVAD()
    if normalized in {"silero-torch", "silero-pytorch"}:
        return SileroVAD()
    if normalized in {"ten", "ten-vad", "tenvad"}:
        return TenVAD()
    raise ValueError(f"unknown VAD backend: {name}")


@dataclass
class VADProcessor:
    config: VADConfig = field(default_factory=VADConfig)
    state: VADState = field(default_factory=VADState)
    buffer: AudioRingBuffer = field(default_factory=lambda: AudioRingBuffer(MAX_BUFFER_SAMPLES))
    _vad_model: VADBackend | None = None

    def __post_init__(self) -> None:
        if self._vad_model is None:
            self._vad_model = create_vad_backend(self.config.backend)

    def _duration_ms(self) -> int:
        return len(self.buffer) // MS_PER_SAMPLE

    def append(self, audio: NDArray[np.float32]) -> tuple[SpeechStarted | SpeechStopped | None, SpeechSegment | None]:
        self.buffer.append(audio)

        audio_window = self.buffer.get_last_n(VAD_WINDOW_SIZE_SAMPLES)
        window_duration_ms = len(audio_window) // MS_PER_SAMPLE

        threshold = (
            self.config.continue_threshold
            if self.state.active or self.state.carryover
            else self.config.start_threshold
        )

        raw_timestamps = self._vad_model.get_speech_timestamps(
            audio_window,
            threshold=threshold,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
        )

        speech_ts = None
        if raw_timestamps:
            latest_end = max(ts["end"] for ts in raw_timestamps)
            silence_after_end_ms = window_duration_ms - (latest_end // MS_PER_SAMPLE)
            if silence_after_end_ms < self.config.min_silence_duration_ms:
                speech_ts = {
                    "start": min(ts["start"] for ts in raw_timestamps),
                    "end": latest_end,
                }

        if self.state.audio_start_ms is None:
            if speech_ts is None:
                self.state.carryover = False
                return None, None

            if self.state.carryover:
                self.state.audio_start_ms = 0
                self.state.carryover = False
            else:
                detected_start_ms = (
                    self._duration_ms() - window_duration_ms + (speech_ts["start"] // MS_PER_SAMPLE)
                )
                self.state.audio_start_ms = max(
                    0,
                    detected_start_ms - max(0, int(self.config.speech_pre_roll_ms)),
                )
            self.state.active = True
            return SpeechStarted(timestamp_ms=self.state.audio_start_ms), None

        if speech_ts is None:
            self.state.audio_end_ms = self._duration_ms() - self.config.speech_pad_ms
            segment = self._extract_segment()
            self._clear_buffer()
            if segment.end_ms - segment.start_ms < self.config.min_audio_duration_ms:
                logger.debug("Segment too short (%dms), skipping", segment.end_ms - segment.start_ms)
                return SpeechStopped(timestamp_ms=segment.end_ms), None
            return SpeechStopped(timestamp_ms=self.state.audio_end_ms), segment

        if self._duration_ms() >= self.config.max_utterance_ms:
            self.state.audio_end_ms = self._duration_ms() - self.config.speech_pad_ms
            segment = self._extract_segment()
            overflow_audio = self.buffer.get_slice(
                self.state.audio_end_ms * MS_PER_SAMPLE, len(self.buffer)
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
                return SpeechStopped(timestamp_ms=segment.end_ms), None
            return SpeechStopped(timestamp_ms=self.state.audio_end_ms), segment

        return None, None

    def _extract_segment(self) -> SpeechSegment:
        if self.state.audio_start_ms is None or self.state.audio_end_ms is None:
            return SpeechSegment(audio=np.array([], dtype=np.float32), start_ms=0, end_ms=0)

        start_sample = self.state.audio_start_ms * MS_PER_SAMPLE
        end_sample = self.state.audio_end_ms * MS_PER_SAMPLE

        return SpeechSegment(
            audio=self.buffer.get_slice(start_sample, end_sample),
            start_ms=self.state.audio_start_ms,
            end_ms=self.state.audio_end_ms,
        )

    def _clear_buffer(self) -> None:
        self.buffer.clear()
        self.state = VADState()

    def reset(self) -> None:
        self._clear_buffer()
