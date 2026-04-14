from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from vox.streaming.buffer import AudioRingBuffer
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    MS_PER_SAMPLE,
    SpeechStarted,
    SpeechStopped,
)

logger = logging.getLogger(__name__)

VAD_WINDOW_SIZE_SAMPLES = 1000 * MS_PER_SAMPLE
MAX_BUFFER_SAMPLES = (15_000 + 3_000 + 1_000) * MS_PER_SAMPLE


@dataclass
class VADConfig:
    start_threshold: float = 0.6
    continue_threshold: float = 0.4
    min_silence_duration_ms: int = 1000
    speech_pad_ms: int = 100
    min_speech_duration_ms: int = 250
    min_audio_duration_ms: int = 500
    max_utterance_ms: int = 15_000


@dataclass
class VADState:
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    active: bool = False


@dataclass
class SpeechSegment:
    audio: NDArray[np.float32]
    start_ms: int
    end_ms: int


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


@dataclass
class VADProcessor:
    config: VADConfig = field(default_factory=VADConfig)
    state: VADState = field(default_factory=VADState)
    buffer: AudioRingBuffer = field(default_factory=lambda: AudioRingBuffer(MAX_BUFFER_SAMPLES))
    _vad_model: SileroVAD = field(default_factory=SileroVAD)

    def _duration_ms(self) -> int:
        return len(self.buffer) // MS_PER_SAMPLE

    def append(self, audio: NDArray[np.float32]) -> tuple[SpeechStarted | SpeechStopped | None, SpeechSegment | None]:
        self.buffer.append(audio)

        audio_window = self.buffer.get_last_n(VAD_WINDOW_SIZE_SAMPLES)
        window_duration_ms = len(audio_window) // MS_PER_SAMPLE

        threshold = self.config.start_threshold if not self.state.active else self.config.continue_threshold

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
                return None, None

            self.state.audio_start_ms = (
                self._duration_ms() - window_duration_ms + (speech_ts["start"] // MS_PER_SAMPLE)
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
            if segment.end_ms - segment.start_ms < self.config.min_audio_duration_ms:
                logger.debug("Segment too short after max_utterance cap (%dms), skipping", segment.end_ms - segment.start_ms)
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
