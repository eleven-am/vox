from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vox.speech_context.types import SpeechContext

TARGET_SAMPLE_RATE = 16_000
MS_PER_SAMPLE = TARGET_SAMPLE_RATE // 1000


def samples_to_ms(sample_count: int, sample_rate: int = TARGET_SAMPLE_RATE) -> int:
    return int(sample_count / sample_rate * 1000)


@dataclass
class StreamSessionConfig:
    language: str = "en"
    sample_rate: int = TARGET_SAMPLE_RATE
    model: str = ""
    partials: bool = False
    partial_window_ms: int = 1500
    partial_stride_ms: int = 700
    include_word_timestamps: bool = False
    temperature: float = 0.0
    speech_context: bool = False


@dataclass
class SpeechStarted:
    timestamp_ms: int = 0
    utterance_id: int = 0


@dataclass
class SpeechStopped:
    timestamp_ms: int = 0
    expects_transcript: bool = True
    utterance_id: int = 0


@dataclass
class StreamTranscript:
    text: str = ""
    is_partial: bool = False
    start_ms: int = 0
    end_ms: int = 0
    audio_duration_ms: int = 0
    processing_duration_ms: int = 0
    model: str = ""
    eou_probability: float | None = None
    segments: list[dict] | None = None
    words: list[dict] | None = None
    entities: list[dict] | None = None
    topics: list[str] | None = None
    utterance_id: int = 0
    speech_context: SpeechContext | None = None
    audio: NDArray[np.float32] | None = None
    _speech_context_task: asyncio.Task[SpeechContext] | None = None


@dataclass
class StreamError:
    message: str = ""


StreamEvent = SpeechStarted | SpeechStopped | StreamTranscript | StreamError
