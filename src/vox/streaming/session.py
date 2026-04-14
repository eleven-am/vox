from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from vox.streaming.buffer import AudioRingBuffer
from vox.streaming.types import TARGET_SAMPLE_RATE

MAX_SESSION_BUFFER_MS = 30_000
MAX_SESSION_BUFFER_SAMPLES = MAX_SESSION_BUFFER_MS * TARGET_SAMPLE_RATE // 1000


@dataclass
class SpeechSession:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: bool = False
    buffer: AudioRingBuffer = field(default_factory=lambda: AudioRingBuffer(MAX_SESSION_BUFFER_SAMPLES))
    confirmed_words: list[str] = field(default_factory=list)
    last_partial_ms: int = 0

    def start_speech(self) -> None:
        with self.lock:
            self.active = True
            self.buffer.clear()
            self.confirmed_words = []
            self.last_partial_ms = 0

    def stop_speech(self) -> None:
        with self.lock:
            self.active = False

    def is_active(self) -> bool:
        with self.lock:
            return self.active

    def append_audio(self, audio: NDArray[np.float32]) -> None:
        with self.lock:
            if self.active:
                self.buffer.append(audio)

    def get_buffer_audio(self) -> NDArray[np.float32]:
        with self.lock:
            return self.buffer.get_all()

    def get_buffer_tail(self, n_samples: int) -> NDArray[np.float32]:
        with self.lock:
            return self.buffer.get_last_n(n_samples)

    def get_buffer_length(self) -> int:
        with self.lock:
            return len(self.buffer)

    def update_partial(self, new_partial_ms: int, new_words: list[str]) -> None:
        with self.lock:
            self.last_partial_ms = new_partial_ms
            self.confirmed_words = new_words

    def get_partial_state(self) -> tuple[int, list[str]]:
        with self.lock:
            return self.last_partial_ms, list(self.confirmed_words)
