from __future__ import annotations

import numpy as np

from vox.streaming.codecs import pcm16_to_float32, resample_audio
from vox.streaming.types import TARGET_SAMPLE_RATE

SPEAKING_ECHO_MIN_WINDOW_MS = 120
SPEAKING_ECHO_COMPARE_WINDOW_MS = 320
SPEAKING_ECHO_MAX_DELAY_MS = 900
SPEAKING_ECHO_SEARCH_STEP_MS = 20
SPEAKING_ECHO_CORRELATION_THRESHOLD = 0.68
SPEAKING_ECHO_MIN_RMS = 0.002


def _normalise_for_correlation(audio: np.ndarray) -> np.ndarray | None:
    if audio.size == 0:
        return None
    centred = audio.astype(np.float32, copy=False) - float(np.mean(audio))
    norm = float(np.linalg.norm(centred))
    if norm <= 1e-6:
        return None
    return centred / norm


def _best_recent_correlation(
    mic_audio: np.ndarray,
    output_audio: np.ndarray,
    *,
    max_delay_ms: int,
    step_ms: int,
) -> float:
    mic = _normalise_for_correlation(mic_audio)
    if mic is None:
        return 0.0

    window_samples = mic.size
    max_delay_samples = max(0, max_delay_ms * TARGET_SAMPLE_RATE // 1000)
    step_samples = max(1, step_ms * TARGET_SAMPLE_RATE // 1000)
    best = 0.0
    for delay in range(0, max_delay_samples + 1, step_samples):
        end = output_audio.size - delay
        start = end - window_samples
        if start < 0:
            continue
        segment = _normalise_for_correlation(output_audio[start:end])
        if segment is None:
            continue
        best = max(best, abs(float(np.dot(mic, segment))))
    return best


class ConversationAudioHistory:
    """Bounded mic/output audio history used for barge-in decisions.

    ConversationSession owns turn semantics. This helper owns only recent audio
    memory and acoustic echo checks so ring buffer behavior has one owner.
    """

    def __init__(
        self,
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
        mic_window_ms: int = 2000,
        output_window_ms: int = 3000,
    ) -> None:
        self._sample_rate = sample_rate
        self._mic_max_samples = sample_rate * mic_window_ms // 1000
        self._output_max_samples = sample_rate * output_window_ms // 1000
        self._mic: np.ndarray = np.empty(0, dtype=np.float32)
        self._output: np.ndarray = np.empty(0, dtype=np.float32)

    @property
    def mic_size(self) -> int:
        return int(self._mic.size)

    @property
    def mic_max_samples(self) -> int:
        return self._mic_max_samples

    @property
    def output_size(self) -> int:
        return int(self._output.size)

    def replace_mic(self, audio: np.ndarray) -> None:
        self._mic = self._coerce_audio(audio)[-self._mic_max_samples :]

    def replace_output(self, audio: np.ndarray) -> None:
        self._output = self._coerce_audio(audio)[-self._output_max_samples :]

    def append_mic(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        self._mic = self._append_bounded(self._mic, audio, self._mic_max_samples)

    def remember_output_pcm16(self, encoded_audio: bytes, sample_rate: int) -> None:
        if sample_rate <= 0 or not encoded_audio:
            return
        audio = pcm16_to_float32(encoded_audio)
        if audio.size == 0:
            return
        if sample_rate != self._sample_rate:
            audio = resample_audio(audio, sample_rate, self._sample_rate)
        self._output = self._append_bounded(self._output, audio, self._output_max_samples)

    def clear(self) -> None:
        self._mic = np.empty(0, dtype=np.float32)
        self._output = np.empty(0, dtype=np.float32)

    def mic_tail_for_duration_ms(self, duration_ms: int) -> np.ndarray | None:
        if duration_ms <= 0 or self._mic.size == 0:
            return None
        tail_samples = min(
            self._mic.size,
            max(1, duration_ms * self._sample_rate // 1000),
        )
        return self._mic[-tail_samples:]

    def looks_like_current_output_echo(self) -> bool:
        min_samples = SPEAKING_ECHO_MIN_WINDOW_MS * self._sample_rate // 1000
        if self._mic.size < min_samples or self._output.size < min_samples:
            return False

        window_samples = min(
            self._mic.size,
            self._output.size,
            SPEAKING_ECHO_COMPARE_WINDOW_MS * self._sample_rate // 1000,
        )
        if window_samples < min_samples:
            return False

        mic = self._mic[-window_samples:]
        mic_rms = float(np.sqrt(np.mean(mic * mic))) if mic.size else 0.0
        if mic_rms < SPEAKING_ECHO_MIN_RMS:
            return False

        best = _best_recent_correlation(
            mic,
            self._output,
            max_delay_ms=SPEAKING_ECHO_MAX_DELAY_MS,
            step_ms=SPEAKING_ECHO_SEARCH_STEP_MS,
        )
        return best >= SPEAKING_ECHO_CORRELATION_THRESHOLD

    @staticmethod
    def _coerce_audio(audio: np.ndarray) -> np.ndarray:
        return np.asarray(audio, dtype=np.float32)

    def _append_bounded(self, current: np.ndarray, audio: np.ndarray, max_samples: int) -> np.ndarray:
        next_audio = np.concatenate([current, self._coerce_audio(audio)])
        if next_audio.size > max_samples:
            return next_audio[-max_samples:]
        return next_audio
