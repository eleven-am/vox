from __future__ import annotations

import numpy as np

from vox.streaming.codecs import pcm16_to_float32, resample_audio
from vox.streaming.types import TARGET_SAMPLE_RATE

SPEAKING_ECHO_MIN_WINDOW_MS = 120
SPEAKING_ECHO_COMPARE_WINDOW_MS = 320
SPEAKING_ECHO_MAX_DELAY_MS = 900
SPEAKING_ECHO_CORRELATION_THRESHOLD = 0.72
SPEAKING_ECHO_MIN_RMS = 0.002


def _best_recent_correlation(
    mic_audio: np.ndarray,
    output_audio: np.ndarray,
    *,
    max_delay_ms: int,
) -> float:
    """Return sample-accurate normalized correlation against recent output.

    WebRTC resampling and two Opus passes add a codec delay that is not aligned
    to a 20 ms media frame. Sampling only frame-aligned delays can therefore
    miss an otherwise strong echo. The bounded sliding calculation below
    evaluates every possible lag while computing segment norms from rolling
    sums, keeping the hot-path cost independent of the full audio history.
    """
    if mic_audio.size == 0 or output_audio.size < mic_audio.size:
        return 0.0

    mic = mic_audio.astype(np.float32, copy=False)
    mic = mic - float(np.mean(mic))
    mic_norm = float(np.linalg.norm(mic))
    if mic_norm <= 1e-6:
        return 0.0

    window_samples = mic.size
    max_delay_samples = max(0, max_delay_ms * TARGET_SAMPLE_RATE // 1000)
    search_samples = min(
        output_audio.size,
        window_samples + max_delay_samples,
    )
    search = output_audio[-search_samples:].astype(np.float32, copy=False)

    dots = np.correlate(search, mic, mode="valid")
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(search, dtype=np.float64))
    )
    cumulative_squared = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.square(search, dtype=np.float32), dtype=np.float64),
        )
    )
    segment_sums = cumulative[window_samples:] - cumulative[:-window_samples]
    segment_squares = (
        cumulative_squared[window_samples:]
        - cumulative_squared[:-window_samples]
    )
    segment_energy = segment_squares - (
        np.square(segment_sums) / window_samples
    )
    segment_norms = np.sqrt(np.maximum(segment_energy, 0.0))
    valid = segment_norms > 1e-6
    if not np.any(valid):
        return 0.0

    scores = np.abs(dots[valid]) / (mic_norm * segment_norms[valid])
    return float(np.max(scores, initial=0.0))


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
