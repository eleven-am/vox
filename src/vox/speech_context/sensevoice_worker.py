from __future__ import annotations

import os
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx
from numpy.typing import NDArray

from vox.speech_context.reducer import reduce_speaker_context
from vox.speech_context.worker import run_analysis_worker

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE * 5 // 2
HOP_SAMPLES = SAMPLE_RATE


class SenseVoiceAnalyzer:
    def __init__(self) -> None:
        assets_value = os.environ.get("VOX_SPEECH_CONTEXT_ASSETS")
        if not assets_value:
            raise RuntimeError("VOX_SPEECH_CONTEXT_ASSETS is not configured")
        assets = Path(assets_value)
        model = assets / "model.int8.onnx"
        tokens = assets / "tokens.txt"
        if not model.is_file() or not tokens.is_file():
            raise RuntimeError("SenseVoice assets are missing from the isolated runtime")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(tokens),
            num_threads=4,
            sample_rate=SAMPLE_RATE,
            provider="cpu",
            language="auto",
            use_itn=True,
        )

    @staticmethod
    def _read_waveform(audio_path: str) -> NDArray[np.float32]:
        with wave.open(audio_path, "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != SAMPLE_RATE:
                raise ValueError("SenseVoice worker requires mono 16 kHz PCM16 WAV input")
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        return (audio.astype(np.float32) / 32768.0).astype(np.float32, copy=False)

    @staticmethod
    def _windows(
        waveform: NDArray[np.float32],
    ) -> Iterator[tuple[int, NDArray[np.float32]]]:
        start = 0
        while start < len(waveform):
            end = min(len(waveform), start + WINDOW_SAMPLES)
            yield start, waveform[start:end]
            if end == len(waveform):
                break
            start += HOP_SAMPLES

    def analyze(self, audio_path: str) -> dict[str, Any]:
        waveform = self._read_waveform(audio_path)
        windows = []
        for start, samples in self._windows(waveform):
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            self._recognizer.decode_stream(stream)
            result = stream.result
            windows.append(
                {
                    "start_ms": round(start / SAMPLE_RATE * 1000),
                    "end_ms": round((start + len(samples)) / SAMPLE_RATE * 1000),
                    "language": result.lang,
                    "emotion": result.emotion,
                    "event": result.event,
                    "text": result.text,
                }
            )
        return {"windows": windows}

    def analyze_compact(self, audio_path: str) -> dict[str, Any]:
        raw = self.analyze(audio_path)
        reduced = reduce_speaker_context(raw)
        reduced["_pre_reduction"] = raw
        return reduced


if __name__ == "__main__":
    analyzer = SenseVoiceAnalyzer()
    raise SystemExit(
        run_analysis_worker(
            {
                "analyze": analyzer.analyze,
                "analyze_compact": analyzer.analyze_compact,
            }
        )
    )
