from __future__ import annotations

import csv
import json
import os
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from numpy.typing import NDArray

from vox.speech_context.audioset import enrich_audioset_classes
from vox.speech_context.reducer import (
    merge_context_chunks,
    offset_context_spans,
    reduce_sound_events,
    summarize_sound_scores,
)
from vox.speech_context.worker import run_analysis_worker

SAMPLE_RATE = 16_000
SCORE_WINDOW_MS = 960.0
SCORE_HOP_MS = 480.0
SPECTROGRAM_WINDOW_MS = 25.0
SPECTROGRAM_HOP_MS = 10.0
COMPACT_CHUNK_SECONDS = 300


class YamnetAnalyzer:
    def __init__(self) -> None:
        assets_value = os.environ.get("VOX_SPEECH_CONTEXT_ASSETS")
        if not assets_value:
            raise RuntimeError("VOX_SPEECH_CONTEXT_ASSETS is not configured")
        assets = Path(assets_value)
        self._model_path = assets / "yamnet.tflite"
        self._class_map_path = assets / "yamnet_class_map.csv"
        self._ontology_path = assets / "audioset_ontology.json"
        if not self._model_path.is_file() or not self._class_map_path.is_file() or not self._ontology_path.is_file():
            raise RuntimeError("YAMNet assets are missing from the isolated runtime")

        with self._class_map_path.open(newline="", encoding="utf-8") as handle:
            classes = [
                {
                    "index": int(row["index"]),
                    "id": row["mid"],
                    "label": row["display_name"],
                }
                for row in csv.DictReader(handle)
            ]
        if len(classes) != 521:
            raise RuntimeError(f"expected 521 YAMNet classes, found {len(classes)}")
        ontology = json.loads(self._ontology_path.read_text(encoding="utf-8"))
        self._classes = enrich_audioset_classes(classes, ontology)

        self._interpreter = Interpreter(model_path=str(self._model_path))

    @staticmethod
    def _read_waveform(audio_path: str) -> NDArray[np.float32]:
        with wave.open(audio_path, "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != SAMPLE_RATE:
                raise ValueError("YAMNet worker requires mono 16 kHz PCM16 WAV input")
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        return (audio.astype(np.float32) / 32768.0).astype(np.float32, copy=False)

    @staticmethod
    def _iter_waveforms(audio_path: str) -> Iterator[NDArray[np.float32]]:
        with wave.open(audio_path, "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != SAMPLE_RATE:
                raise ValueError("YAMNet worker requires mono 16 kHz PCM16 WAV input")
            while pcm := handle.readframes(SAMPLE_RATE * COMPACT_CHUNK_SECONDS):
                audio = np.frombuffer(pcm, dtype="<i2")
                yield (audio.astype(np.float32) / 32768.0).astype(np.float32, copy=False)

    @staticmethod
    def _timed_rows(values: np.ndarray[Any, Any], *, window_ms: float, hop_ms: float) -> list[dict[str, Any]]:
        return [
            {
                "start_ms": round(index * hop_ms, 3),
                "end_ms": round(index * hop_ms + window_ms, 3),
                "values": row.astype(float).tolist(),
            }
            for index, row in enumerate(values)
        ]

    def _invoke(self, waveform: NDArray[np.float32]) -> tuple[np.ndarray[Any, Any], ...]:
        input_detail = self._interpreter.get_input_details()[0]
        self._interpreter.resize_tensor_input(input_detail["index"], [len(waveform)], strict=True)
        self._interpreter.allocate_tensors()
        self._interpreter.set_tensor(input_detail["index"], waveform)
        self._interpreter.invoke()

        output_details = self._interpreter.get_output_details()
        return tuple(self._interpreter.get_tensor(detail["index"]) for detail in output_details)

    def analyze(self, audio_path: str) -> dict[str, Any]:
        waveform = self._read_waveform(audio_path)
        scores, embeddings, spectrogram = self._invoke(waveform)
        return {
            "classes": self._classes,
            "scores": self._timed_rows(scores, window_ms=SCORE_WINDOW_MS, hop_ms=SCORE_HOP_MS),
            "embeddings": self._timed_rows(
                embeddings,
                window_ms=SCORE_WINDOW_MS,
                hop_ms=SCORE_HOP_MS,
            ),
            "log_mel_spectrogram": self._timed_rows(
                spectrogram,
                window_ms=SPECTROGRAM_WINDOW_MS,
                hop_ms=SPECTROGRAM_HOP_MS,
            ),
        }

    def analyze_compact(self, audio_path: str) -> dict[str, Any]:
        chunks: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        offset_samples = 0
        for waveform in self._iter_waveforms(audio_path):
            scores = self._invoke(waveform)[0]
            duration_ms = len(waveform) / SAMPLE_RATE * 1000
            raw_scores = {
                "classes": self._classes,
                "scores": self._timed_rows(
                    scores,
                    window_ms=SCORE_WINDOW_MS,
                    hop_ms=SCORE_HOP_MS,
                ),
            }
            diagnostics.append(
                {
                    "offset_ms": round(offset_samples / SAMPLE_RATE * 1000),
                    **summarize_sound_scores(raw_scores),
                }
            )
            reduced = reduce_sound_events(
                raw_scores,
                duration_ms=duration_ms,
            )
            chunks.append(
                offset_context_spans(
                    reduced,
                    offset_ms=round(offset_samples / SAMPLE_RATE * 1000),
                )
            )
            offset_samples += len(waveform)
        merged = merge_context_chunks(chunks, fields=("sounds",))
        merged["_pre_reduction"] = {"chunks": diagnostics}
        return merged


if __name__ == "__main__":
    analyzer = YamnetAnalyzer()
    raise SystemExit(
        run_analysis_worker(
            {
                "analyze": analyzer.analyze,
                "analyze_compact": analyzer.analyze_compact,
            }
        )
    )
