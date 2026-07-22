from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import wave
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from vox.speech_context.worker import run_analysis_worker

SAMPLE_RATE = 16_000


def _json_number(value: Any) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if number == math.inf:
        return "Infinity"
    if number == -math.inf:
        return "-Infinity"
    return number


def _load_core_library(core: Path) -> ModuleType:
    # Importing opensmile executes its pandas/audinterface wrapper. The experiment
    # needs only the package's native API and configs, so load that module directly.
    spec = importlib.util.spec_from_file_location("_vox_opensmile_core_lib", core / "lib.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the openSMILE native API")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenSmileAnalyzer:
    def __init__(self) -> None:
        distribution = importlib.metadata.distribution("opensmile")
        self._core = Path(distribution.locate_file("opensmile/core"))
        self._library = _load_core_library(self._core)

    @staticmethod
    def _read_pcm(audio_path: str) -> bytes:
        with wave.open(audio_path, "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != SAMPLE_RATE:
                raise ValueError("prosody worker requires mono 16 kHz PCM16 WAV input")
            return handle.readframes(handle.getnframes())

    def _options(self, level: str) -> dict[str, object]:
        config = self._core / "config"
        return {
            "source": str(config / "shared/standard_external_wave_input.conf.inc"),
            "sampleRate": SAMPLE_RATE,
            "nBits": 16,
            "sink": str(config / "shared/standard_external_data_output_single.conf.inc"),
            "sinkLevel": level,
            "bufferModeRbConf": str(config / "shared/BufferModeRb.conf.inc"),
            "frameModeFunctionalsConf": str(config / "shared/FrameModeFunctionals.conf.inc"),
        }

    def _extract(self, pcm: bytes, level: str) -> dict[str, Any]:
        smile = self._library.OpenSMILE()
        rows: list[dict[str, Any]] = []
        try:
            smile.initialize(
                str(self._core / "config/egemaps/v02/eGeMAPSv02.conf"),
                self._options(level),
                loglevel=0,
            )
            columns = [
                smile.external_sink_get_element_name("extsink", index)
                for index in range(smile.external_sink_get_num_elements("extsink"))
            ]

            def collect(values: np.ndarray[Any, Any], metadata: Any) -> None:
                rows.append({
                    "start_ms": round(float(metadata.time) * 1000, 3),
                    "end_ms": round(float(metadata.time + metadata.lengthSec) * 1000, 3),
                    "values": [_json_number(value) for value in values.reshape(-1)],
                })

            smile.external_sink_set_callback_ex("extsink", collect)
            if not smile.external_audio_source_write_data("extsource", pcm):
                raise RuntimeError("openSMILE rejected the complete audio buffer")
            smile.external_audio_source_set_eoi("extsource")
            smile.run()
        finally:
            smile.free()
        return {"columns": columns, "frames": rows}

    def analyze(self, audio_path: str) -> dict[str, Any]:
        pcm = self._read_pcm(audio_path)
        return {
            "low_level_descriptors": self._extract(pcm, "lld"),
            "functionals": self._extract(pcm, "func"),
        }


if __name__ == "__main__":
    analyzer = OpenSmileAnalyzer()
    raise SystemExit(run_analysis_worker(analyzer.analyze))
