from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

PIPER_SAMPLE_RATE = 22_050
_DEFAULT_VOICE_ID = "default"



def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _extract_sample_rate(config: dict[str, Any]) -> int:
    audio = config.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("sample_rate"), int):
        return audio["sample_rate"]
    if isinstance(config.get("sample_rate"), int):
        return config["sample_rate"]
    return PIPER_SAMPLE_RATE


def _extract_speakers(config: dict[str, Any]) -> list[VoiceInfo]:
    speaker_map = config.get("speaker_id_map")
    if isinstance(speaker_map, dict) and speaker_map:
        voices: list[VoiceInfo] = []
        for name, speaker_id in sorted(speaker_map.items(), key=lambda item: item[1]):
            voices.append(
                VoiceInfo(
                    id=str(name),
                    name=str(name),
                    language="en-us",
                    description=f"Piper speaker {speaker_id}",
                )
            )
        return voices

    voice_name = config.get("voice") or config.get("name") or _DEFAULT_VOICE_ID
    return [
        VoiceInfo(
            id=str(voice_name),
            name=str(voice_name),
            language="en-us",
            description="Default Piper voice",
        )
    ]


def _find_model_files(model_dir: Path) -> tuple[Path, Path]:
    candidates = sorted(model_dir.rglob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No Piper ONNX model found in {model_dir}")

    model_file = candidates[0]
    config_candidates = [
        model_file.with_suffix(".onnx.json"),
        model_file.with_suffix(".json"),
    ]
    for config_file in config_candidates:
        if config_file.is_file():
            return model_file, config_file

    raise FileNotFoundError(f"No Piper config JSON found next to {model_file}")


def _pcm16_to_float32_bytes(pcm: bytes) -> NDArray[np.float32]:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return audio
    return audio / 32768.0


def _speaker_id_for_voice(config: dict[str, Any], voice: str | None) -> int | None:
    speaker_map = config.get("speaker_id_map")
    if not isinstance(speaker_map, dict) or not speaker_map:
        return None

    if voice is None:
        default_voice = config.get("default_voice")
        if isinstance(default_voice, str) and default_voice in speaker_map:
            return int(speaker_map[default_voice])
        first_name = next(iter(sorted(speaker_map.items(), key=lambda item: item[1])))[0]
        return int(speaker_map[first_name])

    if voice in speaker_map:
        return int(speaker_map[voice])

    try:
        return int(voice)
    except ValueError as exc:
        available = ", ".join(sorted(speaker_map))
        raise ValueError(f"Unknown Piper voice '{voice}'. Available voices: {available}") from exc


def _load_piper_voice_class() -> Any:
    try:
        from piper import PiperVoice
        return PiperVoice
    except ImportError:
        _install_piper_runtime()
        try:
            from piper import PiperVoice
            return PiperVoice
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise RuntimeError("Piper requires the piper-tts runtime package") from exc


def _build_synthesis_config(
    config: dict[str, Any],
    voice: str | None,
    speed: float,
) -> Any:
    from piper.config import SynthesisConfig

    speaker_id = _speaker_id_for_voice(config, voice)
    length_scale = max(0.25, min(4.0, 1.0 / max(speed, 0.01)))
    return SynthesisConfig(speaker_id=speaker_id, length_scale=length_scale)


def _ensure_pip_available() -> None:
    if importlib.util.find_spec("pip") is not None:
        return

    result = subprocess.run(
        [sys.executable, "-m", "ensurepip", "--default-pip"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to bootstrap pip for Piper runtime install. "
            f"stderr: {result.stderr.strip()}"
        )


def _install_piper_runtime() -> None:
    _ensure_pip_available()
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "piper-tts>=1.2.0,<2.0.0"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install Piper runtime package. "
            f"stderr: {result.stderr.strip()}"
        )


class PiperAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._voice: Any = None
        self._config: dict[str, Any] = {}
        self._model_id: str = ""
        self._device: str = "cpu"
        self._sample_rate: int = PIPER_SAMPLE_RATE
        self._voices: list[VoiceInfo] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="piper-tts-onnx",
            type=ModelType.TTS,
            architectures=("piper-tts-onnx", "piper"),
            default_sample_rate=PIPER_SAMPLE_RATE,
            supported_formats=(ModelFormat.ONNX,),
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=("en-us",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._voice is not None:
            return

        model_dir = Path(model_path)
        model_file, config_file = _find_model_files(model_dir)
        self._config = _read_json(config_file)
        self._sample_rate = _extract_sample_rate(self._config)
        self._voices = _extract_speakers(self._config)
        self._model_id = kwargs.pop("_source", None) or str(model_file)
        self._device = device

        use_cuda = self._device == "cuda"
        PiperVoice = _load_piper_voice_class()
        logger.info(
            "Loading Piper model: %s (device=%s, sample_rate=%s)",
            self._model_id,
            self._device,
            self._sample_rate,
        )
        self._voice = PiperVoice.load(str(model_file), str(config_file), use_cuda=use_cuda)

    def unload(self) -> None:
        self._voice = None
        self._config = {}
        self._voices = []
        self._model_id = ""
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if self._voice is None:
            raise RuntimeError("Piper model is not loaded — call load() first")

        if reference_audio is not None or reference_text is not None:
            raise ValueError("Piper does not support reference_audio/reference_text")

        if not text or not text.strip():
            return

        syn_config = _build_synthesis_config(self._config, voice, speed)
        audio_chunks = list(self._voice.synthesize(text, syn_config=syn_config))
        if not audio_chunks:
            raise RuntimeError("Piper produced no audio")

        sample_rate = int(getattr(audio_chunks[0], "sample_rate", self._sample_rate))
        audio_arrays: list[NDArray[np.float32]] = []
        for chunk in audio_chunks:
            audio = getattr(chunk, "audio_float_array", None)
            if audio is None:
                audio = getattr(chunk, "_audio_int16_array", None)
                if audio is None:
                    audio_bytes = getattr(chunk, "_audio_int16_bytes", None)
                    if audio_bytes is not None:
                        audio = np.frombuffer(audio_bytes, dtype=np.int16)
            if audio is None:
                continue
            audio_arrays.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        if not audio_arrays:
            raise RuntimeError("Piper produced no audio")

        audio = np.concatenate(audio_arrays)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if audio.size == 0:
            raise RuntimeError("Piper produced no audio")

        chunk_size = sample_rate * 2
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield SynthesizeChunk(
                audio=chunk.tobytes(),
                sample_rate=sample_rate,
                is_final=False,
            )

        yield SynthesizeChunk(audio=b"", sample_rate=sample_rate, is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return list(self._voices)

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 220_000_000
