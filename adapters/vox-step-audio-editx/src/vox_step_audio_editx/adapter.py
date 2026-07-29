from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.errors import ModelLoadError
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesisParameterInfo,
    SynthesizeChunk,
    VoiceInfo,
)
from vox.core.worker_host import WorkerHost
from vox.operations.errors import InvalidConfigError
from vox_step_audio_editx.runtime import ensure_runtime, worker_env

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
DEFAULT_STARTUP_TIMEOUT_SECONDS = 1800.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 1800.0
STARTUP_TIMEOUT_ENV = "VOX_STEP_AUDIO_EDITX_STARTUP_TIMEOUT_S"
REQUEST_TIMEOUT_ENV = "VOX_STEP_AUDIO_EDITX_REQUEST_TIMEOUT_S"


def _timeout(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number of seconds, got {raw!r}") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {raw!r}")
    return value


class StepAudioEditXAdapter(TTSAdapter):
    def __init__(self) -> None:
        self._host: WorkerHost | None = None
        self._model_path: Path | None = None
        self._lock = threading.RLock()

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="step-audio-editx-tts-vllm",
            type=ModelType.TTS,
            architectures=("step-audio-editx", "step-audio-editx-tts"),
            default_sample_rate=SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_voice_cloning=True,
            supported_languages=("en", "zh", "ja", "ko"),
            max_input_chars=1_200,
        )

    def prepare_runtime(self) -> None:
        ensure_runtime()

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        with self._lock:
            if self._host is not None:
                if self._host.alive:
                    return
                self._host.close()
                self._host = None
            if device not in ("cuda", "auto"):
                raise ModelLoadError("Step-Audio-EditX requires a CUDA device")
            path = Path(model_path)
            required = (
                path / "config.json",
                path / "CosyVoice-300M-25Hz",
                path / "audio_tokenizer" / "speech_tokenizer_v1.onnx",
            )
            missing = [str(item) for item in required if not item.exists()]
            if missing:
                raise ModelLoadError(f"Step-Audio-EditX model artifacts are incomplete: {missing}")
            runtime = ensure_runtime()
            host = WorkerHost(
                [sys.executable, "-m", "vox_step_audio_editx.worker", "--model-path", str(path)],
                env=worker_env(runtime, "cuda"),
                name="step-audio-editx",
                startup_timeout=_timeout(STARTUP_TIMEOUT_ENV, DEFAULT_STARTUP_TIMEOUT_SECONDS),
            )
            self._model_path = path
            self._host = host

    def unload(self) -> None:
        with self._lock:
            host = self._host
            self._host = None
            self._model_path = None
            if host is not None:
                host.close()
        logger.info("Step-Audio-EditX adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        host = self._host
        return host is not None and host.alive

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if reference_audio is None or np.asarray(reference_audio).size == 0:
            raise InvalidConfigError(
                "Step-Audio-EditX requires reference_audio; create or select a stored Vox voice"
            )
        if not reference_text or not reference_text.strip():
            raise InvalidConfigError(
                "Step-Audio-EditX requires the reference transcript for voice cloning"
            )
        if language and language not in ("auto", *self.info().supported_languages):
            raise InvalidConfigError(f"Step-Audio-EditX does not support language={language!r}")

    def synthesis_parameters(self) -> tuple[SynthesisParameterInfo, ...]:
        return (
            SynthesisParameterInfo(
                name="temperature",
                type="number",
                default=0.7,
                min_value=0.0,
                max_value=2.0,
                description="vLLM sampling temperature.",
            ),
            SynthesisParameterInfo(
                name="seed",
                type="integer",
                default=None,
                min_value=0,
                max_value=2**32 - 1,
                description="vLLM request seed.",
            ),
        )

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        host = self._host
        if host is None or not host.alive:
            raise RuntimeError("Step-Audio-EditX model is not loaded")
        if not text or not text.strip():
            return
        if speed != 1.0:
            raise InvalidConfigError("Step-Audio-EditX does not support speed values other than 1.0")
        self.validate_synthesis_request(
            voice=voice,
            language=language,
            reference_audio=reference_audio,
            reference_text=reference_text,
            params=params,
        )
        values = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
        options = dict(params or {})
        temperature = float(options.get("temperature", 0.7))
        seed = options.get("seed")
        if not 0.0 <= temperature <= 2.0:
            raise InvalidConfigError("Step-Audio-EditX temperature must be between 0.0 and 2.0")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1):
            raise InvalidConfigError("Step-Audio-EditX seed must be an integer between 0 and 4294967295")
        with tempfile.TemporaryDirectory(prefix="vox-step-audio-editx-") as temp_dir:
            reference_path = Path(temp_dir) / "reference.wav"
            output_path = Path(temp_dir) / "output.wav"
            sf.write(reference_path, values, SAMPLE_RATE, subtype="PCM_16")
            response = host.request(
                {
                    "op": "clone",
                    "text": text,
                    "reference_path": str(reference_path),
                    "reference_text": reference_text,
                    "output_path": str(output_path),
                    "temperature": temperature,
                    "seed": seed,
                },
                timeout=_timeout(REQUEST_TIMEOUT_ENV, DEFAULT_REQUEST_TIMEOUT_SECONDS),
            )
            audio, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
        output = np.asarray(audio, dtype=np.float32).reshape(-1)
        if int(response["samples"]) != output.size:
            raise RuntimeError("Step-Audio-EditX worker returned inconsistent audio metadata")
        chunk_samples = int(sample_rate) * 2
        for offset in range(0, output.size, chunk_samples):
            yield SynthesizeChunk(
                audio=output[offset : offset + chunk_samples].tobytes(),
                sample_rate=int(sample_rate),
                is_final=False,
            )
        yield SynthesizeChunk(audio=b"", sample_rate=int(sample_rate), is_final=True)

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="reference",
                name="Reference audio",
                description="Pass a stored voice with reference audio and transcript.",
                is_cloned=True,
            )
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 10_000_000_000
