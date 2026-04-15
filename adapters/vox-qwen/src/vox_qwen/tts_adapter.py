from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from importlib.util import find_spec
import importlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)
from vox_qwen.runtime import ensure_runtime

logger = logging.getLogger(__name__)

QWEN_TTS_SAMPLE_RATE = 24_000

SUPPORTED_LANGUAGES = (
    "zh", "en", "ja", "ko", "fr", "de", "ru", "es", "pt", "it",
)

_QWEN_LANGUAGE_LABELS = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
}

_QWEN_LANGUAGE_SPEAKERS = {
    "Chinese": "Vivian",
    "English": "Ryan",
    "Japanese": "Ono_Anna",
    "Korean": "Sohee",
}

_VRAM_ESTIMATES: dict[str, int] = {
    "0.6b": 2_500_000_000,
    "1.7b": 4_000_000_000,
}


def _torch() -> Any:
    return importlib.import_module("torch")


def _select_device(device: str) -> str:
    torch = _torch()
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if device in ("mps", "auto") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> Any:
    torch = _torch()
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _select_device_map(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return "cpu"


def _estimate_vram(model_id: str) -> int:
    lower = model_id.lower()
    for size_key, vram in _VRAM_ESTIMATES.items():
        if size_key in lower:
            return vram
    return _VRAM_ESTIMATES["0.6b"]


def _supports_flash_attention() -> bool:
    return find_spec("flash_attn") is not None


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    key = language.strip().lower()
    if not key:
        return None
    if key in _QWEN_LANGUAGE_LABELS:
        return _QWEN_LANGUAGE_LABELS[key]
    return language


def _normalize_supported_speakers(speakers: Any) -> list[str]:
    if not speakers:
        return []
    normalized: list[str] = []
    for speaker in speakers:
        if isinstance(speaker, str):
            normalized.append(speaker)
            continue
        if isinstance(speaker, dict):
            speaker_id = speaker.get("id") or speaker.get("speaker") or speaker.get("name")
            if speaker_id:
                normalized.append(str(speaker_id))
            continue
        speaker_id = getattr(speaker, "id", None) or getattr(speaker, "speaker", None) or getattr(speaker, "name", None)
        if speaker_id:
            normalized.append(str(speaker_id))
    return normalized


def _load_qwen_tts_model() -> Any:
    ensure_runtime(
        "qwen-tts",
        "qwen-tts",
        "qwen_tts",
        purge_modules=("accelerate", "transformers", "tokenizers", "qwen_tts"),
    )
    try:
        from qwen_tts import Qwen3TTSModel
        return Qwen3TTSModel
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "Qwen3-TTS requires the qwen-tts runtime package; install qwen-tts in the image"
        ) from exc


class Qwen3TTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"
        self._default_voice: str | None = None
        self._supported_speakers: list[str] = []
        self._is_custom_voice_checkpoint = False
        self._subprocess_only = False

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="qwen3-tts",
            type=ModelType.TTS,
            architectures=("qwen3-tts",),
            default_sample_rate=QWEN_TTS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=SUPPORTED_LANGUAGES,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._default_voice = kwargs.pop("default_voice", None)
        self._model_id = source if source else model_path
        self._is_custom_voice_checkpoint = "customvoice" in self._model_id.lower()

        if not self._is_custom_voice_checkpoint:
            raise ValueError(
                "Qwen3-TTS in Vox only exposes CustomVoice checkpoints; "
                "base voice-clone checkpoints require reference_audio/reference_text, "
                "which the Vox API does not provide"
            )

        self._device = _select_device(device)
        try:
            Qwen3TTSModel = _load_qwen_tts_model()
            dtype = _select_dtype(self._device)
            device_map = _select_device_map(self._device)

            logger.info("Loading Qwen3-TTS model: %s (device=%s, dtype=%s)", self._model_id, self._device, dtype)
            start = time.perf_counter()
            model_kwargs: dict[str, Any] = {
                "device_map": device_map,
                "dtype": dtype,
            }
            if self._device == "cuda" and _supports_flash_attention():
                model_kwargs["attn_implementation"] = "flash_attention_2"

            self._model = Qwen3TTSModel.from_pretrained(self._model_id, **model_kwargs)
            self._tokenizer = getattr(self._model, "processor", None)
            get_supported_speakers = getattr(self._model, "get_supported_speakers", None)
            if callable(get_supported_speakers):
                self._supported_speakers = _normalize_supported_speakers(get_supported_speakers())
            else:
                self._supported_speakers = []

            elapsed = time.perf_counter() - start
            logger.info("Qwen3-TTS model loaded in %.2fs", elapsed)
            self._subprocess_only = False
            self._loaded = True
        except Exception as exc:
            logger.warning(
                "Falling back to subprocess-isolated Qwen3-TTS runtime for %s: %s",
                self._model_id,
                exc,
            )
            self._model = None
            self._tokenizer = None
            self._supported_speakers = [self._default_voice] if self._default_voice else []
            self._subprocess_only = True
            self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._subprocess_only = False
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3-TTS adapter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

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
        if not self._loaded:
            raise RuntimeError("Qwen3-TTS model is not loaded — call load() first")

        if not text or not text.strip():
            return

        qwen_language = _normalize_language(language) or "English"
        output: Any

        if reference_audio is not None:
            raise ValueError(
                "Qwen3-TTS CustomVoice checkpoints do not use reference_audio/reference_text; "
                "use the voice speaker parameter instead"
            )

        speaker = voice or self._default_voice
        if speaker is None and self._supported_speakers:
            speaker = _QWEN_LANGUAGE_SPEAKERS.get(qwen_language) or self._supported_speakers[0]
        if speaker is None:
            raise ValueError(
                "Qwen3-TTS CustomVoice checkpoints require a speaker; "
                "provide voice or use a catalog entry with default_voice"
            )
        if self._supported_speakers and speaker not in self._supported_speakers:
            available = ", ".join(self._supported_speakers)
            raise ValueError(
                f"Unknown Qwen3-TTS speaker '{speaker}'. Available speakers: {available}"
            )
        if self._subprocess_only or self._model is None:
            audio, sample_rate = await self._synthesize_via_subprocess(
                text=text,
                speaker=speaker,
                language=qwen_language,
                reference_text=reference_text,
            )
            chunk_size = sample_rate * 2 * 4
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                yield SynthesizeChunk(
                    audio=chunk,
                    sample_rate=sample_rate,
                    is_final=False,
                )
        else:
            output = self._model.generate_custom_voice(
                text=text,
                language=qwen_language,
                speaker=speaker,
                instruct=reference_text,
            )

            if not (isinstance(output, tuple) and len(output) == 2):
                raise RuntimeError("Unexpected Qwen3-TTS output shape")

            wavs, sample_rate = output
            if not wavs:
                raise RuntimeError("Qwen3-TTS produced no audio")

            for wav in wavs:
                audio = np.asarray(wav, dtype=np.float32)
                if audio.size == 0:
                    continue
                chunk_size = sample_rate * 2
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    yield SynthesizeChunk(
                        audio=chunk.tobytes(),
                        sample_rate=sample_rate,
                        is_final=False,
                    )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=sample_rate,
            is_final=True,
        )

    async def _synthesize_via_subprocess(
        self,
        *,
        text: str,
        speaker: str,
        language: str,
        reference_text: str | None,
    ) -> tuple[bytes, int]:
        runtime_dir = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox"))) / "runtime" / "qwen-tts"
        worker = Path(__file__).with_name("qwen_tts_worker.py")
        cmd = [
            sys.executable,
            str(worker),
            "--runtime-dir",
            str(runtime_dir),
            "--model-id",
            self._model_id,
            "--device",
            self._device,
            "--text",
            text,
            "--speaker",
            speaker,
            "--language",
            language,
        ]
        if reference_text:
            cmd.extend(["--instruct", reference_text])

        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Qwen3-TTS subprocess failed: {detail}")

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Qwen3-TTS subprocess returned no payload")

        payload = json.loads(lines[-1])
        audio = base64.b64decode(payload["audio_b64"])
        sample_rate = int(payload["sample_rate"])
        return audio, sample_rate

    def list_voices(self) -> list[VoiceInfo]:
        if not self._supported_speakers:
            return []
        return [
            VoiceInfo(
                id=speaker,
                name=speaker,
                language=None,
                gender=None,
                description=None,
                is_cloned=False,
            )
            for speaker in self._supported_speakers
        ]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        model_id = kwargs.get("_source") or kwargs.get("model_id") or self._model_id
        return _estimate_vram(str(model_id))
