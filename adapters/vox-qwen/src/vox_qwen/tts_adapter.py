from __future__ import annotations

import asyncio
import base64
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from importlib.util import find_spec
from pathlib import Path
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
QWEN_TTS_RUNTIME_PACKAGES = (
    "sox",
    "einops",
)

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


def _resolve_supported_speaker(speaker: str, supported_speakers: list[str]) -> str:
    if not supported_speakers:
        return speaker

    requested = speaker.strip()
    if requested in supported_speakers:
        return requested

    lowered = requested.casefold()
    for candidate in supported_speakers:
        if candidate.casefold() == lowered:
            return candidate

    available = ", ".join(supported_speakers)
    raise ValueError(
        f"Unknown Qwen3-TTS speaker '{speaker}'. Available speakers: {available}"
    )


def _detect_mode(model_id: str, *, override: str | None = None) -> str:
    if override:
        normalized = override.strip().lower()
        if normalized in ("clone", "custom"):
            return normalized
        raise ValueError(f"Unknown qwen3-tts mode {override!r}; expected 'clone' or 'custom'")
    lower = model_id.lower()
    if "customvoice" in lower:
        return "custom"
    if "-base" in lower:
        return "clone"
    return "custom"


async def _stream_model_output(output: Any, default_sample_rate: int) -> AsyncIterator[SynthesizeChunk]:
    if not (isinstance(output, tuple) and len(output) == 2):
        raise RuntimeError("Unexpected Qwen3-TTS output shape")

    wavs, sample_rate = output
    if sample_rate is None or int(sample_rate) <= 0:
        sample_rate = default_sample_rate
    sample_rate = int(sample_rate)
    if not wavs:
        raise RuntimeError("Qwen3-TTS produced no audio")

    if not isinstance(wavs, (list, tuple)):
        wavs = [wavs]

    for wav in wavs:
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
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

    yield SynthesizeChunk(audio=b"", sample_rate=sample_rate, is_final=True)


def _load_qwen_tts_model() -> Any:
    ensure_runtime(
        "qwen-tts",
        "qwen-tts",
        "qwen_tts",
        purge_modules=("accelerate", "transformers", "tokenizers", "qwen_tts"),
        no_deps=False,
        extra_packages=QWEN_TTS_RUNTIME_PACKAGES,
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
        self._model_ref: str = ""
        self._device: str = "cpu"
        self._default_voice: str | None = None
        self._supported_speakers: list[str] = []
        self._mode: str = "custom"
        self._subprocess_only = False

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="qwen3-tts-torch",
            type=ModelType.TTS,
            architectures=("qwen3-tts-torch", "qwen3-tts"),
            default_sample_rate=QWEN_TTS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=True,
            supports_voice_cloning=True,
            supported_languages=SUPPORTED_LANGUAGES,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._default_voice = kwargs.pop("default_voice", None)
        mode_override = kwargs.pop("mode", None)
        self._model_id = source if source else model_path
        path = Path(model_path)
        self._model_ref = str(path) if path.exists() else self._model_id
        self._mode = _detect_mode(self._model_id, override=mode_override)

        self._device = device
        try:
            Qwen3TTSModel = _load_qwen_tts_model()
            dtype = _select_dtype(self._device)
            device_map = _select_device_map(self._device)

            logger.info(
                "Loading Qwen3-TTS model: %s (device=%s, dtype=%s)",
                self._model_ref,
                self._device,
                dtype,
            )
            start = time.perf_counter()
            model_kwargs: dict[str, Any] = {
                "device_map": device_map,
                "dtype": dtype,
            }
            if self._device == "cuda" and _supports_flash_attention():
                model_kwargs["attn_implementation"] = "flash_attention_2"

            self._model = Qwen3TTSModel.from_pretrained(self._model_ref, **model_kwargs)
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
                self._model_ref,
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
        self._model_ref = ""
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

        if self._mode == "clone":
            async for chunk in self._synthesize_clone(
                text=text,
                language=qwen_language,
                reference_audio=reference_audio,
                reference_text=reference_text,
            ):
                yield chunk
            return


        if reference_audio is not None:
            raise ValueError(
                "Qwen3-TTS CustomVoice checkpoints do not use reference_audio; "
                "load a Base checkpoint (e.g. 'qwen3-tts:0.6b-clone') for zero-shot cloning, "
                "or pass a speaker name via the `voice` parameter"
            )

        speaker = voice or self._default_voice
        if speaker is None and self._supported_speakers:
            speaker = _QWEN_LANGUAGE_SPEAKERS.get(qwen_language) or self._supported_speakers[0]
        if speaker is None:
            raise ValueError(
                "Qwen3-TTS CustomVoice checkpoints require a speaker; "
                "provide voice or use a catalog entry with default_voice"
            )
        if self._supported_speakers:
            speaker = _resolve_supported_speaker(speaker, self._supported_speakers)
        if self._subprocess_only or self._model is None:
            async for chunk in self._stream_subprocess(
                mode="custom",
                text=text,
                language=qwen_language,
                speaker=speaker,
                reference_text=reference_text,
            ):
                yield chunk
            return

        output = self._model.generate_custom_voice(
            text=text,
            language=qwen_language,
            speaker=speaker,
            instruct=reference_text,
        )
        async for chunk in _stream_model_output(output, QWEN_TTS_SAMPLE_RATE):
            yield chunk

    async def _synthesize_clone(
        self,
        *,
        text: str,
        language: str,
        reference_audio: NDArray[np.float32] | None,
        reference_text: str | None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if reference_audio is None or reference_audio.size == 0:
            raise ValueError(
                "Qwen3-TTS Base (clone) checkpoints require reference_audio; "
                "upload a reference sample via POST /v1/audio/voices and pass its voice id, "
                "or load a CustomVoice checkpoint (e.g. 'qwen3-tts:0.6b') for speaker-based synthesis"
            )

        if self._subprocess_only or self._model is None:
            async for chunk in self._stream_subprocess(
                mode="clone",
                text=text,
                language=language,
                reference_audio=reference_audio,
                reference_text=reference_text,
            ):
                yield chunk
            return

        output = self._model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=(np.asarray(reference_audio, dtype=np.float32), QWEN_TTS_SAMPLE_RATE),
            ref_text=reference_text,
        )
        async for chunk in _stream_model_output(output, QWEN_TTS_SAMPLE_RATE):
            yield chunk

    async def _stream_subprocess(
        self,
        *,
        mode: str,
        text: str,
        language: str,
        speaker: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        audio, sample_rate = await self._synthesize_via_subprocess(
            mode=mode,
            text=text,
            language=language,
            speaker=speaker,
            reference_audio=reference_audio,
            reference_text=reference_text,
        )
        chunk_size = sample_rate * 2 * 4
        if not audio:
            yield SynthesizeChunk(audio=b"", sample_rate=sample_rate, is_final=True)
            return
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield SynthesizeChunk(
                audio=chunk,
                sample_rate=sample_rate,
                is_final=False,
            )
        yield SynthesizeChunk(audio=b"", sample_rate=sample_rate, is_final=True)

    async def _synthesize_via_subprocess(
        self,
        *,
        mode: str,
        text: str,
        language: str,
        speaker: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
    ) -> tuple[bytes, int]:
        runtime_dir = Path(os.environ.get("VOX_HOME", str(Path.home() / ".vox"))) / "runtime" / "qwen-tts"
        worker = Path(__file__).with_name("qwen_tts_worker.py")
        cmd = [
            sys.executable,
            str(worker),
            "--runtime-dir",
            str(runtime_dir),
            "--model-id",
            self._model_ref or self._model_id,
            "--device",
            self._device,
            "--mode",
            mode,
            "--text",
            text,
            "--language",
            language,
        ]

        ref_tmp: Path | None = None
        try:
            if mode == "clone":
                if reference_audio is None:
                    raise ValueError("Clone mode requires reference_audio")
                import soundfile as sf

                with tempfile.NamedTemporaryFile(
                    prefix="qwen3-ref-", suffix=".wav", delete=False,
                ) as fd:
                    ref_tmp = Path(fd.name)
                sf.write(
                    str(ref_tmp),
                    np.asarray(reference_audio, dtype=np.float32),
                    QWEN_TTS_SAMPLE_RATE,
                )
                cmd.extend(["--ref-audio-path", str(ref_tmp)])
                if reference_text:
                    cmd.extend(["--ref-text", reference_text])
            else:
                if speaker:
                    cmd.extend(["--speaker", speaker])
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
        finally:
            if ref_tmp is not None:
                try:
                    ref_tmp.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to remove temporary reference audio %s", ref_tmp)

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
