from __future__ import annotations

import asyncio
import logging
import os
import platform
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession, get_available_providers

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# Maps the first 3 characters of a Kokoro voice ID (e.g. "af_") to the
# single-letter language code expected by kokoro-onnx's create_stream().
VOICE_PREFIX_TO_LANG: dict[str, str] = {
    "af_": "a",  # American English female
    "am_": "a",  # American English male
    "bf_": "b",  # British English female
    "bm_": "b",  # British English male
    "jf_": "j",  # Japanese female
    "jm_": "j",  # Japanese male
    "zf_": "z",  # Chinese female
    "zm_": "z",  # Chinese male
    "ef_": "e",  # Spanish female
    "em_": "e",  # Spanish male
    "ff_": "f",  # French female
    "fm_": "f",  # French male
    "hf_": "h",  # Hindi female
    "hm_": "h",  # Hindi male
    "if_": "i",  # Italian female
    "im_": "i",  # Italian male
    "pf_": "p",  # Portuguese female
    "pm_": "p",  # Portuguese male
}

# Human-readable language labels keyed by the Kokoro single-letter lang code.
LANG_CODE_TO_LANGUAGE: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "z": "zh",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "p": "pt",
}

# Gender hint derived from the second character of the voice prefix.
_GENDER_MAP: dict[str, str] = {"f": "female", "m": "male"}


def _get_onnx_providers(device: str) -> list[tuple[str, dict]]:
    """Choose ONNX execution providers based on *device* and platform."""
    available = get_available_providers()
    system = platform.system()
    machine = platform.machine()

    logger.info("Available ONNX providers: %s", available)

    if device == "cpu":
        return [("CPUExecutionProvider", {})]

    providers: list[tuple[str, dict]] = []

    if system == "Darwin" and machine == "arm64" and "CoreMLExecutionProvider" in available:
        providers.append(("CoreMLExecutionProvider", {}))

    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {}))

    providers.append(("CPUExecutionProvider", {}))

    logger.info("Using ONNX providers: %s", providers)
    return providers


def _voice_lang(voice_id: str) -> str:
    """Return the Kokoro single-letter language code for a voice ID."""
    prefix = voice_id[:3] if len(voice_id) >= 3 else ""
    return VOICE_PREFIX_TO_LANG.get(prefix, "a")


def _voice_info(voice_id: str) -> VoiceInfo:
    """Build a VoiceInfo from a Kokoro voice ID like ``af_heart``."""
    prefix = voice_id[:3] if len(voice_id) >= 3 else ""
    lang_code = VOICE_PREFIX_TO_LANG.get(prefix, "a")
    language = LANG_CODE_TO_LANGUAGE.get(lang_code)
    gender = _GENDER_MAP.get(prefix[1:2]) if len(prefix) >= 2 else None
    return VoiceInfo(
        id=voice_id,
        name=voice_id,
        language=language,
        gender=gender,
    )


class KokoroAdapter(TTSAdapter):
    """Vox TTS adapter backed by Kokoro-82M via ``kokoro-onnx``."""

    def __init__(self) -> None:
        self._kokoro: Kokoro | None = None
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    # TTSAdapter interface
    # ------------------------------------------------------------------

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="kokoro",
            type=ModelType.TTS,
            architectures=["kokoro"],
            default_sample_rate=SAMPLE_RATE,
            supported_formats=[ModelFormat.ONNX],
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages=list(LANG_CODE_TO_LANGUAGE.values()),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        """Load the Kokoro model from *model_path*.

        *model_path* should be a directory containing ``model.onnx`` and
        ``voices.bin``.  *device* is mapped to ONNX execution providers
        (``"cpu"``, ``"cuda"``, or ``"auto"``).
        """
        model_dir = Path(model_path)
        model_file = model_dir / "model.onnx"
        voices_file = model_dir / "voices.bin"

        if not model_file.exists():
            raise FileNotFoundError(f"model.onnx not found in {model_dir}")
        if not voices_file.exists():
            raise FileNotFoundError(f"voices.bin not found in {model_dir}")

        self._device = device
        providers = _get_onnx_providers(device)

        logger.info("Loading Kokoro model from %s (device=%s)", model_dir, device)
        session = InferenceSession(str(model_file), providers=providers)
        self._kokoro = Kokoro.from_session(session, str(voices_file))
        logger.info("Kokoro model loaded")

    def unload(self) -> None:
        if self._kokoro is not None:
            self._kokoro = None
            logger.info("Kokoro model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._kokoro is not None

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
        reference_audio: np.ndarray | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        if self._kokoro is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not text or not text.strip():
            return

        voice_id = voice or "af_heart"
        speed = max(0.5, min(speed, 2.0))

        # Determine the language code Kokoro expects.
        if language is not None:
            # Allow callers to pass the single-letter code directly or a
            # BCP-47-style tag (e.g. "en-us").  We try a reverse lookup first.
            lang = language
            for code, bcp in LANG_CODE_TO_LANGUAGE.items():
                if bcp == language:
                    lang = code
                    break
        else:
            lang = _voice_lang(voice_id)

        async for audio_chunk, _token in self._kokoro.create_stream(
            text, voice_id, lang=lang, speed=speed
        ):
            yield SynthesizeChunk(
                audio=audio_chunk.astype(np.float32).tobytes(),
                sample_rate=SAMPLE_RATE,
                is_final=False,
            )

        # Send a final empty chunk to signal completion.
        yield SynthesizeChunk(
            audio=b"",
            sample_rate=SAMPLE_RATE,
            is_final=True,
        )

    def list_voices(self) -> list[VoiceInfo]:
        if self._kokoro is None:
            return []
        return [_voice_info(v) for v in self._kokoro.get_voices()]

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        # Kokoro-82M ONNX is ~330 MB in memory.
        return 330 * 1024 * 1024 if self._kokoro is not None else 0
