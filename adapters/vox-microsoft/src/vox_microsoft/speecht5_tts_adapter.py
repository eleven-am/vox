from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from vox.core.adapter import TTSAdapter
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

SPEECHT5_TTS_SAMPLE_RATE = 16_000

DEFAULT_SPEAKER_EMBEDDING_DATASET = "Matthijs/cmu-arctic-xvectors"

PRESET_VOICES: list[VoiceInfo] = [
    VoiceInfo(id="default", name="Default (CMU Arctic)", language="en", gender=None),
    VoiceInfo(id="clb", name="CLB (US Female)", language="en", gender="female"),
    VoiceInfo(id="bdl", name="BDL (US Male)", language="en", gender="male"),
    VoiceInfo(id="slt", name="SLT (US Female)", language="en", gender="female"),
    VoiceInfo(id="rms", name="RMS (US Male)", language="en", gender="male"),
]

VOICE_TO_XVECTOR_INDEX: dict[str, int] = {
    "default": 7306,
    "clb": 1,
    "bdl": 0,
    "slt": 3,
    "rms": 2,
}


def _select_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device in ("cuda", "auto"):
        if torch.cuda.is_available():
            return "cuda"
    if device in ("mps", "auto"):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


class SpeechT5TTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._model: SpeechT5ForTextToSpeech | None = None
        self._processor: SpeechT5Processor | None = None
        self._vocoder: SpeechT5HifiGan | None = None
        self._speaker_embeddings: dict[str, torch.Tensor] = {}
        self._loaded = False
        self._model_id: str = ""
        self._device: str = "cpu"

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="speecht5-tts",
            type=ModelType.TTS,
            architectures=("speecht5", "speecht5-tts"),
            default_sample_rate=SPEECHT5_TTS_SAMPLE_RATE,
            supported_formats=(ModelFormat.PYTORCH,),
            supports_streaming=False,
            supports_voice_cloning=False,
            supported_languages=("en",),
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        if self._loaded:
            return

        source = kwargs.pop("_source", None)
        self._model_id = source if source else model_path
        self._device = _select_device(device)

        logger.info("Loading SpeechT5 TTS model: %s (device=%s)", self._model_id, self._device)
        start = time.perf_counter()

        self._processor = SpeechT5Processor.from_pretrained(self._model_id)
        self._model = SpeechT5ForTextToSpeech.from_pretrained(self._model_id).to(self._device)
        self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self._device)

        self._load_speaker_embeddings()

        elapsed = time.perf_counter() - start
        logger.info("SpeechT5 TTS model loaded in %.2fs", elapsed)
        self._loaded = True

    def _load_speaker_embeddings(self) -> None:
        try:
            from datasets import load_dataset

            ds = load_dataset(DEFAULT_SPEAKER_EMBEDDING_DATASET, split="validation")
            for voice_id, idx in VOICE_TO_XVECTOR_INDEX.items():
                if idx < len(ds):
                    xvector = torch.tensor(ds[idx]["xvector"]).unsqueeze(0).to(self._device)
                    self._speaker_embeddings[voice_id] = xvector
            logger.info("Loaded %d speaker embeddings", len(self._speaker_embeddings))
        except Exception as e:
            logger.warning("Failed to load speaker embeddings: %s — using zeros", e)
            self._speaker_embeddings["default"] = torch.zeros(1, 512).to(self._device)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._vocoder = None
        self._speaker_embeddings.clear()
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SpeechT5 TTS adapter unloaded")

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
        if not self._loaded or self._model is None or self._processor is None or self._vocoder is None:
            raise RuntimeError("SpeechT5 TTS model is not loaded — call load() first")

        if not text or not text.strip():
            return

        voice_id = voice or "default"
        speaker_embedding = self._speaker_embeddings.get(
            voice_id,
            self._speaker_embeddings.get("default", torch.zeros(1, 512).to(self._device)),
        )

        inputs = self._processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            speech = self._model.generate_speech(
                inputs["input_ids"],
                speaker_embedding,
                vocoder=self._vocoder,
            )

        audio = speech.cpu().numpy().astype(np.float32)

        chunk_size = SPEECHT5_TTS_SAMPLE_RATE * 2
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield SynthesizeChunk(
                audio=chunk.tobytes(),
                sample_rate=SPEECHT5_TTS_SAMPLE_RATE,
                is_final=False,
            )

        yield SynthesizeChunk(
            audio=b"",
            sample_rate=SPEECHT5_TTS_SAMPLE_RATE,
            is_final=True,
        )

    def list_voices(self) -> list[VoiceInfo]:
        return list(PRESET_VOICES)

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        return 350_000_000
