from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.core.types import AdapterInfo, TranscribeResult, SynthesizeChunk, VoiceInfo


class BaseAdapter(ABC):
    """Shared interface for all model adapters (STT and TTS)."""

    @abstractmethod
    def info(self) -> AdapterInfo: ...

    @abstractmethod
    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        """Load model weights from a local path onto the specified device."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all GPU/CPU memory held by the model."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    def estimate_vram_bytes(self, **kwargs: Any) -> int:
        """Return estimated VRAM/RAM usage in bytes. Used by the scheduler for device placement."""
        return 0


class STTAdapter(BaseAdapter):
    """Base class every STT model adapter must implement."""

    @abstractmethod
    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> TranscribeResult:
        """Synchronous full-utterance transcription."""
        ...

    def transcribe_stream(
        self,
        audio_iter: AsyncIterator[NDArray[np.float32]],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscribeResult]:
        """Optional: streaming partial transcripts."""
        raise NotImplementedError(f"{self.info().name} does not support streaming STT")

    def detect_language(self, audio: NDArray[np.float32]) -> str:
        """Optional: language identification from audio."""
        raise NotImplementedError(f"{self.info().name} does not support language detection")


class TTSAdapter(BaseAdapter):
    """Base class every TTS model adapter must implement."""

    @abstractmethod
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
        """Stream audio chunks as they are synthesized."""
        ...

    def list_voices(self) -> list[VoiceInfo]:
        """Return built-in voice options. Empty for voice-cloning-only models."""
        return []
