from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vox.core.device_placement import PlacementTier
from vox.core.types import (
    AdapterInfo,
    SynthesisParameterInfo,
    SynthesizeChunk,
    TranscribeResult,
    VoiceInfo,
)


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

    def placement_tiers(self) -> tuple[PlacementTier, ...]:
        return ()

    def trim(self) -> None:
        """Release non-essential request/cache memory while keeping model weights loaded."""
        return None

    def memory_status(self) -> dict[str, Any]:
        """Return optional backend-specific memory details for diagnostics."""
        return {}

    def prepare_runtime(self) -> None:
        """Install or verify adapter-owned runtime dependencies without loading weights."""
        return None


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
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[SynthesizeChunk]:
        """Stream audio chunks as they are synthesized."""
        ...

    def list_voices(self) -> list[VoiceInfo]:
        """Return built-in voice options. Empty for voice-cloning-only models."""
        return []

    def validate_synthesis_request(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        reference_audio: NDArray[np.float32] | None = None,
        reference_text: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Validate adapter-specific synthesis inputs before a response stream starts."""
        return None

    def synthesis_parameters(self) -> tuple[SynthesisParameterInfo, ...]:
        """Return adapter-specific JSON parameters accepted by synthesize(..., params=...)."""
        return ()


class TurnDetectorAdapter(BaseAdapter):
    @abstractmethod
    def predict(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate: int,
    ) -> float: ...
