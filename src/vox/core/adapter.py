from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from vox.core.adapter_execution import AdapterExecutionLane
from vox.core.device_placement import PlacementTier
from vox.core.types import (
    AdapterInfo,
    SynthesisParameterInfo,
    SynthesizeChunk,
    TranscribeResult,
    VoiceInfo,
)

T = TypeVar("T")


class BaseAdapter(ABC):
    """Shared interface for all model adapters (STT and TTS)."""

    def __new__(cls, *args: Any, **kwargs: Any):
        instance = super().__new__(cls)
        instance._vox_execution_lane = AdapterExecutionLane()
        return instance

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

    @property
    def supports_trim(self) -> bool:
        return type(self).trim is not BaseAdapter.trim

    @property
    def physical_work_count(self) -> int:
        return self._vox_execution_lane.pending_count

    def close_execution_lane(self) -> None:
        self._vox_execution_lane.close()

    async def wait_execution_idle(self, *, timeout: float | None = None) -> None:
        await self._vox_execution_lane.wait_idle(timeout=timeout)

    def run_exclusive(self, operation: Callable[[], None]) -> None:
        self._get_execution_lane().run_sync(operation)

    async def execute_sync(self, operation: Callable[[], T]) -> T:
        return await self._get_execution_lane().run(operation)

    def _get_execution_lane(self) -> AdapterExecutionLane:
        return self._vox_execution_lane

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

    async def iterate_synthesis(
        self,
        iterator: AsyncIterator[SynthesizeChunk],
    ) -> AsyncIterator[SynthesizeChunk]:
        async for item in self._get_execution_lane().iterate(iterator):
            yield item

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
