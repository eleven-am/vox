from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import numpy as np

from vox.core.adapter import STTAdapter, TTSAdapter
from vox.core.errors import ModelNotFoundError
from vox.core.types import (
    AdapterInfo,
    ModelFormat,
    ModelType,
    SynthesizeChunk,
    TranscribeResult,
    VoiceInfo,
)


class FakeSTTAdapter(STTAdapter):
    def __init__(self, text: str = "hello world", language: str = "en") -> None:
        self._text = text
        self._language = language
        self.last_kwargs: dict | None = None

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-stt",
            type=ModelType.STT,
            architectures=("fake",),
            default_sample_rate=16_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def transcribe(self, audio: Any, **kwargs: Any) -> TranscribeResult:
        self.last_kwargs = kwargs
        return TranscribeResult(
            text=self._text,
            language=self._language,
            duration_ms=1000,
        )


class FakeTTSAdapter(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts",
            type=ModelType.TTS,
            architectures=("fake",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="default", name="Default", language="en")]

    async def synthesize(self, text: str, **kwargs: Any):
        yield SynthesizeChunk(
            audio=np.zeros(24_000, dtype=np.float32).tobytes(),
            sample_rate=24_000,
            is_final=True,
        )


class FakeCloneableTTSAdapter(TTSAdapter):
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake-tts-cloneable",
            type=ModelType.TTS,
            architectures=("fake",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
            supports_voice_cloning=True,
        )

    def load(self, *a: Any, **k: Any) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="default", name="Default", language="en")]

    async def synthesize(self, text: str, **kwargs: Any):
        yield SynthesizeChunk(
            audio=np.zeros(64, dtype=np.float32).tobytes(),
            sample_rate=24_000,
            is_final=False,
        )
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class FakeScheduler:
    def __init__(self, adapter: STTAdapter | TTSAdapter | None = None) -> None:
        self._adapters: dict[str, STTAdapter | TTSAdapter] = {}
        if adapter is not None:
            self._adapters["__default__"] = adapter
        self._default_adapter = adapter

    def register(self, name: str, adapter: STTAdapter | TTSAdapter) -> None:
        self._adapters[name] = adapter

    @asynccontextmanager
    async def acquire(self, name: str):
        adapter = self._adapters.get(name, self._default_adapter)
        if adapter is None:
            raise ModelNotFoundError(name)
        yield adapter

    def list_loaded(self) -> list:
        return []

    async def unload(self, name: str) -> bool:
        return True

    async def preload(self, name: str) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


DummyScheduler = FakeScheduler
MockScheduler = FakeScheduler
