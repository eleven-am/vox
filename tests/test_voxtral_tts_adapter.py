from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import SynthesizeChunk


class FakeOmniBackend:
    def __init__(self, chunks: list[SynthesizeChunk], *, raise_on_generate: Exception | None = None) -> None:
        self._chunks = chunks
        self._raise_on_generate = raise_on_generate
        self.close_called = False

    async def generate(self, text: str, voice: str) -> AsyncIterator[SynthesizeChunk]:
        if self._raise_on_generate is not None:
            raise self._raise_on_generate
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        self.close_called = True


def _make_audio_bytes(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


def _make_chunks() -> list[SynthesizeChunk]:
    return [
        SynthesizeChunk(audio=_make_audio_bytes([0.1, 0.2]), sample_rate=24000, is_final=False),
        SynthesizeChunk(audio=_make_audio_bytes([0.3, 0.4]), sample_rate=24000, is_final=False),
        SynthesizeChunk(audio=b"", sample_rate=24000, is_final=True),
    ]


def _make_adapter_with_backend(backend: FakeOmniBackend):
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        from vox_voxtral.tts_adapter import VoxtralTTSAdapter

        adapter = VoxtralTTSAdapter()
        adapter._loaded = True
        adapter._backend = backend
        return adapter


async def _collect(adapter, **kwargs) -> list[SynthesizeChunk]:
    chunks: list[SynthesizeChunk] = []
    async for chunk in adapter.synthesize("hello world", **kwargs):
        chunks.append(chunk)
    return chunks


class TestFakeOmniBackendChunkOrdering:
    def test_chunks_arrive_in_order(self):
        expected = _make_chunks()
        backend = FakeOmniBackend(expected)
        adapter = _make_adapter_with_backend(backend)

        chunks = asyncio.run(_collect(adapter))

        assert len(chunks) == 3
        assert chunks[0].audio == expected[0].audio
        assert chunks[1].audio == expected[1].audio
        assert chunks[2].audio == expected[2].audio

    def test_final_sentinel_is_last_and_empty(self):
        backend = FakeOmniBackend(_make_chunks())
        adapter = _make_adapter_with_backend(backend)

        chunks = asyncio.run(_collect(adapter))

        assert chunks[-1].is_final is True
        assert chunks[-1].audio == b""

    def test_non_final_chunks_before_sentinel(self):
        backend = FakeOmniBackend(_make_chunks())
        adapter = _make_adapter_with_backend(backend)

        chunks = asyncio.run(_collect(adapter))

        for chunk in chunks[:-1]:
            assert chunk.is_final is False

    def test_sample_rate_consistent(self):
        backend = FakeOmniBackend(_make_chunks())
        adapter = _make_adapter_with_backend(backend)

        chunks = asyncio.run(_collect(adapter))

        for chunk in chunks:
            assert chunk.sample_rate == 24000


class TestFakeOmniBackendErrorPropagation:
    def test_generate_error_propagates(self):
        backend = FakeOmniBackend([], raise_on_generate=RuntimeError("vllm exploded"))
        adapter = _make_adapter_with_backend(backend)

        with pytest.raises(RuntimeError, match="vllm exploded"):
            asyncio.run(_collect(adapter))

    def test_synthesize_raises_when_not_loaded(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            with pytest.raises(RuntimeError, match="not loaded"):
                asyncio.run(_collect(adapter))

    def test_synthesize_raises_for_reference_audio(self):
        backend = FakeOmniBackend(_make_chunks())
        adapter = _make_adapter_with_backend(backend)

        with pytest.raises(NotImplementedError, match="reference-audio cloning"):
            asyncio.run(
                _collect(adapter, reference_audio=np.zeros(24000, dtype=np.float32))
            )

    def test_synthesize_empty_text_yields_nothing(self):
        backend = FakeOmniBackend(_make_chunks())
        adapter = _make_adapter_with_backend(backend)

        chunks = asyncio.run(_collect(adapter))

        async def synthesize_empty():
            result = []
            async for chunk in adapter.synthesize("   "):
                result.append(chunk)
            return result

        empty_chunks = asyncio.run(synthesize_empty())
        assert empty_chunks == []


class TestCloseOnUnload:
    def test_close_called_on_unload(self):
        backend = FakeOmniBackend(_make_chunks())
        adapter = _make_adapter_with_backend(backend)

        with patch.dict("sys.modules", {"torch": MagicMock()}):
            adapter.unload()

        assert backend.close_called is True

    def test_backend_cleared_after_unload(self):
        backend = FakeOmniBackend(_make_chunks())
        adapter = _make_adapter_with_backend(backend)

        with patch.dict("sys.modules", {"torch": MagicMock()}):
            adapter.unload()

        assert adapter._backend is None
        assert adapter.is_loaded is False
