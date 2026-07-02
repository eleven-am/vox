"""Tests for the one-shot HTTP /v1/audio/speech chunking wrapper around split_for_tts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.operations.synthesis import _split_for_adapter
from vox.operations.tts_chunking import (
    effective_tts_text_cap,
    split_text_for_tts_adapter,
    tts_adapter_text_cap,
)


class _CappedTTS(TTSAdapter):
    def __init__(self, max_input_chars: int) -> None:
        self._max = max_input_chars
        self.calls: list[str] = []

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="capped-tts",
            type=ModelType.TTS,
            architectures=("fake",),
            default_sample_rate=24000,
            supported_formats=(ModelFormat.ONNX,),
            max_input_chars=self._max,
        )

    def load(self, model_path: str, device: str, **kwargs: Any) -> None:
        pass

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default", language="en")]

    async def synthesize(self, text: str, **kwargs: Any) -> AsyncIterator[SynthesizeChunk]:
        self.calls.append(text)
        audio = np.zeros(1200, dtype=np.float32)
        yield SynthesizeChunk(audio=audio.tobytes(), sample_rate=24000, is_final=True)


class TestSplitForAdapter:
    def test_adapter_text_cap_normalizes_invalid_caps(self):
        assert tts_adapter_text_cap(_CappedTTS(max_input_chars=200)) == 200
        assert tts_adapter_text_cap(_CappedTTS(max_input_chars=0)) == 0
        assert tts_adapter_text_cap(_CappedTTS(max_input_chars=-5)) == 0

    def test_effective_text_cap_prefers_explicit_override(self):
        adapter = _CappedTTS(max_input_chars=200)

        assert effective_tts_text_cap(adapter, None) == 200
        assert effective_tts_text_cap(adapter, 50) == 50
        assert effective_tts_text_cap(adapter, 0) == 0
        assert effective_tts_text_cap(adapter, -5) == 0

    def test_short_text_stays_single_chunk(self):
        adapter = _CappedTTS(max_input_chars=200)
        chunks = _split_for_adapter("hello world", adapter)
        assert chunks == ["hello world"]

    def test_long_text_respects_adapter_cap(self):
        adapter = _CappedTTS(max_input_chars=50)
        text = "First sentence here. Second sentence here. Third sentence follows. Fourth wraps it up."
        chunks = _split_for_adapter(text, adapter)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_text_exactly_at_cap_is_single_chunk(self):
        adapter = _CappedTTS(max_input_chars=20)
        text = "x" * 20
        chunks = _split_for_adapter(text, adapter)
        assert chunks == [text]

    def test_zero_cap_bypasses_chunking(self):
        adapter = _CappedTTS(max_input_chars=0)
        text = "this is a pretty long string that has no business being chunked."
        chunks = _split_for_adapter(text, adapter)
        assert chunks == [text]

    def test_empty_text_returns_empty_list(self):
        adapter = _CappedTTS(max_input_chars=100)
        assert _split_for_adapter("", adapter) == []

    def test_whitespace_only_text_returns_empty_list_when_uncapped(self):
        adapter = _CappedTTS(max_input_chars=0)
        assert _split_for_adapter("   \n\t ", adapter) == []

    def test_sentence_boundaries_are_preferred(self):
        adapter = _CappedTTS(max_input_chars=25)
        text = "Short one. Short two. Short three. Short four."
        chunks = _split_for_adapter(text, adapter)

        for chunk in chunks:
            assert "." in chunk

    def test_negative_cap_behaves_like_zero(self):
        adapter = MagicMock(spec=TTSAdapter)
        adapter.info.return_value = AdapterInfo(
            name="broken",
            type=ModelType.TTS,
            architectures=("fake",),
            default_sample_rate=24000,
            supported_formats=(ModelFormat.ONNX,),
            max_input_chars=-5,
        )
        text = "abc"
        chunks = _split_for_adapter(text, adapter)
        assert chunks == [text]

    def test_explicit_override_controls_shared_adapter_splitter(self):
        adapter = _CappedTTS(max_input_chars=200)
        text = "First sentence here. Second sentence here. Third sentence follows."

        assert split_text_for_tts_adapter(text, adapter, override_chars=0) == [text]
        chunks = split_text_for_tts_adapter(text, adapter, override_chars=25)

        assert len(chunks) > 1
        assert all(len(chunk) <= 25 for chunk in chunks)
