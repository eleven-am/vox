from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


class TestQwen3ASRAdapterInfo:
    def test_package_import_does_not_require_all_qwen_variants(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            sys.modules.pop("vox_qwen", None)
            module = importlib.import_module("vox_qwen")
            assert module.__all__ == ["Qwen3ASRAdapter", "Qwen3TTSAdapter"]

    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            info = adapter.info()

            assert info.name == "qwen3-asr"
            assert info.type == ModelType.STT
            assert "qwen3-asr" in info.architectures
            assert info.default_sample_rate == 16000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_word_timestamps is True
            assert info.supports_language_detection is True
            assert len(info.supported_languages) >= 30

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            assert adapter.is_loaded is False

    def test_transcribe_raises_when_not_loaded(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            audio = np.zeros(16000, dtype=np.float32)

            with pytest.raises(RuntimeError, match="not loaded"):
                adapter.transcribe(audio)

    def test_transcribe_empty_audio_returns_empty(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()
            adapter._model_id = "test-model"

            result = adapter.transcribe(np.array([], dtype=np.float32))
            assert result.text == ""
            assert result.duration_ms == 0

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None
            assert adapter._processor is None

    def test_estimate_vram_0_6b(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter._model_id = "Qwen/Qwen3-ASR-0.6B"
            assert adapter.estimate_vram_bytes() == 1_500_000_000

    def test_estimate_vram_1_7b(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter._model_id = "Qwen/Qwen3-ASR-1.7B"
            assert adapter.estimate_vram_bytes() == 4_000_000_000

    def test_estimate_vram_uses_source_hint_before_load(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            assert adapter.estimate_vram_bytes(_source="Qwen/Qwen3-ASR-1.7B") == 4_000_000_000

    def test_parse_timestamps(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            raw = "<|0.00|>Hello <|0.50|>world <|1.00|>test"
            words = adapter._parse_timestamps(raw)

            assert len(words) == 3
            assert words[0].word == "Hello"
            assert words[0].start_ms == 0
            assert words[0].end_ms == 500
            assert words[1].word == "world"
            assert words[1].start_ms == 500
            assert words[1].end_ms == 1000
            assert words[2].word == "test"
            assert words[2].start_ms == 1000


class TestQwen3TTSAdapterInfo:
    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            info = adapter.info()

            assert info.name == "qwen3-tts"
            assert info.type == ModelType.TTS
            assert "qwen3-tts" in info.architectures
            assert info.default_sample_rate == 24000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_streaming is True
            assert info.supports_voice_cloning is True

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            assert adapter.is_loaded is False

    def test_list_voices_returns_empty(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            assert adapter.list_voices() == []

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._tokenizer = MagicMock()

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None
            assert adapter._tokenizer is None

    def test_estimate_vram_0_6b(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            assert adapter.estimate_vram_bytes() == 2_500_000_000

    def test_estimate_vram_1_7b(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            assert adapter.estimate_vram_bytes() == 4_000_000_000

    def test_estimate_vram_uses_source_hint_before_load(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            assert adapter.estimate_vram_bytes(_source="Qwen/Qwen3-TTS-12Hz-1.7B-Base") == 4_000_000_000
