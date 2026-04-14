from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


class TestSpeechT5STTAdapterInfo:
    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            adapter = SpeechT5STTAdapter()
            info = adapter.info()

            assert info.name == "speecht5-stt"
            assert info.type == ModelType.STT
            assert "speecht5" in info.architectures
            assert info.default_sample_rate == 16000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_streaming is False
            assert info.supported_languages == ("en",)

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            adapter = SpeechT5STTAdapter()
            assert adapter.is_loaded is False

    def test_transcribe_raises_when_not_loaded(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            adapter = SpeechT5STTAdapter()
            audio = np.zeros(16000, dtype=np.float32)

            with pytest.raises(RuntimeError, match="not loaded"):
                adapter.transcribe(audio)

    def test_transcribe_empty_audio_returns_empty(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            adapter = SpeechT5STTAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()
            adapter._model_id = "test-model"

            result = adapter.transcribe(np.array([], dtype=np.float32))
            assert result.text == ""
            assert result.duration_ms == 0

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            adapter = SpeechT5STTAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None

    def test_estimate_vram(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            adapter = SpeechT5STTAdapter()
            assert adapter.estimate_vram_bytes() == 320_000_000


class TestSpeechT5TTSAdapterInfo:
    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_tts_adapter import SpeechT5TTSAdapter

            adapter = SpeechT5TTSAdapter()
            info = adapter.info()

            assert info.name == "speecht5-tts"
            assert info.type == ModelType.TTS
            assert "speecht5" in info.architectures
            assert info.default_sample_rate == 16000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supported_languages == ("en",)

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_tts_adapter import SpeechT5TTSAdapter

            adapter = SpeechT5TTSAdapter()
            assert adapter.is_loaded is False

    def test_list_voices_returns_presets(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_tts_adapter import SpeechT5TTSAdapter

            adapter = SpeechT5TTSAdapter()
            voices = adapter.list_voices()
            assert len(voices) > 0
            voice_ids = [v.id for v in voices]
            assert "default" in voice_ids
            assert "clb" in voice_ids

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_tts_adapter import SpeechT5TTSAdapter

            adapter = SpeechT5TTSAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()
            adapter._vocoder = MagicMock()
            adapter._speaker_embeddings = {"default": MagicMock()}

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None
            assert adapter._vocoder is None
            assert len(adapter._speaker_embeddings) == 0

    def test_estimate_vram(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_tts_adapter import SpeechT5TTSAdapter

            adapter = SpeechT5TTSAdapter()
            assert adapter.estimate_vram_bytes() == 350_000_000


class TestVibeVoiceTTSAdapterInfo:
    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            info = adapter.info()

            assert info.name == "vibevoice-tts"
            assert info.type == ModelType.TTS
            assert "vibevoice" in info.architectures
            assert info.default_sample_rate == 24000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_streaming is True

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            assert adapter.is_loaded is False

    def test_list_voices_returns_empty(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            assert adapter.list_voices() == []

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._tokenizer = MagicMock()

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None
            assert adapter._tokenizer is None

    def test_estimate_vram_0_5b(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            adapter._model_id = "microsoft/VibeVoice-Realtime-0.5B"
            assert adapter.estimate_vram_bytes() == 2_000_000_000

    def test_estimate_vram_1_5b(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            adapter._model_id = "microsoft/VibeVoice-1.5B"
            assert adapter.estimate_vram_bytes() == 6_000_000_000

    def test_estimate_vram_uses_source_hint_before_load(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            assert adapter.estimate_vram_bytes(_source="microsoft/VibeVoice-1.5B") == 6_000_000_000
