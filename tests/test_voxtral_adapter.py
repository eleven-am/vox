from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


class TestVoxtralSTTAdapterInfo:
    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            info = adapter.info()

            assert info.name == "voxtral-stt"
            assert info.type == ModelType.STT
            assert "voxtral" in info.architectures
            assert info.default_sample_rate == 16000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_language_detection is True

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            assert adapter.is_loaded is False

    def test_transcribe_raises_when_not_loaded(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            audio = np.zeros(16000, dtype=np.float32)

            with pytest.raises(RuntimeError, match="not loaded"):
                adapter.transcribe(audio)

    def test_transcribe_empty_audio_returns_empty(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()
            adapter._model_id = "test-model"

            result = adapter.transcribe(np.array([], dtype=np.float32))
            assert result.text == ""
            assert result.duration_ms == 0

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None
            assert adapter._processor is None

    def test_estimate_vram_3b(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            adapter._model_id = "mistralai/Voxtral-Mini-3B-2507"
            assert adapter.estimate_vram_bytes() > 0

    def test_estimate_vram_24b(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            adapter._model_id = "mistralai/Voxtral-Small-24B-2507"
            assert adapter.estimate_vram_bytes() > 9_500_000_000


class TestVoxtralTTSAdapterInfo:
    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            info = adapter.info()

            assert info.name == "voxtral-tts"
            assert info.type == ModelType.TTS
            assert "voxtral-tts" in info.architectures
            assert info.default_sample_rate == 24000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_voice_cloning is True

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            assert adapter.is_loaded is False

    def test_list_voices_returns_presets(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            voices = adapter.list_voices()
            assert len(voices) > 0
            voice_ids = [v.id for v in voices]
            assert "jessica" in voice_ids

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None
            assert adapter._processor is None

    def test_estimate_vram(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            assert adapter.estimate_vram_bytes() == 16_000_000_000
