from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


def _mock_torch(cuda_available: bool = True, mps_available: bool = False):
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = cuda_available
    torch_mock.backends.mps.is_available.return_value = mps_available
    torch_mock.bfloat16 = object()
    torch_mock.float32 = object()
    torch_mock.inference_mode.return_value.__enter__.return_value = None
    torch_mock.inference_mode.return_value.__exit__.return_value = False
    torch_mock.tensor.side_effect = lambda value: MagicMock()
    return torch_mock


def _mock_qwen_asr_module():
    module = MagicMock()
    model_cls = MagicMock()
    forced_aligner_cls = MagicMock()
    module.Qwen3ASRModel = model_cls
    module.Qwen3ForcedAligner = forced_aligner_cls
    return module, model_cls, forced_aligner_cls


def _mock_qwen_tts_module():
    module = MagicMock()
    model_cls = MagicMock()
    module.Qwen3TTSModel = model_cls
    return module, model_cls


class TestQwen3ASRAdapterInfo:
    def test_package_import_does_not_require_all_qwen_variants(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            sys.modules.pop("vox_qwen", None)
            module = importlib.import_module("vox_qwen")
            assert module.__all__ == ["Qwen3ASRAdapter", "Qwen3TTSAdapter"]

    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
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

    def test_load_uses_official_qwen_runtime(self):
        torch_mock = _mock_torch()
        qwen_asr_module, model_cls, forced_aligner_cls = _mock_qwen_asr_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": qwen_asr_module, "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            with patch("vox_qwen.asr_adapter._supports_flash_attention", return_value=True):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-ASR-0.6B")

            model_cls.from_pretrained.assert_called_once()
            kwargs = model_cls.from_pretrained.call_args.kwargs
            assert kwargs["device_map"] == "cuda:0"
            assert kwargs["dtype"] is torch_mock.bfloat16
            assert kwargs["attn_implementation"] == "flash_attention_2"
            assert adapter._model is model_instance
            assert adapter._processor is model_instance.processor
            assert forced_aligner_cls.from_pretrained.call_count == 0

    def test_load_skips_flash_attention_when_runtime_missing(self):
        torch_mock = _mock_torch()
        qwen_asr_module, model_cls, _ = _mock_qwen_asr_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": qwen_asr_module, "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            with patch("vox_qwen.asr_adapter._supports_flash_attention", return_value=False):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-ASR-0.6B")

            kwargs = model_cls.from_pretrained.call_args.kwargs
            assert "attn_implementation" not in kwargs

    def test_transcribe_uses_forced_aligner_for_word_timestamps(self):
        torch_mock = _mock_torch()
        qwen_asr_module, model_cls, forced_aligner_cls = _mock_qwen_asr_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.transcribe.return_value = [SimpleNamespace(text="hello world", language="English")]
        model_cls.from_pretrained.return_value = model_instance

        aligner_instance = MagicMock()
        aligner_instance.align.return_value = [[
            SimpleNamespace(text="hello", start_time=0.0, end_time=0.5, confidence=0.99),
            SimpleNamespace(text="world", start_time=0.5, end_time=1.0, confidence=0.98),
        ]]
        forced_aligner_cls.from_pretrained.return_value = aligner_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": qwen_asr_module, "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            with patch("vox_qwen.asr_adapter._supports_flash_attention", return_value=True):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-ASR-0.6B")
            result = adapter.transcribe(np.ones(16000, dtype=np.float32), word_timestamps=True)

            assert result.text == "hello world"
            assert result.language == "en"
            assert len(result.segments) == 1
            assert len(result.segments[0].words) == 2
            assert result.segments[0].words[0].word == "hello"
            assert result.segments[0].words[0].start_ms == 0
            assert result.segments[0].words[1].word == "world"
            assert forced_aligner_cls.from_pretrained.called
            aligner_instance.align.assert_called_once()

    def test_detect_language_uses_official_runtime(self):
        torch_mock = _mock_torch()
        qwen_asr_module, model_cls, _ = _mock_qwen_asr_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.transcribe.return_value = [SimpleNamespace(language="English", text="")]
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": qwen_asr_module, "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            with patch("vox_qwen.asr_adapter._supports_flash_attention", return_value=True):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-ASR-0.6B")
            language = adapter.detect_language(np.ones(16000, dtype=np.float32))

            assert language == "en"

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            assert adapter.is_loaded is False

    def test_transcribe_raises_when_not_loaded(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            audio = np.zeros(16000, dtype=np.float32)

            with pytest.raises(RuntimeError, match="not loaded"):
                adapter.transcribe(audio)

    def test_transcribe_empty_audio_returns_empty(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._model_id = "test-model"

            result = adapter.transcribe(np.array([], dtype=np.float32))
            assert result.text == ""
            assert result.duration_ms == 0

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()
            adapter._aligner = MagicMock()

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None
            assert adapter._processor is None
            assert adapter._aligner is None

    def test_estimate_vram_0_6b(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter._model_id = "Qwen/Qwen3-ASR-0.6B"
            assert adapter.estimate_vram_bytes() == 1_500_000_000

    def test_estimate_vram_1_7b(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter._model_id = "Qwen/Qwen3-ASR-1.7B"
            assert adapter.estimate_vram_bytes() == 4_000_000_000

    def test_estimate_vram_uses_source_hint_before_load(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            assert adapter.estimate_vram_bytes(_source="Qwen/Qwen3-ASR-1.7B") == 4_000_000_000

    def test_parse_timestamps(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
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
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            info = adapter.info()

            assert info.name == "qwen3-tts"
            assert info.type == ModelType.TTS
            assert "qwen3-tts" in info.architectures
            assert info.default_sample_rate == 24000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_streaming is True
            assert info.supports_voice_cloning is False
            assert "ru" in info.supported_languages

    def test_load_rejects_base_voice_clone_checkpoints(self):
        torch_mock = _mock_torch()
        qwen_tts_module, _ = _mock_qwen_tts_module()

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()

            with pytest.raises(ValueError, match="CustomVoice checkpoints"):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    def test_load_uses_official_qwen_runtime(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["Ryan", "Aiden"]
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            with patch("vox_qwen.tts_adapter._supports_flash_attention", return_value=True):
                adapter.load(
                    "local-path",
                    "cuda",
                    _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    default_voice="Ryan",
                )

            model_cls.from_pretrained.assert_called_once()
            kwargs = model_cls.from_pretrained.call_args.kwargs
            assert kwargs["device_map"] == "cuda:0"
            assert kwargs["dtype"] is torch_mock.bfloat16
            assert kwargs["attn_implementation"] == "flash_attention_2"
            assert adapter._model is model_instance
            assert adapter._tokenizer is model_instance.processor
            assert adapter._default_voice == "Ryan"
            assert adapter._supported_speakers == ["Ryan", "Aiden"]

    def test_load_skips_flash_attention_when_runtime_missing(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["Ryan"]
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            with patch("vox_qwen.tts_adapter._supports_flash_attention", return_value=False):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

            kwargs = model_cls.from_pretrained.call_args.kwargs
            assert "attn_implementation" not in kwargs

    def test_synthesize_uses_custom_voice_runtime_with_default_voice(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["Ryan", "Aiden"]
        model_instance.generate_custom_voice.return_value = ([np.array([0.0, 0.25, 0.5], dtype=np.float32)], 24000)
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            with patch("vox_qwen.tts_adapter._supports_flash_attention", return_value=True):
                adapter.load(
                    "local-path",
                    "cuda",
                    _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    default_voice="Ryan",
                )

            async def run():
                chunks = []
                async for chunk in adapter.synthesize(
                    "Hello",
                    language="en",
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(run())

            model_instance.generate_custom_voice.assert_called_once()
            kwargs = model_instance.generate_custom_voice.call_args.kwargs
            assert kwargs["speaker"] == "Ryan"
            assert kwargs["language"] == "English"
            assert kwargs["instruct"] is None
            assert chunks[-1].is_final is True
            assert any(chunk.audio for chunk in chunks[:-1])

    def test_synthesize_normalizes_russian_language(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["Ryan", "Aiden"]
        model_instance.generate_custom_voice.return_value = ([np.array([0.0, 0.25, 0.5], dtype=np.float32)], 24000)
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            with patch("vox_qwen.tts_adapter._supports_flash_attention", return_value=True):
                adapter.load(
                    "local-path",
                    "cuda",
                    _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    default_voice="Ryan",
                )

            async def run():
                async for _ in adapter.synthesize("Hello", language="ru"):
                    pass

            asyncio.run(run())

            kwargs = model_instance.generate_custom_voice.call_args.kwargs
            assert kwargs["language"] == "Russian"

    def test_load_rejects_base_model_without_reference_audio(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = []
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            with pytest.raises(ValueError, match="CustomVoice checkpoints"):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    def test_list_voices_returns_supported_speakers(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["Ryan", "Aiden"]
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter.load("local-path", "cuda", _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

            voices = adapter.list_voices()

            assert [voice.id for voice in voices] == ["Ryan", "Aiden"]

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            assert adapter.is_loaded is False

    def test_load_falls_back_to_subprocess_mode(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            with patch("vox_qwen.tts_adapter._load_qwen_tts_model", side_effect=RuntimeError("broken")):
                adapter.load(
                    "local-path",
                    "cuda",
                    _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    default_voice="Ryan",
                )

            assert adapter.is_loaded is True
            assert adapter._subprocess_only is True
            assert adapter._supported_speakers == ["Ryan"]

    def test_synthesize_uses_subprocess_fallback(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._subprocess_only = True
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            adapter._default_voice = "Ryan"
            adapter._device = "cpu"

            audio = np.zeros(24_000, dtype=np.float32).tobytes()
            payload = json.dumps({
                "sample_rate": 24_000,
                "audio_b64": base64.b64encode(audio).decode("ascii"),
            })

            async def collect():
                chunks = []
                async for chunk in adapter.synthesize("hello"):
                    chunks.append(chunk)
                return chunks

            with patch("vox_qwen.tts_adapter.subprocess.run", return_value=subprocess.CompletedProcess([], 0, payload, "")):
                chunks = asyncio.run(collect())

            assert any(chunk.audio for chunk in chunks)
            assert chunks[-1].is_final is True

    def test_list_voices_returns_empty(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            assert adapter.list_voices() == []

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
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
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            assert adapter.estimate_vram_bytes() == 2_500_000_000

    def test_estimate_vram_1_7b(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            assert adapter.estimate_vram_bytes() == 4_000_000_000

    def test_estimate_vram_uses_source_hint_before_load(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            assert adapter.estimate_vram_bytes(_source="Qwen/Qwen3-TTS-12Hz-1.7B-Base") == 4_000_000_000


class TestQwenRuntimeBootstrap:
    def test_ensure_runtime_prefers_uv_before_python_pip(self, tmp_path):
        import vox_qwen.runtime as runtime

        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            patch.object(runtime, "_runtime_root", return_value=tmp_path),
            patch.object(runtime, "_module_available", side_effect=[False, True]),
            patch("subprocess.run", side_effect=fake_run),
        ):
            runtime.ensure_runtime("qwen-tts", "qwen-tts", "qwen_tts")

        assert calls[0][0:2] == ["uv", "pip"]

    def test_qwen_tts_runtime_purges_accelerate(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen import tts_adapter

            with patch.object(tts_adapter, "ensure_runtime") as ensure_runtime:
                tts_adapter._load_qwen_tts_model()

            assert ensure_runtime.call_args.kwargs["purge_modules"] == (
                "accelerate",
                "transformers",
                "tokenizers",
                "qwen_tts",
            )

    def test_qwen_asr_runtime_purges_accelerate(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen import asr_adapter

            with patch.object(asr_adapter, "ensure_runtime") as ensure_runtime:
                asr_adapter._load_qwen_asr_model()

            assert ensure_runtime.call_args.kwargs["purge_modules"] == (
                "accelerate",
                "transformers",
                "tokenizers",
                "qwen_asr",
            )
