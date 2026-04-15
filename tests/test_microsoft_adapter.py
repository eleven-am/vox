from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


class TestSpeechT5STTAdapterInfo:
    def test_package_import_does_not_require_all_microsoft_variants(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            sys.modules.pop("vox_microsoft", None)
            module = importlib.import_module("vox_microsoft")
            assert module.__all__ == ["SpeechT5STTAdapter", "SpeechT5TTSAdapter", "VibeVoiceTTSAdapter"]

    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            adapter = SpeechT5STTAdapter()
            info = adapter.info()

            assert info.name == "speecht5-stt-torch"
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

    def test_load_puts_model_in_eval_mode(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            processor = MagicMock()
            model = MagicMock()
            model.to.return_value = model
            transformers.SpeechT5Processor.from_pretrained.return_value = processor
            transformers.SpeechT5ForSpeechToText.from_pretrained.return_value = model

            adapter = SpeechT5STTAdapter()
            adapter.load("microsoft/speecht5_asr", "cpu")

            model.to.assert_called_once_with("cpu")
            model.eval.assert_called_once()

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

            assert info.name == "speecht5-tts-torch"
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

    def test_load_puts_model_and_vocoder_in_eval_mode(self):
        transformers = MagicMock()
        torch = MagicMock()
        datasets = MagicMock()
        datasets.load_dataset.side_effect = Exception("no dataset")
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch, "datasets": datasets}):
            from vox_microsoft.speecht5_tts_adapter import SpeechT5TTSAdapter

            processor = MagicMock()
            model = MagicMock()
            vocoder = MagicMock()
            model.to.return_value = model
            vocoder.to.return_value = vocoder
            transformers.SpeechT5Processor.from_pretrained.return_value = processor
            transformers.SpeechT5ForTextToSpeech.from_pretrained.return_value = model
            transformers.SpeechT5HifiGan.from_pretrained.return_value = vocoder

            adapter = SpeechT5TTSAdapter()
            adapter.load("microsoft/speecht5_tts", "cpu")

            model.to.assert_called_once_with("cpu")
            vocoder.to.assert_called_once_with("cpu")
            model.eval.assert_called_once()
            vocoder.eval.assert_called_once()

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

            assert info.name == "vibevoice-tts-torch"
            assert info.type == ModelType.TTS
            assert "vibevoice" in info.architectures
            assert info.default_sample_rate == 24000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_streaming is False

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            assert adapter.is_loaded is False

    def test_load_rejects_wrong_runtime_versions(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            with patch(
                "vox_microsoft.vibevoice_tts_adapter.find_spec",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.metadata.version",
                side_effect=lambda name: {
                    "vibevoice": "0.1.0",
                    "transformers": "4.50.0",
                    "accelerate": "1.6.0",
                }[name],
            ):
                adapter = VibeVoiceTTSAdapter()

                with pytest.raises(RuntimeError, match="transformers>=4.51.3 required"):
                    adapter.load("microsoft/VibeVoice-Realtime-0.5B", "cpu")

    def test_load_primes_streaming_runtime_and_puts_model_in_eval_mode(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            with patch(
                "vox_microsoft.vibevoice_tts_adapter.find_spec",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.metadata.version",
                side_effect=lambda name: {
                    "vibevoice": "0.1.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                }[name],
            ):
                tokenizer = MagicMock()
                model = MagicMock()

                adapter = VibeVoiceTTSAdapter()
                with patch("vox_microsoft.vibevoice_tts_adapter.importlib.import_module") as import_module:
                    config_module = MagicMock()
                    config_module.VibeVoiceStreamingConfig = MagicMock(model_type="vibevoice_streaming")
                    runtime_model_module = MagicMock()
                    (
                        runtime_model_module.VibeVoiceStreamingForConditionalGenerationInference.from_pretrained.return_value
                    ) = model
                    processor_module = MagicMock()
                    processor_class = MagicMock()
                    processor_class.from_pretrained.return_value = tokenizer
                    processor_module.VibeVoiceStreamingProcessor = processor_class
                    import_module.side_effect = [
                        config_module,
                        runtime_model_module,
                        processor_module,
                        runtime_model_module,
                    ]
                    adapter.load("microsoft/VibeVoice-Realtime-0.5B", "cpu")
                    import_module.assert_any_call("vibevoice.modular.configuration_vibevoice_streaming")
                    import_module.assert_any_call(
                        "vibevoice.modular.modeling_vibevoice_streaming_inference"
                    )
                    import_module.assert_any_call("vibevoice.processor.vibevoice_streaming_processor")

                runtime_model_module.VibeVoiceStreamingForConditionalGenerationInference.from_pretrained.assert_called_once()
                model.eval.assert_called_once()

    def test_load_primes_longform_runtime(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            with patch(
                "vox_microsoft.vibevoice_tts_adapter.find_spec",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.metadata.version",
                side_effect=lambda name: {
                    "vibevoice": "0.1.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                }[name],
            ):
                tokenizer = MagicMock()
                model = MagicMock()

                adapter = VibeVoiceTTSAdapter()
                with patch("vox_microsoft.vibevoice_tts_adapter.importlib.import_module") as import_module:
                    config_module = MagicMock()
                    config_module.VibeVoiceConfig = MagicMock(model_type="vibevoice")
                    runtime_model_module = MagicMock()
                    runtime_model_module.VibeVoiceForConditionalGeneration.from_pretrained.return_value = model
                    processor_module = MagicMock()
                    processor_class = MagicMock()
                    processor_class.from_pretrained.return_value = tokenizer
                    processor_module.VibeVoiceProcessor = processor_class
                    import_module.side_effect = [
                        config_module,
                        runtime_model_module,
                        processor_module,
                        runtime_model_module,
                    ]
                    adapter.load("microsoft/VibeVoice-1.5B", "cpu")
                    import_module.assert_any_call("vibevoice.modular.configuration_vibevoice")
                    import_module.assert_any_call("vibevoice.modular.modeling_vibevoice")
                    import_module.assert_any_call("vibevoice.processor.vibevoice_processor")

                runtime_model_module.VibeVoiceForConditionalGeneration.from_pretrained.assert_called_once()
                model.eval.assert_called_once()

    def test_load_ignores_duplicate_transformers_registration_conflicts(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            with patch(
                "vox_microsoft.vibevoice_tts_adapter.find_spec",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.metadata.version",
                side_effect=lambda name: {
                    "vibevoice": "0.1.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                }[name],
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.AutoModel.register",
                side_effect=ValueError(
                    "<class 'vibevoice.modular.configuration_vibevoice."
                    "VibeVoiceAcousticTokenizerConfig'> is already used by a Transformers model."
                ),
            ):
                processor = MagicMock()
                model = MagicMock()

                adapter = VibeVoiceTTSAdapter()
                with patch("vox_microsoft.vibevoice_tts_adapter.importlib.import_module") as import_module:
                    config_module = MagicMock()
                    config_module.VibeVoiceConfig = MagicMock(model_type="vibevoice")
                    runtime_model_module = MagicMock()
                    runtime_model_module.VibeVoiceForConditionalGeneration.from_pretrained.return_value = model
                    processor_module = MagicMock()
                    processor_class = MagicMock()
                    processor_class.from_pretrained.return_value = processor
                    processor_module.VibeVoiceProcessor = processor_class
                    import_module.side_effect = [
                        config_module,
                        runtime_model_module,
                        processor_module,
                        runtime_model_module,
                    ]

                    adapter.load("microsoft/VibeVoice-1.5B", "cpu")

                model.eval.assert_called_once()

    def test_bootstrap_runtime_when_package_missing(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            completed = MagicMock(returncode=0, stderr="")
            with patch(
                "vox_microsoft.vibevoice_tts_adapter.find_spec",
                return_value=None,
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.subprocess.run",
                return_value=completed,
            ) as run, patch(
                "vox_microsoft.vibevoice_tts_adapter.metadata.version",
                side_effect=lambda name: {
                    "vibevoice": "0.1.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                }[name],
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.importlib.import_module"
            ):
                adapter = VibeVoiceTTSAdapter()
                adapter.load("microsoft/VibeVoice-Realtime-0.5B", "cpu")

                assert run.call_count == 2
                assert any(
                    "vibevoice[streamingtts] @ git+https://github.com/microsoft/VibeVoice.git@main"
                    in " ".join(call.args[0])
                    for call in run.call_args_list
                )
                assert any("diffusers" in " ".join(call.args[0]) for call in run.call_args_list)

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
            adapter._processor = MagicMock()
            adapter._streaming_prompt_cache = {"en-Carter_man": {"cached": True}}

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._model is None
            assert adapter._processor is None
            assert adapter._streaming_prompt_cache == {}

    def test_load_streaming_prompt_downloads_and_caches_default_voice(self, tmp_path):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            adapter._device = "cpu"
            prompt_path = tmp_path / "en-Carter_man.pt"

            with patch.object(
                VibeVoiceTTSAdapter,
                "_streaming_prompt_dir",
                return_value=tmp_path,
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.urlretrieve"
            ) as retrieve, patch(
                "vox_microsoft.vibevoice_tts_adapter.torch.load",
                return_value={"cached": "prompt"},
            ) as torch_load:
                prompt = adapter._load_streaming_prompt(None)
                cached = adapter._load_streaming_prompt(None)

            retrieve.assert_called_once_with(
                "https://raw.githubusercontent.com/microsoft/VibeVoice/main/"
                "demo/voices/streaming_model/en-Carter_man.pt",
                prompt_path,
            )
            torch_load.assert_called_once_with(prompt_path, map_location="cpu", weights_only=False)
            assert prompt == {"cached": "prompt"}
            assert cached == prompt

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
