from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


class TestSpeechT5STTAdapterInfo:
    def test_module_shims_huggingface_hub_offline_mode_for_transformers_compat(self):
        transformers = MagicMock()
        torch = MagicMock()
        huggingface_hub = ModuleType("huggingface_hub")
        huggingface_hub_dataclasses = ModuleType("huggingface_hub.dataclasses")

        with patch.dict(
            "sys.modules",
            {
                "transformers": transformers,
                "torch": torch,
                "huggingface_hub": huggingface_hub,
                "huggingface_hub.dataclasses": huggingface_hub_dataclasses,
            },
        ):
            sys.modules.pop("vox_microsoft.speecht5_stt_adapter", None)
            importlib.import_module("vox_microsoft.speecht5_stt_adapter")

            assert hasattr(huggingface_hub, "is_offline_mode")
            assert huggingface_hub.is_offline_mode() is False
            assert hasattr(huggingface_hub_dataclasses, "validate_typed_dict")
            sentinel = object()
            assert huggingface_hub_dataclasses.validate_typed_dict(sentinel) is sentinel

    def test_module_shims_validate_typed_dict_even_if_offline_mode_already_exists(self):
        transformers = MagicMock()
        torch = MagicMock()
        huggingface_hub = ModuleType("huggingface_hub")
        huggingface_hub.is_offline_mode = lambda: True
        huggingface_hub_dataclasses = ModuleType("huggingface_hub.dataclasses")

        with patch.dict(
            "sys.modules",
            {
                "transformers": transformers,
                "torch": torch,
                "huggingface_hub": huggingface_hub,
                "huggingface_hub.dataclasses": huggingface_hub_dataclasses,
            },
        ):
            sys.modules.pop("vox_microsoft.speecht5_stt_adapter", None)
            importlib.import_module("vox_microsoft.speecht5_stt_adapter")

            assert huggingface_hub.is_offline_mode() is True
            assert hasattr(huggingface_hub_dataclasses, "validate_typed_dict")
            sentinel = object()
            assert huggingface_hub_dataclasses.validate_typed_dict(sentinel) is sentinel

    def test_module_shims_wrap_existing_typed_dict_validator_that_rejects_union_types(self):
        transformers = MagicMock()
        torch = MagicMock()
        huggingface_hub = ModuleType("huggingface_hub")
        huggingface_hub_dataclasses = ModuleType("huggingface_hub.dataclasses")

        def _raising_validate_typed_dict(*args, **kwargs):
            raise TypeError("Unsupported type for field 'transformers_version': str | None")

        huggingface_hub_dataclasses.validate_typed_dict = _raising_validate_typed_dict

        with patch.dict(
            "sys.modules",
            {
                "transformers": transformers,
                "torch": torch,
                "huggingface_hub": huggingface_hub,
                "huggingface_hub.dataclasses": huggingface_hub_dataclasses,
            },
        ):
            sys.modules.pop("vox_microsoft.speecht5_stt_adapter", None)
            importlib.import_module("vox_microsoft.speecht5_stt_adapter")

            sentinel = object()
            assert huggingface_hub_dataclasses.validate_typed_dict(sentinel) is sentinel

    def test_module_shims_wrap_hf_hub_download_without_tqdm_class_support(self):
        transformers = MagicMock()
        torch = MagicMock()
        huggingface_hub = ModuleType("huggingface_hub")
        huggingface_hub_dataclasses = ModuleType("huggingface_hub.dataclasses")
        huggingface_hub_file_download = ModuleType("huggingface_hub.file_download")
        calls: list[tuple[str, str, str | None]] = []

        def _hf_hub_download(repo_id: str, filename: str, *, cache_dir: str | None = None):
            calls.append((repo_id, filename, cache_dir))
            return "ok"

        huggingface_hub.hf_hub_download = _hf_hub_download
        huggingface_hub_file_download.hf_hub_download = _hf_hub_download

        with patch.dict(
            "sys.modules",
            {
                "transformers": transformers,
                "torch": torch,
                "huggingface_hub": huggingface_hub,
                "huggingface_hub.dataclasses": huggingface_hub_dataclasses,
                "huggingface_hub.file_download": huggingface_hub_file_download,
            },
        ):
            sys.modules.pop("vox_microsoft.speecht5_stt_adapter", None)
            importlib.import_module("vox_microsoft.speecht5_stt_adapter")

            assert huggingface_hub.hf_hub_download(
                "repo",
                "weights.bin",
                cache_dir="/tmp/cache",
                tqdm_class=object,
            ) == "ok"
            assert huggingface_hub_file_download.hf_hub_download(
                "repo",
                "weights.bin",
                cache_dir="/tmp/cache",
                tqdm_class=object,
            ) == "ok"
            assert calls == [
                ("repo", "weights.bin", "/tmp/cache"),
                ("repo", "weights.bin", "/tmp/cache"),
            ]

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

    def test_load_prefers_local_model_directory(self, tmp_path: Path):
        transformers = MagicMock()
        torch = MagicMock()
        model_dir = tmp_path / "speecht5-asr"
        model_dir.mkdir()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.speecht5_stt_adapter import SpeechT5STTAdapter

            processor = MagicMock()
            model = MagicMock()
            model.to.return_value = model
            transformers.SpeechT5Processor.from_pretrained.return_value = processor
            transformers.SpeechT5ForSpeechToText.from_pretrained.return_value = model

            adapter = SpeechT5STTAdapter()
            adapter.load(str(model_dir), "cpu", _source="microsoft/speecht5_asr")

            transformers.SpeechT5Processor.from_pretrained.assert_called_once_with(str(model_dir))
            transformers.SpeechT5ForSpeechToText.from_pretrained.assert_called_once_with(str(model_dir))

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

    def test_load_prefers_local_model_directory(self, tmp_path: Path):
        transformers = MagicMock()
        torch = MagicMock()
        datasets = MagicMock()
        datasets.load_dataset.side_effect = Exception("no dataset")
        model_dir = tmp_path / "speecht5-tts"
        model_dir.mkdir()
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
            adapter.load(str(model_dir), "cpu", _source="microsoft/speecht5_tts")

            transformers.SpeechT5Processor.from_pretrained.assert_called_once_with(str(model_dir))
            transformers.SpeechT5ForTextToSpeech.from_pretrained.assert_called_once_with(str(model_dir))

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
                "vox_microsoft.vibevoice_tts_adapter._runtime_has_package_path",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_dist_version",
                side_effect=lambda name: {
                    "vibevoice": "1.0.0",
                    "transformers": "4.50.0",
                    "accelerate": "1.6.0",
                    "huggingface_hub": "0.35.3",
                    "tokenizers": "0.21.4",
                }.get(name, "1.0.0"),
            ):
                adapter = VibeVoiceTTSAdapter()

                with pytest.raises(RuntimeError, match="transformers==4.51.3 required"):
                    adapter.load("microsoft/VibeVoice-Realtime-0.5B", "cpu")

    def test_load_primes_streaming_runtime_and_puts_model_in_eval_mode(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            with patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_has_package_path",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_dist_version",
                side_effect=lambda name: {
                    "vibevoice": "1.0.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                    "huggingface_hub": "0.35.3",
                    "tokenizers": "0.21.4",
                }.get(name, "1.0.0"),
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
                "vox_microsoft.vibevoice_tts_adapter._runtime_has_package_path",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_dist_version",
                side_effect=lambda name: {
                    "vibevoice": "1.0.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                    "huggingface_hub": "0.35.3",
                    "tokenizers": "0.21.4",
                }.get(name, "1.0.0"),
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

            transformers.AutoModel.register.side_effect = ValueError(
                "<class 'vibevoice.modular.configuration_vibevoice."
                "VibeVoiceAcousticTokenizerConfig'> is already used by a Transformers model."
            )
            with patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_has_package_path",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_dist_version",
                side_effect=lambda name: {
                    "vibevoice": "1.0.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                    "huggingface_hub": "0.35.3",
                    "tokenizers": "0.21.4",
                }.get(name, "1.0.0"),
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
            installed = {
                "vibevoice": False,
                "diffusers": False,
                "PIL": False,
                "transformers": False,
                "accelerate": False,
                "huggingface_hub": False,
                "tokenizers": False,
            }
            processor_module = MagicMock()
            processor_module.VibeVoiceStreamingProcessor.from_pretrained.return_value = MagicMock()
            model = MagicMock()
            model_module = MagicMock()
            model_module.VibeVoiceStreamingForConditionalGenerationInference.from_pretrained.return_value = model
            with patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_has_package_path",
                side_effect=lambda name: installed.get(name, False),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.Path.home",
                return_value=Path("/tmp/home"),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.subprocess.run",
                side_effect=lambda *args, **kwargs: installed.update({key: True for key in installed}) or completed,
            ) as run, patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_dist_version",
                side_effect=lambda name: {
                    "vibevoice": "1.0.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                    "huggingface_hub": "0.35.3",
                    "tokenizers": "0.21.4",
                }.get(name, "1.0.0"),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter._prime_runtime"
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.importlib.import_module",
                side_effect=[processor_module, model_module],
            ):
                adapter = VibeVoiceTTSAdapter()
                adapter.load("microsoft/VibeVoice-Realtime-0.5B", "cpu")

                assert run.call_count == 7
                assert any(
                    call.args[0]
                    == [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--target",
                        "/tmp/home/.vox/runtime/vibevoice",
                        "--no-deps",
                        "vibevoice[streamingtts] @ git+https://github.com/microsoft/VibeVoice.git@main",
                    ]
                    for call in run.call_args_list
                )
                assert any(
                    call.args[0]
                    == [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--target",
                        "/tmp/home/.vox/runtime/vibevoice",
                        "--no-deps",
                        "diffusers",
                    ]
                    for call in run.call_args_list
                )
                assert any(
                    call.args[0]
                    == [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--target",
                        "/tmp/home/.vox/runtime/vibevoice",
                        "--no-deps",
                        "pillow",
                    ]
                    for call in run.call_args_list
                )
                assert any(
                    call.args[0]
                    == [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--target",
                        "/tmp/home/.vox/runtime/vibevoice",
                        "--no-deps",
                        "transformers==4.51.3",
                    ]
                    for call in run.call_args_list
                )
                assert any(
                    call.args[0]
                    == [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--target",
                        "/tmp/home/.vox/runtime/vibevoice",
                        "--no-deps",
                        "accelerate==1.6.0",
                    ]
                    for call in run.call_args_list
                )
                assert any(
                    call.args[0]
                    == [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--target",
                        "/tmp/home/.vox/runtime/vibevoice",
                        "--no-deps",
                        "huggingface-hub==0.35.3",
                    ]
                    for call in run.call_args_list
                )
                assert any(
                    call.args[0]
                    == [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--target",
                        "/tmp/home/.vox/runtime/vibevoice",
                        "--no-deps",
                        "tokenizers==0.21.4",
                    ]
                    for call in run.call_args_list
                )

    def test_bootstrap_runtime_reinstalls_transformers_when_global_version_is_too_new(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            completed = MagicMock(returncode=0, stderr="")
            state = {"transformers": "4.57.6"}
            installed = {
                "vibevoice": True,
                "diffusers": False,
                "PIL": True,
                "transformers": True,
                "accelerate": True,
                "huggingface_hub": True,
                "tokenizers": True,
            }
            processor_module = MagicMock()
            processor_module.VibeVoiceStreamingProcessor.from_pretrained.return_value = MagicMock()
            model = MagicMock()
            model_module = MagicMock()
            model_module.VibeVoiceStreamingForConditionalGenerationInference.from_pretrained.return_value = model

            def version_side_effect(name: str) -> str:
                versions = {
                    "vibevoice": "1.0.0",
                    "transformers": state["transformers"],
                    "accelerate": "1.6.0",
                    "huggingface_hub": "0.35.3",
                    "tokenizers": "0.21.4",
                }
                return versions.get(name, "1.0.0")

            with patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_has_package_path",
                side_effect=lambda name: installed.get(name, False),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.Path.home",
                return_value=Path("/tmp/home"),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.subprocess.run",
                side_effect=lambda *args, **kwargs: (
                    state.__setitem__("transformers", "4.51.3"),
                    installed.update({"diffusers": True}),
                    completed,
                )[-1],
            ) as run, patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_dist_version",
                side_effect=version_side_effect,
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter._prime_runtime"
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.importlib.import_module",
                side_effect=[processor_module, model_module],
            ):
                adapter = VibeVoiceTTSAdapter()
                adapter.load("microsoft/VibeVoice-Realtime-0.5B", "cpu")

                assert any(
                    call.args[0][-1] == "transformers==4.51.3"
                    for call in run.call_args_list
                )

    def test_vibevoice_accepts_newer_runtime_package_version(self):
        transformers = MagicMock()
        torch = MagicMock()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            with patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_has_package_path",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_dist_version",
                side_effect=lambda name: {
                    "vibevoice": "1.0.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                    "huggingface_hub": "0.35.3",
                    "tokenizers": "0.21.4",
                }.get(name, "1.0.0"),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.importlib.import_module"
            ):
                adapter = VibeVoiceTTSAdapter()
                adapter.load("microsoft/VibeVoice-Realtime-0.5B", "cpu")

    def test_vibevoice_load_prefers_local_model_directory(self, tmp_path: Path):
        transformers = MagicMock()
        torch = MagicMock()
        model_dir = tmp_path / "vibevoice"
        model_dir.mkdir()
        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            with patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_has_package_path",
                return_value=MagicMock(),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter._runtime_dist_version",
                side_effect=lambda name: {
                    "vibevoice": "1.0.0",
                    "transformers": "4.51.3",
                    "accelerate": "1.6.0",
                    "huggingface_hub": "0.35.3",
                    "tokenizers": "0.21.4",
                }.get(name, "1.0.0"),
            ), patch(
                "vox_microsoft.vibevoice_tts_adapter.importlib.import_module"
            ) as import_module:
                processor = MagicMock()
                model = MagicMock()
                config_module = MagicMock()
                config_module.VibeVoiceStreamingConfig = MagicMock(model_type="vibevoice_streaming")
                runtime_model_module = MagicMock()
                (
                    runtime_model_module
                    .VibeVoiceStreamingForConditionalGenerationInference
                    .from_pretrained
                    .return_value
                ) = model
                processor_module = MagicMock()
                processor_class = MagicMock()
                processor_class.from_pretrained.return_value = processor
                processor_module.VibeVoiceStreamingProcessor = processor_class
                import_module.side_effect = [
                    config_module,
                    runtime_model_module,
                    processor_module,
                    runtime_model_module,
                ]

                adapter = VibeVoiceTTSAdapter()
                adapter.load(str(model_dir), "cpu", _source="microsoft/VibeVoice-Realtime-0.5B")

                processor_class.from_pretrained.assert_called_once_with(str(model_dir))
                runtime_model_module.VibeVoiceStreamingForConditionalGenerationInference.from_pretrained.assert_called_once_with(
                    str(model_dir),
                    torch_dtype=ANY,
                    device_map="cpu",
                    attn_implementation="sdpa",
                )

    def test_list_voices_includes_default_alias(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            voices = adapter.list_voices()
            assert voices[0].id == "default"
            assert voices[1].id == "en-Carter_man"

    @pytest.mark.asyncio
    async def test_synthesize_prefers_speech_outputs_over_generation_tokens(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            adapter._loaded = True
            adapter._device = "cpu"
            adapter._model_id = "microsoft/VibeVoice-Realtime-0.5B"
            adapter._processor = MagicMock()
            adapter._processor.process_input_with_cached_prompt.return_value = {}
            adapter._load_streaming_prompt = MagicMock(return_value={"cached": True})

            token_ids = np.arange(12, dtype=np.int32)
            speech = np.linspace(-0.25, 0.25, num=4096, dtype=np.float32)
            adapter._model = MagicMock(
                generate=MagicMock(
                    return_value=type(
                        "Output",
                        (),
                        {
                            "speech_outputs": [speech],
                            "__getitem__": lambda self, idx: token_ids,
                        },
                    )()
                )
            )

            chunks = [chunk async for chunk in adapter.synthesize("hello world", voice="alloy")]

            assert len(chunks) == 2
            assert chunks[0].audio == speech.tobytes()
            assert chunks[0].sample_rate == 24000
            assert chunks[0].is_final is False
            assert chunks[1].audio == b""
            assert chunks[1].is_final is True

    def test_coerce_audio_array_casts_torch_like_tensors_to_float32(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            class FakeTensor:
                def __init__(self, values):
                    self._values = values

                def float(self):
                    return self

                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return self._values

            audio = VibeVoiceTTSAdapter._coerce_audio_array(
                type("Output", (), {"speech_outputs": [FakeTensor(np.array([1, 2, 3], dtype=np.float32))]})()
            )

            assert audio.dtype == np.float32
            assert audio.tolist() == [1.0, 2.0, 3.0]

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
                "vox_microsoft.vibevoice_tts_adapter._download_streaming_prompt"
            ) as download, patch(
                "vox_microsoft.vibevoice_tts_adapter.torch.load",
                return_value={"cached": "prompt"},
            ) as torch_load:
                prompt = adapter._load_streaming_prompt(None)
                cached = adapter._load_streaming_prompt(None)

            download.assert_called_once_with(
                "https://raw.githubusercontent.com/microsoft/VibeVoice/main/"
                "demo/voices/streaming_model/en-Carter_man.pt",
                prompt_path,
                accept=None,
            )
            torch_load.assert_called_once_with(prompt_path, map_location="cpu", weights_only=False)
            assert prompt == {"cached": "prompt"}
            assert cached == prompt

    def test_load_streaming_prompt_maps_aliases_to_real_prompt(self, tmp_path):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_microsoft.vibevoice_tts_adapter import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            adapter._device = "cpu"
            prompt_path = tmp_path / "en-Carter_man.pt"

            for alias in ("default", "alloy"):
                with patch.object(
                    VibeVoiceTTSAdapter,
                    "_streaming_prompt_dir",
                    return_value=tmp_path,
                ), patch(
                    "vox_microsoft.vibevoice_tts_adapter._download_streaming_prompt"
                ) as download, patch(
                    "vox_microsoft.vibevoice_tts_adapter.torch.load",
                    return_value={"cached": "prompt"},
                ):
                    adapter._streaming_prompt_cache.clear()
                    adapter._load_streaming_prompt(alias)

                download.assert_called_once_with(
                    "https://raw.githubusercontent.com/microsoft/VibeVoice/main/"
                    "demo/voices/streaming_model/en-Carter_man.pt",
                    prompt_path,
                    accept=None,
                )

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
