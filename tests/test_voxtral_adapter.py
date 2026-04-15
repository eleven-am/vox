from __future__ import annotations

import asyncio
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


def _build_voxtral_tts_runtime(
    *,
    supported_voices: list[dict[str, str]] | None = None,
    generate_side_effect: Any | None = None,
):
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.empty_cache = MagicMock()
    torch.backends.mps.is_available.return_value = False
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"

    sampling_params_cls = MagicMock(name="SamplingParams")

    class SpeechRequest:
        def __init__(self, *, input: str, voice: str):
            self.input = input
            self.voice = voice

    tokenizer = MagicMock()
    instruct_tokenizer = MagicMock()
    tokenizer.instruct_tokenizer = instruct_tokenizer
    tokenizer_cls = MagicMock()
    tokenizer_cls.from_hf_hub.return_value = tokenizer
    tokenizer_cls.from_file.return_value = tokenizer

    runtime = MagicMock()
    runtime.shutdown = MagicMock()
    runtime.get_supported_voices = MagicMock(return_value=supported_voices or [])

    if generate_side_effect is None:
        async def generate_side_effect(*args, **kwargs):
            if False:
                yield None
    runtime.generate = MagicMock(side_effect=generate_side_effect)

    modules = {
        "torch": torch,
        "vllm": ModuleType("vllm"),
        "vllm_omni": ModuleType("vllm_omni"),
        "mistral_common": ModuleType("mistral_common"),
        "mistral_common.protocol": ModuleType("mistral_common.protocol"),
        "mistral_common.protocol.speech": ModuleType("mistral_common.protocol.speech"),
        "mistral_common.protocol.speech.request": ModuleType("mistral_common.protocol.speech.request"),
        "mistral_common.tokens": ModuleType("mistral_common.tokens"),
        "mistral_common.tokens.tokenizers": ModuleType("mistral_common.tokens.tokenizers"),
        "mistral_common.tokens.tokenizers.mistral": ModuleType("mistral_common.tokens.tokenizers.mistral"),
    }
    modules["vllm"].SamplingParams = sampling_params_cls
    modules["vllm_omni"].AsyncOmni = MagicMock(return_value=runtime)
    modules["mistral_common.protocol.speech.request"].SpeechRequest = SpeechRequest
    modules["mistral_common.tokens.tokenizers.mistral"].MistralTokenizer = tokenizer_cls
    return modules, torch, runtime, tokenizer, tokenizer_cls, sampling_params_cls


class TestVoxtralSTTAdapterInfo:
    def test_package_import_does_not_require_tts_support(self):
        transformers = MagicMock()
        transformers.AutoProcessor = MagicMock()
        transformers.VoxtralForConditionalGeneration = MagicMock()
        torch = MagicMock()

        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            sys.modules.pop("vox_voxtral", None)
            module = importlib.import_module("vox_voxtral")
            assert module.__all__ == ["VoxtralSTTAdapter", "VoxtralTTSAdapter"]

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

    def test_transcribe_uses_transcription_request(self):
        torch = MagicMock()
        torch.float32 = "float32"
        torch.bfloat16 = "bfloat16"
        torch.inference_mode.return_value.__enter__.return_value = None
        torch.inference_mode.return_value.__exit__.return_value = False

        processor = MagicMock()
        input_ids = MagicMock()
        input_ids.shape = (1, 2)
        moved_input_ids = MagicMock()
        moved_features = MagicMock()
        processor.apply_transcription_request.return_value = {
            "input_ids": moved_input_ids,
            "input_features": moved_features,
        }
        moved_input_ids.to.return_value = input_ids
        moved_features.to.return_value = moved_features
        processor.decode.return_value = "hello world"

        output_ids = np.array([[10, 11, 12, 13]])
        model = MagicMock()
        model.device = "cuda"
        model.generate.return_value = output_ids

        transformers = MagicMock()
        transformers.AutoProcessor = MagicMock()
        transformers.VoxtralForConditionalGeneration = MagicMock()

        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            adapter._loaded = True
            adapter._processor = processor
            adapter._model = model
            adapter._model_id = "mistralai/Voxtral-Mini-3B-2507"

            result = adapter.transcribe(np.zeros(16000, dtype=np.float32), language="en")

            processor.apply_transcription_request.assert_called_once()
            _, kwargs = processor.apply_transcription_request.call_args
            assert kwargs["model_id"] == "mistralai/Voxtral-Mini-3B-2507"
            assert kwargs["language"] == "en"
            assert kwargs["sampling_rate"] == 16000
            assert kwargs["format"] == ["wav"]
            assert result.text == "hello world"

    def test_detect_language_raises_not_implemented(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()

            with pytest.raises(NotImplementedError, match="auto-detect"):
                adapter.detect_language(np.zeros(16000, dtype=np.float32))

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

    def test_estimate_vram_uses_source_hint_before_load(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.stt_adapter import VoxtralSTTAdapter

            adapter = VoxtralSTTAdapter()
            assert adapter.estimate_vram_bytes(_source="mistralai/Voxtral-Small-24B-2507") > 9_500_000_000


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
            assert info.supports_voice_cloning is False

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
            assert "neutral_female" in voice_ids

    def test_unload_resets_state(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            adapter._loaded = True
            adapter._runtime = MagicMock()
            adapter._tokenizer = MagicMock()
            adapter._speech_request_cls = MagicMock()

            adapter.unload()

            assert adapter.is_loaded is False
            assert adapter._runtime is None
            assert adapter._tokenizer is None
            assert adapter._speech_request_cls is None

    def test_estimate_vram(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            assert adapter.estimate_vram_bytes() == 16_000_000_000

    def test_load_raises_clear_error_when_vllm_omni_lacks_runtime(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.backends.mps.is_available.return_value = False
        torch.float32 = "float32"
        torch.bfloat16 = "bfloat16"

        with patch.dict("sys.modules", {"torch": torch}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            with patch(
                "vox_voxtral.tts_adapter.ensure_voxtral_tts_runtime",
                side_effect=RuntimeError("vllm-omni unavailable"),
            ), pytest.raises(RuntimeError, match="vllm-omni"):
                adapter.load("mistralai/Voxtral-4B-TTS-2603", "auto")

    def test_load_uses_vllm_omni_runtime(self):
        modules, torch, runtime, tokenizer, tokenizer_cls, sampling_params_cls = _build_voxtral_tts_runtime()

        with patch.dict("sys.modules", modules):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            adapter.load(
                "mistralai/Voxtral-4B-TTS-2603",
                "cuda",
                _source="mistralai/Voxtral-4B-TTS-2603",
                _stage_configs_path="/tmp/voxtral_tts.yaml",
            )

            assert adapter.is_loaded is True
            assert adapter._runtime is runtime
            assert adapter._tokenizer is tokenizer
            tokenizer_cls.from_hf_hub.assert_called_once_with("mistralai/Voxtral-4B-TTS-2603")
            modules["vllm_omni"].AsyncOmni.assert_called_once()
            _, kwargs = modules["vllm_omni"].AsyncOmni.call_args
            assert kwargs["model"] == "mistralai/Voxtral-4B-TTS-2603"
            assert kwargs["stage_configs_path"] == "/tmp/voxtral_tts.yaml"
            assert len(adapter._sampling_params) == 2
            sampling_params_cls.assert_called_once_with(max_tokens=2500)
            assert torch.cuda.is_available.called

    def test_synthesize_streams_voice_preset(self):
        async def generate(*args, **kwargs):
            yield SimpleNamespace(
                multimodal_output={"audio": np.array([0.1, 0.2], dtype=np.float32)},
                finished=False,
            )
            yield SimpleNamespace(
                multimodal_output={"audio": np.array([0.3, 0.4], dtype=np.float32)},
                finished=True,
            )

        modules, torch, runtime, tokenizer, tokenizer_cls, _ = _build_voxtral_tts_runtime(
            supported_voices=[{"id": "neutral_female", "name": "neutral_female"}],
            generate_side_effect=generate,
        )
        tokenizer.instruct_tokenizer.encode_speech_request.return_value = MagicMock(tokens=[1, 2, 3])

        with patch.dict("sys.modules", modules):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            adapter.load(
                "mistralai/Voxtral-4B-TTS-2603",
                "cuda",
                _source="mistralai/Voxtral-4B-TTS-2603",
                _stage_configs_path="/tmp/voxtral_tts.yaml",
                default_voice="neutral_female",
            )

            async def collect() -> list[Any]:
                chunks: list[Any] = []
                async for chunk in adapter.synthesize("hello world", voice="neutral_female"):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(collect())

            assert len(chunks) == 3
            assert chunks[0].is_final is False
            assert chunks[1].is_final is False
            assert chunks[2].is_final is True
            assert chunks[0].sample_rate == 24000
            assert runtime.generate.called
            tokenizer.instruct_tokenizer.encode_speech_request.assert_called_once()
            request = tokenizer.instruct_tokenizer.encode_speech_request.call_args.args[0]
            assert request.input == "hello world"
            assert request.voice == "neutral_female"

    def test_load_falls_back_to_subprocess_worker(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.backends.mps.is_available.return_value = False
        torch.float32 = "float32"
        torch.bfloat16 = "bfloat16"

        runtime_info = SimpleNamespace(
            python_executable="/tmp/voxtral-python",
            env={},
            stage_configs_path="/tmp/voxtral_tts.yaml",
            site_packages="/tmp/site-packages",
        )
        worker = MagicMock()
        worker.stdout.readline.return_value = json.dumps({"status": "ready"}) + "\n"
        worker.stderr.read.return_value = ""

        with patch.dict("sys.modules", {"torch": torch}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            with (
                patch("vox_voxtral.tts_adapter.ensure_voxtral_tts_runtime", return_value=runtime_info),
                patch("vox_voxtral.tts_adapter.subprocess.Popen", return_value=worker) as popen,
            ):
                adapter.load("mistralai/Voxtral-4B-TTS-2603", "cuda")

            assert adapter.is_loaded is True
            assert adapter._subprocess_only is True
            popen.assert_called_once()

    def test_synthesize_uses_subprocess_worker(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            adapter._loaded = True
            adapter._subprocess_only = True
            adapter._worker_request = MagicMock(
                return_value={
                    "status": "ok",
                    "sample_rate": 24000,
                    "audio_b64": "AACAPwAAAEA=",
                }
            )

            chunks = asyncio.run(_collect_voxtral_tts_stream(adapter))

            assert len(chunks) == 2
            assert chunks[0].is_final is False
            assert chunks[0].audio == b"\x00\x00\x80?\x00\x00\x00@"
            assert chunks[1].is_final is True

    def test_synthesize_rejects_reference_audio_until_released(self):
        modules, _, _, tokenizer, _, _ = _build_voxtral_tts_runtime()
        tokenizer.instruct_tokenizer.encode_speech_request.return_value = MagicMock(tokens=[1, 2, 3])

        with patch.dict("sys.modules", modules):
            from vox_voxtral.tts_adapter import VoxtralTTSAdapter

            adapter = VoxtralTTSAdapter()
            adapter.load(
                "mistralai/Voxtral-4B-TTS-2603",
                "cuda",
                _source="mistralai/Voxtral-4B-TTS-2603",
                _stage_configs_path="/tmp/voxtral_tts.yaml",
            )

            with pytest.raises(NotImplementedError, match="reference-audio cloning"):
                asyncio.run(
                    _collect_voxtral_tts_stream(
                        adapter,
                        reference_audio=np.zeros(24000, dtype=np.float32),
                        reference_text="hello",
                    )
                )


async def _collect_voxtral_tts_stream(adapter: Any, **kwargs: Any) -> list[Any]:
    chunks: list[Any] = []
    async for chunk in adapter.synthesize("hello world", **kwargs):
        chunks.append(chunk)
    return chunks
