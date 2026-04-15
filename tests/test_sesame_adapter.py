from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


def _clear_sesame_modules() -> None:
    sys.modules.pop("vox_sesame", None)
    sys.modules.pop("vox_sesame.adapter", None)


class TestSesameAdapterInfo:
    def test_package_import_does_not_require_csm_support(self):
        transformers = MagicMock()
        transformers.AutoProcessor = MagicMock()
        torch = MagicMock()

        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            _clear_sesame_modules()
            module = importlib.import_module("vox_sesame")
            assert module.__all__ == ["SesameTTSAdapter"]

    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            info = adapter.info()

            assert info.name == "sesame-tts-torch"
            assert info.type == ModelType.TTS
            assert "sesame" in info.architectures
            assert info.default_sample_rate == 24_000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_streaming is True
            assert info.supported_languages == ("en",)

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            assert adapter.is_loaded is False

    def test_list_voices_returns_default_speaker(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            voices = adapter.list_voices()
            assert len(voices) == 1
            assert voices[0].id == "0"

    def test_load_uses_csm_model_class(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.backends.mps.is_available.return_value = False
        torch.bfloat16 = "bfloat16"
        torch.float32 = "float32"

        transformers = MagicMock()
        transformers.AutoProcessor = MagicMock()
        processor = MagicMock()
        transformers.AutoProcessor.from_pretrained.return_value = processor

        model_cls = MagicMock()
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = None
        model_cls.from_pretrained.return_value = model
        transformers.CsmForConditionalGeneration = model_cls

        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            adapter.load("sesame/csm-1b", "cuda")

            transformers.AutoProcessor.from_pretrained.assert_called_once_with("sesame/csm-1b")
            model_cls.from_pretrained.assert_called_once_with("sesame/csm-1b", torch_dtype="bfloat16")
            model.to.assert_called_once_with("cuda")
            model.eval.assert_called_once()
            assert adapter.is_loaded is True

    def test_load_raises_clear_error_when_transformers_lacks_csm_model(self):
        transformers = ModuleType("transformers")
        transformers.AutoProcessor = MagicMock()
        torch = MagicMock()

        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            with pytest.raises(RuntimeError, match="CsmForConditionalGeneration"):
                adapter.load("sesame/csm-1b", "auto")

    def test_estimate_vram(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            assert adapter.estimate_vram_bytes() == 5_000_000_000


class TestSesameAdapterSynthesis:
    def test_synthesize_emits_audio_chunks(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        torch.float32 = "float32"
        torch.bfloat16 = "bfloat16"
        inference_ctx = torch.inference_mode.return_value
        inference_ctx.__enter__.return_value = None
        inference_ctx.__exit__.return_value = False

        transformers = MagicMock()
        transformers.AutoProcessor = MagicMock()
        processor = MagicMock()
        inputs = MagicMock()
        inputs.to.return_value = inputs
        processor.apply_chat_template.return_value = inputs
        transformers.AutoProcessor.from_pretrained.return_value = processor

        model_cls = MagicMock()
        model = MagicMock()
        model.to.return_value = model
        model.generate.return_value = SimpleNamespace(
            audio=np.arange(48_000, dtype=np.float32),
            sample_rate=24_000,
        )
        model_cls.from_pretrained.return_value = model
        transformers.CsmForConditionalGeneration = model_cls

        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            adapter._loaded = True
            adapter._model = model
            adapter._processor = processor
            adapter._device = "cpu"

            chunks = []
            async def collect() -> None:
                async for chunk in adapter.synthesize("Hello Sesame", voice="0"):
                    chunks.append(chunk)

            import asyncio
            asyncio.run(collect())

            assert len(chunks) == 2
            assert chunks[0].sample_rate == 24_000
            assert chunks[0].audio
            assert chunks[0].is_final is False
            assert chunks[-1].is_final is True
            assert chunks[-1].audio == b""

    def test_synthesize_uses_context_when_reference_pair_present(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        torch.float32 = "float32"
        torch.bfloat16 = "bfloat16"
        inference_ctx = torch.inference_mode.return_value
        inference_ctx.__enter__.return_value = None
        inference_ctx.__exit__.return_value = False

        transformers = MagicMock()
        transformers.AutoProcessor = MagicMock()
        processor = MagicMock()
        inputs = MagicMock()
        inputs.to.return_value = inputs
        processor.apply_chat_template.return_value = inputs
        transformers.AutoProcessor.from_pretrained.return_value = processor

        model_cls = MagicMock()
        model = MagicMock()
        model.to.return_value = model
        model.generate.return_value = SimpleNamespace(
            audio=np.arange(24_000, dtype=np.float32),
            sample_rate=24_000,
        )
        model_cls.from_pretrained.return_value = model
        transformers.CsmForConditionalGeneration = model_cls

        with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            adapter._loaded = True
            adapter._model = model
            adapter._processor = processor
            adapter._device = "cpu"

            import asyncio

            async def collect() -> None:
                async for _chunk in adapter.synthesize(
                    "Hello Sesame",
                    voice="7",
                    reference_audio=np.zeros(16_000, dtype=np.float32),
                    reference_text="Earlier context",
                ):
                    pass

            asyncio.run(collect())

            conversation = processor.apply_chat_template.call_args.args[0]
            assert conversation[0]["role"] == "7"
            assert conversation[0]["content"][0]["text"] == "Earlier context"
            assert conversation[0]["content"][1]["type"] == "audio"
            assert conversation[1]["content"][0]["text"] == "Hello Sesame"

    def test_synthesize_requires_context_pairs(self):
        with patch.dict("sys.modules", {"transformers": MagicMock(), "torch": MagicMock()}):
            _clear_sesame_modules()
            from vox_sesame.adapter import SesameTTSAdapter

            adapter = SesameTTSAdapter()
            adapter._loaded = True
            adapter._model = MagicMock()
            adapter._processor = MagicMock()
            adapter._device = "cpu"

            async def collect() -> None:
                async for _ in adapter.synthesize(
                    "Hello",
                    reference_audio=np.zeros(16000, dtype=np.float32),
                ):
                    pass

            import asyncio
            with pytest.raises(ValueError, match="reference_audio and reference_text"):
                asyncio.run(collect())
