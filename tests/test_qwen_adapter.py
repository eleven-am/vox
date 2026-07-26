from __future__ import annotations

import asyncio
import base64
import importlib
import json
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType
from vox.operations.errors import InvalidConfigError


def _mock_torch(cuda_available: bool = True, mps_available: bool = False):
    torch_mock = MagicMock()
    torch_mock.__version__ = "2.8.0"
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


def _mock_faster_qwen_model():
    model = MagicMock()
    model.get_supported_speakers.return_value = ["Ryan", "Aiden", "Sohee"]
    model.generate_custom_voice_streaming.return_value = iter(
        [
            (np.array([0.0, 0.25], dtype=np.float32), 24_000, {"ttfa_ms": 150}),
            (np.array([0.5, 0.75], dtype=np.float32), 24_000, {"rtf": 4.0}),
        ]
    )
    model.generate_voice_clone_streaming.return_value = iter(
        [
            (np.array([0.1, 0.2], dtype=np.float32), 24_000, {"ttfa_ms": 180}),
        ]
    )
    return model


class _BlockingSubprocess:
    def __init__(self, payload: str) -> None:
        self.pid = 4321
        self.returncode: int | None = None
        self.started = threading.Event()
        self.terminated = threading.Event()
        self.reaped = threading.Event()
        self.payload = payload
        self.stdin_payload: str | None = None

    def run(self, cmd, **_):
        self.started.set()
        self.terminated.wait(2.0)
        self.reaped.set()
        return subprocess.CompletedProcess(cmd, self.returncode or 0, self.payload, "")

    def communicate(self, input=None, timeout=None):
        self.stdin_payload = input
        self.started.set()
        if not self.terminated.wait(timeout):
            raise subprocess.TimeoutExpired([], timeout)
        self.reaped.set()
        return self.payload, ""

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.terminated.set()

    def kill(self):
        self.returncode = -9
        self.terminated.set()

    def wait(self, timeout=None):
        if not self.terminated.wait(timeout):
            raise subprocess.TimeoutExpired([], timeout)
        self.reaped.set()
        return self.returncode


def _completed_subprocess(payload: str) -> _BlockingSubprocess:
    process = _BlockingSubprocess(payload)
    process.returncode = 0
    process.terminated.set()
    return process


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

            assert info.name == "qwen3-stt-torch"
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
            with (
                patch("vox_qwen.asr_adapter.ensure_runtime"),
                patch("vox_qwen.asr_adapter._supports_flash_attention", return_value=True),
            ):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-ASR-0.6B")

            model_cls.from_pretrained.assert_called_once()
            kwargs = model_cls.from_pretrained.call_args.kwargs
            assert kwargs["device_map"] == "cuda:0"
            assert kwargs["dtype"] is torch_mock.bfloat16
            assert kwargs["attn_implementation"] == "flash_attention_2"
            assert adapter._model is model_instance
            assert adapter._processor is model_instance.processor
            assert forced_aligner_cls.from_pretrained.call_count == 0
            assert adapter._model_ref == "Qwen/Qwen3-ASR-0.6B"

    def test_load_prefers_local_model_path_when_present(self, tmp_path: Path):
        torch_mock = _mock_torch()
        qwen_asr_module, model_cls, _ = _mock_qwen_asr_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_cls.from_pretrained.return_value = model_instance
        model_dir = tmp_path / "qwen-asr"
        model_dir.mkdir()

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": qwen_asr_module, "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            adapter.load(str(model_dir), "cuda", _source="Qwen/Qwen3-ASR-0.6B")

            model_cls.from_pretrained.assert_called_once()
            assert model_cls.from_pretrained.call_args.args[0] == str(model_dir)
            assert adapter._model_id == "Qwen/Qwen3-ASR-0.6B"
            assert adapter._model_ref == str(model_dir)

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
        aligner_instance.align.return_value = [
            [
                SimpleNamespace(text="hello", start_time=0.0, end_time=0.5, confidence=0.99),
                SimpleNamespace(text="world", start_time=0.5, end_time=1.0, confidence=0.98),
            ]
        ]
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

    def test_trim_drops_aligner_and_next_aligned_transcribe_reloads(self):
        torch_mock = _mock_torch()
        qwen_asr_module, model_cls, forced_aligner_cls = _mock_qwen_asr_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.transcribe.return_value = [SimpleNamespace(text="hello world", language="English")]
        model_cls.from_pretrained.return_value = model_instance

        aligner_instance = MagicMock()
        aligner_instance.align.return_value = [
            [
                SimpleNamespace(text="hello", start_time=0.0, end_time=0.5, confidence=0.99),
            ]
        ]
        forced_aligner_cls.from_pretrained.return_value = aligner_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": qwen_asr_module, "qwen_tts": MagicMock()}):
            from vox_qwen.asr_adapter import Qwen3ASRAdapter

            adapter = Qwen3ASRAdapter()
            with patch("vox_qwen.asr_adapter._supports_flash_attention", return_value=True):
                adapter.load("local-path", "cuda", _source="Qwen/Qwen3-ASR-0.6B")

            adapter.transcribe(np.ones(16000, dtype=np.float32), word_timestamps=True)
            assert adapter._aligner is aligner_instance
            assert forced_aligner_cls.from_pretrained.call_count == 1

            adapter.trim()
            assert adapter._aligner is None

            adapter.transcribe(np.ones(16000, dtype=np.float32), word_timestamps=True)
            assert adapter._aligner is aligner_instance
            assert forced_aligner_cls.from_pretrained.call_count == 2

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

            assert info.name == "qwen3-tts-torch"
            assert info.type == ModelType.TTS
            assert "qwen3-tts" in info.architectures
            assert info.default_sample_rate == 24000
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_streaming is True
            assert info.supports_voice_cloning is True
            assert "ru" in info.supported_languages

    def test_load_base_checkpoint_sets_clone_mode(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = []
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter.load("local-path", "cuda", _source="Qwen/Qwen3-TTS-12Hz-0.6B-Base")

            assert adapter._mode == "clone"
            assert adapter.is_loaded is True

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
            assert adapter._model_ref == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    def test_tts_load_prefers_local_model_path_when_present(self, tmp_path: Path):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["Ryan"]
        model_cls.from_pretrained.return_value = model_instance
        model_dir = tmp_path / "qwen-tts"
        model_dir.mkdir()

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter.load(
                str(model_dir),
                "cuda",
                _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                default_voice="Ryan",
            )

            model_cls.from_pretrained.assert_called_once()
            assert model_cls.from_pretrained.call_args.args[0] == str(model_dir)
            assert adapter._model_id == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            assert adapter._model_ref == str(model_dir)

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

    def test_load_uses_faster_qwen_backend_when_cuda_runtime_available(self):
        torch_mock = _mock_torch()
        faster_model = _mock_faster_qwen_model()
        faster_cls = MagicMock()
        faster_cls.from_pretrained.return_value = faster_model

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            with (
                patch("vox_qwen.tts_adapter._load_faster_qwen_tts_model", return_value=faster_cls),
                patch("vox_qwen.tts_adapter._supports_flash_attention", return_value=False),
            ):
                adapter.load(
                    "local-path",
                    "cuda",
                    _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    default_voice="Ryan",
                )

            faster_cls.from_pretrained.assert_called_once()
            kwargs = faster_cls.from_pretrained.call_args.kwargs
            assert kwargs["device"] == "cuda:0"
            assert kwargs["dtype"] is torch_mock.bfloat16
            assert kwargs["attn_implementation"] == "sdpa"
            assert kwargs["backend"] == "torch"
            assert adapter._backend == "faster-qwen3-tts"
            assert adapter._model is faster_model
            # The fast path must reflect the model's real speaker list, not just
            # the catalog default, so built-in speakers stay usable.
            assert adapter._supported_speakers == ["Ryan", "Aiden", "Sohee"]

    def test_load_falls_back_to_official_backend_when_faster_backend_fails(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["Ryan"]
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            with (
                patch("vox_qwen.tts_adapter._load_faster_qwen_tts_model", side_effect=RuntimeError("fast missing")),
                patch("vox_qwen.tts_adapter._supports_flash_attention", return_value=False),
            ):
                adapter.load(
                    "local-path",
                    "cuda",
                    _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    default_voice="Ryan",
                )

            model_cls.from_pretrained.assert_called_once()
            assert adapter._backend == "qwen-tts"
            assert adapter._model is model_instance

    def test_synthesize_custom_streams_from_faster_backend(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._backend = "faster-qwen3-tts"
            adapter._model = _mock_faster_qwen_model()
            adapter._mode = "custom"
            adapter._default_voice = "Ryan"
            adapter._supported_speakers = ["Ryan"]

            async def run():
                chunks = []
                async for chunk in adapter.synthesize("Hello", language="en"):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(run())

            adapter._model.generate_custom_voice_streaming.assert_called_once()
            kwargs = adapter._model.generate_custom_voice_streaming.call_args.kwargs
            assert kwargs["speaker"] == "Ryan"
            assert kwargs["language"] == "English"
            assert kwargs["chunk_size"] == 8
            assert chunks[-1].is_final is True
            assert len([chunk for chunk in chunks if chunk.audio]) == 2

    def test_synthesize_custom_passes_params_to_faster_backend(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._backend = "faster-qwen3-tts"
            adapter._model = _mock_faster_qwen_model()
            adapter._mode = "custom"
            adapter._default_voice = "Ryan"
            adapter._supported_speakers = ["Ryan"]

            async def run():
                async for _ in adapter.synthesize(
                    "Hello",
                    language="en",
                    params={"chunk_size": 12, "seed": 123},
                ):
                    pass

            asyncio.run(run())

            kwargs = adapter._model.generate_custom_voice_streaming.call_args.kwargs
            assert kwargs["chunk_size"] == 12

    def test_synthesize_clone_streams_from_faster_backend_with_temp_reference_wav(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._backend = "faster-qwen3-tts"
            adapter._model = _mock_faster_qwen_model()
            adapter._mode = "clone"

            ref = np.linspace(0.0, 0.5, num=24_000, dtype=np.float32)

            async def run():
                chunks = []
                async for chunk in adapter.synthesize(
                    "Hello",
                    language="en",
                    reference_audio=ref,
                    reference_text="reference text",
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(run())

            adapter._model.generate_voice_clone_streaming.assert_called_once()
            kwargs = adapter._model.generate_voice_clone_streaming.call_args.kwargs
            assert kwargs["language"] == "English"
            assert kwargs["ref_text"] == "reference text"
            assert kwargs["chunk_size"] == 8
            assert isinstance(kwargs["ref_audio"], str)
            assert chunks[-1].is_final is True
            assert any(chunk.audio for chunk in chunks[:-1])

    def test_synthesize_clone_passes_chunk_size_to_faster_backend(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._backend = "faster-qwen3-tts"
            adapter._model = _mock_faster_qwen_model()
            adapter._mode = "clone"

            ref = np.linspace(0.0, 0.5, num=24_000, dtype=np.float32)

            async def run():
                async for _ in adapter.synthesize(
                    "Hello",
                    language="en",
                    reference_audio=ref,
                    params={"chunk_size": 16, "seed": 321},
                ):
                    pass

            asyncio.run(run())

            kwargs = adapter._model.generate_voice_clone_streaming.call_args.kwargs
            assert kwargs["chunk_size"] == 16

    def test_synthesize_clone_falls_back_when_faster_backend_hits_cuda_graph_error(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._backend = "faster-qwen3-tts"
            adapter._model = _mock_faster_qwen_model()
            adapter._mode = "clone"
            adapter._model.generate_voice_clone_streaming.side_effect = RuntimeError(
                "Offset increment outside graph capture encountered unexpectedly"
            )

            async def fake_subprocess(**kwargs):
                yield kwargs
                yield type("chunk", (), {"audio": b"", "is_final": True})()

            ref = np.linspace(0.0, 0.5, num=24_000, dtype=np.float32)

            with patch.object(adapter, "_stream_subprocess", side_effect=fake_subprocess) as subprocess:

                async def run():
                    chunks = []
                    async for chunk in adapter.synthesize("Hello", language="en", reference_audio=ref):
                        chunks.append(chunk)
                    return chunks

                chunks = asyncio.run(run())

            subprocess.assert_called_once()
            assert chunks[0]["mode"] == "clone"
            assert chunks[0]["language"] == "English"
            assert chunks[-1].is_final is True

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

    def test_synthesize_matches_supported_speaker_case_insensitively(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["ryan", "aiden"]
        model_instance.generate_custom_voice.return_value = ([np.array([0.0, 0.25, 0.5], dtype=np.float32)], 24000)
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter.load(
                "local-path",
                "cuda",
                _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                default_voice="Ryan",
            )

            async def run():
                async for _ in adapter.synthesize("Hello", voice="Ryan", language="en"):
                    pass

            asyncio.run(run())

            kwargs = model_instance.generate_custom_voice.call_args.kwargs
            assert kwargs["speaker"] == "ryan"

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

    def test_synthesize_clone_requires_reference_audio(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = []
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter.load("local-path", "cuda", _source="Qwen/Qwen3-TTS-12Hz-0.6B-Base")

            async def run():
                async for _ in adapter.synthesize("Hello", language="en"):
                    pass

            with pytest.raises(ValueError, match="reference_audio"):
                asyncio.run(run())

    def test_preflight_clone_requires_reference_audio(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._mode = "clone"

            with pytest.raises(InvalidConfigError, match="require reference_audio"):
                adapter.validate_synthesis_request()

    def test_synthesize_clone_calls_generate_voice_clone(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = []
        model_instance.generate_voice_clone.return_value = (
            [np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)],
            24000,
        )
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter.load("local-path", "cuda", _source="Qwen/Qwen3-TTS-12Hz-0.6B-Base")

            ref = np.linspace(0.0, 0.5, num=24_000, dtype=np.float32)

            async def run():
                out = []
                async for chunk in adapter.synthesize(
                    "Hello",
                    language="en",
                    reference_audio=ref,
                    reference_text="quick brown fox",
                ):
                    out.append(chunk)
                return out

            chunks = asyncio.run(run())

            model_instance.generate_voice_clone.assert_called_once()
            kwargs = model_instance.generate_voice_clone.call_args.kwargs
            assert kwargs["language"] == "English"
            assert kwargs["ref_text"] == "quick brown fox"
            ref_audio_tuple = kwargs["ref_audio"]
            assert isinstance(ref_audio_tuple, tuple)
            np.testing.assert_array_equal(ref_audio_tuple[0], ref)
            assert ref_audio_tuple[1] == 24000
            assert chunks[-1].is_final is True
            assert any(c.audio for c in chunks[:-1])

    def test_synthesize_custom_rejects_reference_audio(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = ["Ryan"]
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter.load(
                "local-path",
                "cuda",
                _source="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                default_voice="Ryan",
            )

            ref = np.zeros(1000, dtype=np.float32)

            async def run():
                async for _ in adapter.synthesize("hi", reference_audio=ref):
                    pass

            with pytest.raises(ValueError, match="CustomVoice checkpoints do not use reference_audio"):
                asyncio.run(run())

    def test_preflight_custom_rejects_reference_audio(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._mode = "custom"

            with pytest.raises(InvalidConfigError, match="do not use reference_audio"):
                adapter.validate_synthesis_request(reference_audio=np.ones(16_000, dtype=np.float32))

    def test_load_mode_override_forces_clone(self):
        torch_mock = _mock_torch()
        qwen_tts_module, model_cls = _mock_qwen_tts_module()
        model_instance = MagicMock()
        model_instance.processor = MagicMock()
        model_instance.get_supported_speakers.return_value = []
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict("sys.modules", {"torch": torch_mock, "qwen_asr": MagicMock(), "qwen_tts": qwen_tts_module}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()

            adapter.load("local-path", "cuda", _source="unrelated/some-ckpt", mode="clone")
            assert adapter._mode == "clone"

    def test_synthesize_clone_subprocess_writes_reference_wav(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._subprocess_only = True
            adapter._mode = "clone"
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            adapter._device = "cpu"

            audio_bytes = np.zeros(2400, dtype=np.float32).tobytes()
            payload = json.dumps(
                {
                    "sample_rate": 24000,
                    "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                }
            )

            ref_paths_seen = []

            def fake_popen(cmd, **_):
                args = dict(zip(cmd[::2], cmd[1::2], strict=False))
                ref_paths_seen.append(args.get("--ref-audio-path"))
                assert "--mode" in cmd
                assert cmd[cmd.index("--mode") + 1] == "clone"
                return _completed_subprocess(payload)

            async def collect():
                out = []
                async for chunk in adapter.synthesize(
                    "hi",
                    reference_audio=np.zeros(24000, dtype=np.float32),
                    reference_text="hi",
                ):
                    out.append(chunk)
                return out

            with patch("vox_qwen.tts_adapter.subprocess.Popen", side_effect=fake_popen):
                chunks = asyncio.run(collect())

            assert chunks[-1].is_final is True
            assert ref_paths_seen and ref_paths_seen[0] is not None

    def test_subprocess_fallback_receives_seed(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._subprocess_only = True
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            adapter._default_voice = "Ryan"
            adapter._device = "cpu"

            audio = np.zeros(24_000, dtype=np.float32).tobytes()
            payload = json.dumps(
                {
                    "sample_rate": 24_000,
                    "audio_b64": base64.b64encode(audio).decode("ascii"),
                }
            )

            def fake_popen(cmd, **_):
                assert "--seed" in cmd
                assert cmd[cmd.index("--seed") + 1] == "777"
                return _completed_subprocess(payload)

            async def collect():
                async for _ in adapter.synthesize("hello", params={"seed": 777}):
                    pass

            with patch("vox_qwen.tts_adapter.subprocess.Popen", side_effect=fake_popen):
                asyncio.run(collect())

    def test_subprocess_fallback_keeps_private_text_out_of_process_arguments(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._subprocess_only = True
            adapter._mode = "clone"
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            adapter._device = "cpu"

            audio = np.zeros(24_000, dtype=np.float32).tobytes()
            payload = json.dumps(
                {
                    "sample_rate": 24_000,
                    "audio_b64": base64.b64encode(audio).decode("ascii"),
                }
            )
            process = _completed_subprocess(payload)
            private_text = "private synthesis text"
            private_reference = "private reference transcript"

            def fake_popen(cmd, **_):
                assert private_text not in cmd
                assert private_reference not in cmd
                return process

            async def collect():
                async for _ in adapter.synthesize(
                    private_text,
                    reference_audio=np.zeros(24_000, dtype=np.float32),
                    reference_text=private_reference,
                ):
                    pass

            with patch("vox_qwen.tts_adapter.subprocess.Popen", side_effect=fake_popen):
                asyncio.run(collect())

            assert json.loads(process.stdin_payload or "{}") == {
                "text": private_text,
                "reference_text": private_reference,
            }

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

    def test_subprocess_fallback_normalizes_default_voice_case(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._subprocess_only = True
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            adapter._default_voice = "Ryan"
            adapter._supported_speakers = ["ryan"]
            adapter._device = "cpu"

            audio = np.zeros(24_000, dtype=np.float32).tobytes()
            payload = json.dumps(
                {
                    "sample_rate": 24_000,
                    "audio_b64": base64.b64encode(audio).decode("ascii"),
                }
            )

            async def collect():
                chunks = []
                async for chunk in adapter.synthesize("hello"):
                    chunks.append(chunk)
                return chunks

            completed = _completed_subprocess(payload)
            with patch("vox_qwen.tts_adapter.subprocess.Popen", return_value=completed):
                chunks = asyncio.run(collect())

            assert any(chunk.audio for chunk in chunks)
            assert chunks[-1].is_final is True

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
            payload = json.dumps(
                {
                    "sample_rate": 24_000,
                    "audio_b64": base64.b64encode(audio).decode("ascii"),
                }
            )

            async def collect():
                chunks = []
                async for chunk in adapter.synthesize("hello"):
                    chunks.append(chunk)
                return chunks

            completed = _completed_subprocess(payload)
            with patch("vox_qwen.tts_adapter.subprocess.Popen", return_value=completed):
                chunks = asyncio.run(collect())

            assert any(chunk.audio for chunk in chunks)
            assert chunks[-1].is_final is True

    def test_cancelled_subprocess_fallback_is_terminated_and_reaped(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._subprocess_only = True
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            adapter._device = "cpu"
            payload = json.dumps({"sample_rate": 24_000, "audio_b64": ""})
            process = _BlockingSubprocess(payload)

            async def run():
                task = asyncio.create_task(
                    adapter._synthesize_via_subprocess(
                        mode="custom",
                        text="hello",
                        language="en",
                    )
                )
                assert await asyncio.to_thread(process.started.wait, 1.0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert process.terminated.is_set()
                assert process.reaped.is_set()
                assert adapter._active_subprocesses == set()

            def kill_process_group(_, sig):
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()

            try:
                with (
                    patch("vox_qwen.tts_adapter.subprocess.Popen", return_value=process),
                    patch("vox_qwen.tts_adapter.subprocess.run", side_effect=process.run),
                    patch("vox_qwen.tts_adapter.os.killpg", side_effect=kill_process_group),
                ):
                    asyncio.run(run())
            finally:
                process.terminated.set()

    def test_unload_terminates_active_subprocess_fallback(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_asr": MagicMock(), "qwen_tts": MagicMock()}):
            from vox_qwen.tts_adapter import Qwen3TTSAdapter

            adapter = Qwen3TTSAdapter()
            adapter._loaded = True
            adapter._subprocess_only = True
            adapter._model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            adapter._device = "cpu"
            payload = json.dumps({"sample_rate": 24_000, "audio_b64": ""})
            process = _BlockingSubprocess(payload)

            async def run():
                task = asyncio.create_task(
                    adapter._synthesize_via_subprocess(
                        mode="custom",
                        text="hello",
                        language="en",
                    )
                )
                assert await asyncio.to_thread(process.started.wait, 1.0)
                await asyncio.to_thread(adapter.unload)
                assert process.terminated.is_set()
                assert process.reaped.is_set()
                assert adapter._active_subprocesses == set()
                with pytest.raises(RuntimeError, match="subprocess failed"):
                    await task

            def kill_process_group(_, sig):
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()

            try:
                with (
                    patch("vox_qwen.tts_adapter.subprocess.Popen", return_value=process),
                    patch("vox_qwen.tts_adapter.subprocess.run", side_effect=process.run),
                    patch("vox_qwen.tts_adapter.os.killpg", side_effect=kill_process_group),
                ):
                    asyncio.run(run())
            finally:
                process.terminated.set()

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
            runtime.ensure_runtime("qwen-tts", "qwen-tts", "qwen_tts", no_deps=True)

        assert calls[0][0:2] == ["uv", "pip"]
        assert "--upgrade" in calls[0]
        assert "--no-deps" in calls[0]

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
            assert ensure_runtime.call_args.kwargs["no_deps"] is True
            assert ensure_runtime.call_args.kwargs["extra_packages"] == (
                "onnxruntime>=1.20,<2",
                "sox",
                "einops",
            )

    def test_faster_qwen_runtime_installs_into_qwen_tts_runtime(self):
        faster_module = ModuleType("faster_qwen3_tts")
        faster_module.FasterQwen3TTS = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "torch": _mock_torch(),
                "qwen_asr": MagicMock(),
                "qwen_tts": MagicMock(),
                "faster_qwen3_tts": faster_module,
            },
        ):
            from vox_qwen import tts_adapter

            with patch.object(tts_adapter, "ensure_runtime") as ensure_runtime:
                loaded = tts_adapter._load_faster_qwen_tts_model()

            assert loaded is faster_module.FasterQwen3TTS
            ensure_runtime.assert_called_once()
            assert ensure_runtime.call_args.args[:3] == (
                "qwen-tts",
                "qwen-tts",
                "faster_qwen3_tts",
            )
            assert ensure_runtime.call_args.kwargs["no_deps"] is True
            assert "faster-qwen3-tts>=0.2.6" in ensure_runtime.call_args.kwargs["extra_packages"]
            assert "onnxruntime>=1.20,<2" in ensure_runtime.call_args.kwargs["extra_packages"]
            assert "qwen_tts" in ensure_runtime.call_args.kwargs["required_imports"]

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
            assert ensure_runtime.call_args.kwargs["no_deps"] is True
            assert ensure_runtime.call_args.kwargs["extra_packages"] == (
                "qwen-omni-utils",
                "DyNet38==2.2",
                "nagisa==0.2.11",
                "soynlp==0.0.493",
                "librosa",
                "soundfile",
                "sox",
            )
            assert ensure_runtime.call_args.kwargs["required_imports"] == ("dynet_config",)

    def test_qwen_asr_model_stub_avoids_forced_aligner_import_for_plain_transcription(self):
        with patch.dict("sys.modules", {"torch": _mock_torch(), "qwen_tts": MagicMock()}, clear=False):
            from vox_qwen import asr_adapter

            qwen_asr_module = ModuleType("qwen_asr")
            inference_module = ModuleType("qwen_asr.inference")
            qwen3_asr_module = ModuleType("qwen_asr.inference.qwen3_asr")

            model_cls = MagicMock()
            qwen3_asr_module.Qwen3ASRModel = model_cls

            with (
                patch.object(asr_adapter, "ensure_runtime"),
                patch.dict(
                    "sys.modules",
                    {
                        "qwen_asr": qwen_asr_module,
                        "qwen_asr.inference": inference_module,
                        "qwen_asr.inference.qwen3_asr": qwen3_asr_module,
                    },
                    clear=False,
                ),
            ):
                loaded = asr_adapter._load_qwen_asr_model()

            assert loaded is model_cls
