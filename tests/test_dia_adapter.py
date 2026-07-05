from __future__ import annotations

import asyncio
import builtins
import importlib
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from vox.core.types import ModelFormat, ModelType


def test_dia_package_metadata_keeps_torch_out_of_adapter_dependencies():
    pyproject = Path(__file__).parents[1] / "adapters" / "vox-dia" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    dependencies = data["project"]["dependencies"]
    assert data["project"]["version"] == "0.2.10"
    assert not any(dep.startswith("torch") for dep in dependencies)


class TestDiaAdapterInfo:
    def test_package_import_is_light_without_torch(self, monkeypatch):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("vox-dia imported torch during package import")
            return original_import(name, *args, **kwargs)

        sys.modules.pop("vox_dia", None)
        sys.modules.pop("vox_dia.adapter", None)
        monkeypatch.setattr(builtins, "__import__", guarded_import)

        module = importlib.import_module("vox_dia")

        assert module.__all__ == ["DiaAdapter"]

    def test_package_import_does_not_require_transformers_dia_class(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            sys.modules.pop("vox_dia", None)
            module = importlib.import_module("vox_dia")
            assert module.__all__ == ["DiaAdapter"]

    def test_info_returns_correct_metadata(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from vox_dia.adapter import DiaAdapter

            adapter = DiaAdapter()
            info = adapter.info()

            assert info.name == "dia-tts-torch"
            assert info.type == ModelType.TTS
            assert info.architectures == ("dia-tts-torch", "dia")
            assert info.default_sample_rate == 44100
            assert ModelFormat.PYTORCH in info.supported_formats
            assert info.supports_streaming is False
            assert info.supports_voice_cloning is False
            assert info.supported_languages == ("en",)

    def test_is_loaded_initially_false(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from vox_dia.adapter import DiaAdapter

            adapter = DiaAdapter()
            assert adapter.is_loaded is False

    def test_load_rejects_cpu(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from vox_dia.adapter import DiaAdapter

            adapter = DiaAdapter()

            with pytest.raises(RuntimeError, match="CUDA-capable GPU"):
                adapter.load("nari-labs/Dia-1.6B", "cpu")

    def test_load_requires_transformers_main_branch(self):
        torch = MagicMock()
        with patch.dict("sys.modules", {"torch": torch}):
            from vox_dia.adapter import DiaAdapter

            with patch(
                "vox_dia.adapter._load_transformers_runtime",
                side_effect=RuntimeError("Dia requires Hugging Face Transformers"),
            ):
                adapter = DiaAdapter()
                with pytest.raises(RuntimeError, match="Dia requires Hugging Face Transformers"):
                    adapter.load("nari-labs/Dia-1.6B", "cuda")

    def test_prepare_runtime_verifies_transformers_runtime(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from vox_dia.adapter import DiaAdapter

            with patch("vox_dia.adapter._load_transformers_runtime") as load_runtime:
                DiaAdapter().prepare_runtime()

            load_runtime.assert_called_once_with()

    def test_runtime_install_does_not_upgrade_moving_transformers_source(self, tmp_path: Path, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], timeout: int):
            calls.append(cmd)
            if any(str(part).startswith("git+https://github.com/huggingface/transformers.git") for part in cmd):
                runtime = tmp_path / "vox-home" / "runtime" / "dia"
                (runtime / "transformers" / "models" / "dia").mkdir(parents=True)
                (runtime / "transformers" / "__init__.py").write_text(
                    "class AutoProcessor: pass\nclass DiaForConditionalGeneration: pass\n"
                )
                (runtime / "transformers" / "models" / "__init__.py").write_text("")
                (runtime / "transformers" / "models" / "dia" / "__init__.py").write_text("")
                (runtime / "transformers" / "models" / "dia" / "modeling_dia.py").write_text("")
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from vox_dia.adapter import _install_transformers_runtime

            with patch("vox_dia.adapter._run_install_command", side_effect=fake_run):
                _install_transformers_runtime()

        source_call = next(
            call for call in calls
            if any(str(part).startswith("git+https://github.com/huggingface/transformers.git") for part in call)
        )
        deps_call = next(call for call in calls if "sentencepiece>=0.2.0,<0.3" in call)

        assert "--no-deps" in source_call
        assert "--upgrade" not in source_call
        assert "--upgrade" in deps_call
        assert "tiktoken>=0.9.0,<1" in deps_call

    def test_load_bootstraps_transformers_when_dia_symbol_is_missing(self):
        torch = MagicMock()
        module = ModuleType("transformers")
        module.__path__ = []  # type: ignore[attr-defined]
        processor_cls = MagicMock()
        processor = MagicMock()
        processor_cls.from_pretrained.return_value = processor
        module.AutoProcessor = processor_cls
        module._bootstrapped = False  # type: ignore[attr-defined]

        def module_getattr(name: str):
            if name == "DiaForConditionalGeneration" and not getattr(module, "_bootstrapped", False):
                raise ImportError("DiaForConditionalGeneration not available")
            raise AttributeError(name)

        module.__getattr__ = module_getattr  # type: ignore[attr-defined]

        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "transformers": module,
                "transformers.models": ModuleType("transformers.models"),
                "transformers.models.dia": ModuleType("transformers.models.dia"),
            },
        ):
            from vox_dia.adapter import DiaAdapter

            model_cls = MagicMock()
            model = MagicMock()
            model.to.return_value = model
            model_cls.from_pretrained.return_value = model

            def install_side_effect():
                module._bootstrapped = True  # type: ignore[attr-defined]
                module.DiaForConditionalGeneration = model_cls  # type: ignore[attr-defined]
                module.AutoProcessor = processor_cls  # type: ignore[attr-defined]
                sys.modules["transformers"] = module

            with (
                patch("vox_dia.adapter._install_transformers_runtime", side_effect=install_side_effect),
                patch("vox_dia.adapter._clear_transformers_modules"),
                patch("vox_dia.adapter._runtime_has_dia_support", side_effect=[False, True]),
            ):
                adapter = DiaAdapter()
                adapter.load("nari-labs/Dia-1.6B", "cuda")

            processor_cls.from_pretrained.assert_called_once_with("nari-labs/Dia-1.6B")
            model_cls.from_pretrained.assert_called_once_with("nari-labs/Dia-1.6B")
            model.to.assert_called_once_with("cuda")
            model.eval.assert_called_once()

    def test_load_puts_model_in_eval_mode(self):
        torch = MagicMock()
        with patch.dict("sys.modules", {"torch": torch}):
            from vox_dia.adapter import DiaAdapter

            processor_cls = MagicMock()
            processor = MagicMock()
            processor_cls.from_pretrained.return_value = processor
            model = MagicMock()
            model.to.return_value = model
            model_cls = MagicMock()
            model_cls.from_pretrained.return_value = model

            with patch(
                "vox_dia.adapter._load_transformers_runtime",
                return_value=(processor_cls, model_cls),
            ):
                adapter = DiaAdapter()
                adapter.load("nari-labs/Dia-1.6B", "cuda")

            processor_cls.from_pretrained.assert_called_once_with("nari-labs/Dia-1.6B")
            model_cls.from_pretrained.assert_called_once_with("nari-labs/Dia-1.6B")
            model.to.assert_called_once_with("cuda")
            model.eval.assert_called_once()

    def test_load_prefers_local_model_path_when_present(self, tmp_path: Path):
        torch = MagicMock()
        with patch.dict("sys.modules", {"torch": torch}):
            from vox_dia.adapter import DiaAdapter

            processor_cls = MagicMock()
            processor = MagicMock()
            processor_cls.from_pretrained.return_value = processor
            model = MagicMock()
            model.to.return_value = model
            model_cls = MagicMock()
            model_cls.from_pretrained.return_value = model
            model_dir = tmp_path / "dia"
            model_dir.mkdir()

            with patch(
                "vox_dia.adapter._load_transformers_runtime",
                return_value=(processor_cls, model_cls),
            ):
                adapter = DiaAdapter()
                adapter.load(str(model_dir), "cuda", _source="nari-labs/Dia-1.6B")

            processor_cls.from_pretrained.assert_called_once_with(str(model_dir))
            model_cls.from_pretrained.assert_called_once_with(str(model_dir))
            assert adapter._model_id == "nari-labs/Dia-1.6B"
            assert adapter._model_ref == str(model_dir)

    def test_synthesize_raises_when_not_loaded(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from vox_dia.adapter import DiaAdapter

            adapter = DiaAdapter()

            async def _run() -> None:
                async for _chunk in adapter.synthesize("hello"):
                    pass

            with pytest.raises(RuntimeError, match="not loaded"):
                asyncio.run(_run())

    def test_synthesize_streams_audio_from_saved_output(self):
        torch = MagicMock()
        sf = MagicMock()
        sf.read.return_value = (np.array([0.1, -0.1], dtype=np.float32), 44100)

        with patch.dict("sys.modules", {"torch": torch, "soundfile": sf}):
            from vox_dia.adapter import DiaAdapter

            processor = MagicMock()
            processor.text = None
            processor.batch_decode.return_value = ["decoded"]
            processor.save_audio.return_value = None
            model = MagicMock()
            model.generate.return_value = MagicMock()

            adapter = DiaAdapter()
            adapter._loaded = True
            adapter._processor = processor
            adapter._model = model
            adapter._device = "cuda"

            async def _run() -> list:
                chunks = []
                async for chunk in adapter.synthesize("hello [S1] world"):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

            model.generate.assert_called_once()
            processor.save_audio.assert_called_once_with(["decoded"], ANY)
            assert len(chunks) == 2
            assert chunks[0].sample_rate == 44100
            assert chunks[0].is_final is False
            assert chunks[1].is_final is True

    def test_estimate_vram(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from vox_dia.adapter import DiaAdapter

            adapter = DiaAdapter()
            assert adapter.estimate_vram_bytes() == 10_000_000_000

    def test_unload_does_not_require_torch_when_not_loaded(self, monkeypatch):
        from vox_dia.adapter import DiaAdapter

        adapter = DiaAdapter()
        monkeypatch.setattr("vox_dia.adapter._torch_module", MagicMock(side_effect=RuntimeError("no torch")))

        adapter.unload()
