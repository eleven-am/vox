from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import yaml
from packaging.requirements import Requirement


def _write_source_stage_config(root: Path) -> None:
    source = root / "vllm_omni" / "model_executor" / "stage_configs"
    source.mkdir(parents=True, exist_ok=True)
    config = {
        "stage_args": [
            {
                "engine_args": {
                    "model_stage": "audio_generation",
                    "gpu_memory_utilization": 0.8,
                    "max_num_seqs": 32,
                    "max_model_len": 4096,
                }
            },
            {
                "engine_args": {
                    "model_stage": "audio_tokenizer",
                    "gpu_memory_utilization": 0.1,
                    "max_num_seqs": 32,
                    "max_num_batched_tokens": 65536,
                    "max_model_len": 65536,
                }
            },
        ]
    }
    (source / "voxtral_tts.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_voxtral_package_metadata_uses_compatible_transformers_and_tokenizers_ranges():
    pyproject = Path(__file__).resolve().parents[1] / "adapters" / "vox-voxtral" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    requirements = {
        Requirement(raw_requirement).name: Requirement(raw_requirement)
        for raw_requirement in data["project"]["dependencies"]
    }

    assert str(requirements["transformers"].specifier) == "<4.58,>=4.57.6"
    assert str(requirements["tokenizers"].specifier) == "<0.24,>=0.22"


def test_has_gpu_torch_requires_runtime_local_torch_lib(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    runtime_purelib = tmp_path / "runtime-purelib"
    runtime_purelib.mkdir(parents=True)
    run_mock = MagicMock()

    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: runtime_purelib)
    monkeypatch.setattr(runtime_module, "_run", run_mock)

    assert runtime_module._has_gpu_torch(Path("/tmp/fake-python")) is False
    run_mock.assert_not_called()


def test_has_vllm_omni_runtime_requires_runtime_local_stage_config(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    runtime_purelib = tmp_path / "runtime-purelib"
    (runtime_purelib / "vllm").mkdir(parents=True)
    app_purelib = tmp_path / "app-purelib" / "vllm_omni" / "model_executor" / "stage_configs"
    app_purelib.mkdir(parents=True, exist_ok=True)
    (app_purelib / "voxtral_tts.yaml").write_text("stage_args: []\n", encoding="utf-8")

    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: runtime_purelib)

    assert runtime_module._has_vllm_omni_runtime(
        Path("/tmp/fake-python"),
        extra_pythonpaths=[str(tmp_path / "app-purelib")],
    ) is False

    runtime_stage_config = runtime_purelib / "vllm_omni" / "model_executor" / "stage_configs"
    runtime_stage_config.mkdir(parents=True, exist_ok=True)
    (runtime_stage_config / "voxtral_tts.yaml").write_text("stage_args: []\n", encoding="utf-8")

    assert runtime_module._has_vllm_omni_runtime(Path("/tmp/fake-python")) is True


def test_build_env_can_exclude_app_torch_lib(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    runtime_purelib = tmp_path / "runtime-purelib"
    app_purelib = tmp_path / "app-purelib"
    (runtime_purelib / "torch" / "lib").mkdir(parents=True)
    (app_purelib / "torch" / "lib").mkdir(parents=True)

    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: runtime_purelib)
    monkeypatch.setattr(runtime_module, "_app_purelib", lambda: app_purelib)

    env = runtime_module._build_env(
        Path("/tmp/fake-python"),
        include_app_torch_lib=False,
    )

    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(runtime_purelib / "torch" / "lib")
    assert str(app_purelib / "torch" / "lib") not in env["LD_LIBRARY_PATH"].split(":")


def test_gpu_torch_index_url_defaults_to_cu129_for_arm64_cuda_12_6(monkeypatch):
    from vox_voxtral import runtime as runtime_module

    class _TorchVersion:
        cuda = "12.6"

    fake_torch = ModuleType("torch")
    fake_torch.version = _TorchVersion()

    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "aarch64")
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert runtime_module._gpu_torch_index_url() == "https://download.pytorch.org/whl/cu129"


def test_ensure_voxtral_stt_runtime_bootstraps_runtime_dependencies(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    vox_home = tmp_path / "vox-home"
    runtime_dir = vox_home / "runtime" / "voxtral-stt"
    app_purelib = tmp_path / "app-purelib"
    app_purelib.mkdir(parents=True)

    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return MagicMock(returncode=0, stderr="", stdout="")

    monkeypatch.setenv("VOX_HOME", str(vox_home))
    monkeypatch.setattr(runtime_module, "_app_purelib", lambda: app_purelib)
    monkeypatch.setattr(runtime_module, "_module_available", lambda name: False)
    monkeypatch.setattr(
        runtime_module,
        "_pinned_stt_dependency_versions",
        lambda: {
            "transformers": "4.57.6",
            "tokenizers": "0.22.2",
            "huggingface_hub": "0.36.2",
        },
    )
    monkeypatch.setattr(runtime_module, "_run", fake_run)

    runtime_path = runtime_module.ensure_voxtral_stt_runtime()

    assert runtime_path == str(runtime_dir)
    assert str(runtime_dir) == sys.path[0]
    assert (runtime_dir / "_vox_runtime_fallback_paths.pth").read_text(encoding="utf-8") == f"{app_purelib}\n"
    assert commands == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(runtime_dir),
            "--upgrade",
            "transformers==4.57.6",
            "tokenizers==0.22.2",
            "huggingface-hub==0.36.2",
            "mistral-common[audio]>=1.10.0",
        ]
    ]


def test_stt_adapter_shims_huggingface_hub_offline_mode_for_transformers_compat():
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
        sys.modules.pop("vox_voxtral.stt_adapter", None)
        importlib.import_module("vox_voxtral.stt_adapter")

        assert hasattr(huggingface_hub, "is_offline_mode")
        assert huggingface_hub.is_offline_mode() is False
        assert hasattr(huggingface_hub_dataclasses, "validate_typed_dict")
        sentinel = object()
        assert huggingface_hub_dataclasses.validate_typed_dict(sentinel) is sentinel


def test_stt_adapter_shims_validate_typed_dict_even_if_offline_mode_already_exists():
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
        sys.modules.pop("vox_voxtral.stt_adapter", None)
        importlib.import_module("vox_voxtral.stt_adapter")

        assert huggingface_hub.is_offline_mode() is True
        assert hasattr(huggingface_hub_dataclasses, "validate_typed_dict")
        sentinel = object()
        assert huggingface_hub_dataclasses.validate_typed_dict(sentinel) is sentinel


def test_stt_adapter_wraps_existing_typed_dict_validator_that_rejects_union_types():
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
        sys.modules.pop("vox_voxtral.stt_adapter", None)
        importlib.import_module("vox_voxtral.stt_adapter")

        sentinel = object()
        assert huggingface_hub_dataclasses.validate_typed_dict(sentinel) is sentinel


def test_stt_adapter_wraps_hf_hub_download_without_tqdm_class_support():
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
        sys.modules.pop("vox_voxtral.stt_adapter", None)
        importlib.import_module("vox_voxtral.stt_adapter")

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


def test_stt_runtime_loader_falls_back_to_voxtral_submodule_when_top_level_export_is_missing():
    from vox_voxtral import stt_adapter as stt_adapter_module

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = object()
    voxtral_module = ModuleType("transformers.models.voxtral")
    voxtral_module.VoxtralForConditionalGeneration = object()

    with (
        patch.dict(
            "sys.modules",
            {
                "transformers": transformers,
                "transformers.models.voxtral": voxtral_module,
            },
        ),
        patch.object(stt_adapter_module, "ensure_voxtral_stt_runtime"),
    ):
        auto_processor, model_cls = stt_adapter_module._load_voxtral_stt_runtime()

    assert auto_processor is transformers.AutoProcessor
    assert model_cls is voxtral_module.VoxtralForConditionalGeneration


def test_write_stage_config_uses_stage_specific_spark_defaults(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    purelib = tmp_path / "purelib"
    _write_source_stage_config(purelib)

    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: purelib)
    monkeypatch.setattr(runtime_module, "_detected_total_gpu_memory_bytes", lambda: 80 * 1024**3)

    output = runtime_module._write_stage_config(Path("/tmp/fake-python"))
    config = yaml.safe_load(output.read_text(encoding="utf-8"))

    generation_args = config["stage_args"][0]["engine_args"]
    tokenizer_args = config["stage_args"][1]["engine_args"]

    assert generation_args["gpu_memory_utilization"] == 0.4
    assert generation_args["max_num_seqs"] == 1
    assert generation_args["enforce_eager"] is True
    assert generation_args["max_model_len"] == 2048

    assert tokenizer_args["gpu_memory_utilization"] == 0.1
    assert tokenizer_args["max_num_seqs"] == 1
    assert tokenizer_args["max_num_batched_tokens"] == 8192
    assert tokenizer_args["max_model_len"] == 8192


def test_write_stage_config_uses_triton_profile_on_16gb_gpu(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    purelib = tmp_path / "purelib"
    _write_source_stage_config(purelib)

    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: purelib)
    monkeypatch.setattr(runtime_module, "_detected_total_gpu_memory_bytes", lambda: 16 * 1024**3)

    output = runtime_module._write_stage_config(Path("/tmp/fake-python"))
    config = yaml.safe_load(output.read_text(encoding="utf-8"))

    generation_args = config["stage_args"][0]["engine_args"]
    tokenizer_args = config["stage_args"][1]["engine_args"]

    assert generation_args["gpu_memory_utilization"] == 0.9
    assert generation_args["max_model_len"] == 512
    assert generation_args["kv_cache_dtype"] == "fp8"
    assert generation_args["attention_backend"] == "triton_attn"
    assert "dtype" not in generation_args
    assert tokenizer_args["gpu_memory_utilization"] == 0.05
    assert tokenizer_args["max_num_batched_tokens"] == 4096
    assert tokenizer_args["max_model_len"] == 4096
    assert tokenizer_args["attention_backend"] == "triton_attn"
    assert "dtype" not in tokenizer_args


def test_write_stage_config_strips_inherited_half_dtype_on_16gb_gpu(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    purelib = tmp_path / "purelib"
    _write_source_stage_config(purelib)

    source = purelib / "vllm_omni" / "model_executor" / "stage_configs" / "voxtral_tts.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["stage_args"][0]["engine_args"]["dtype"] = "half"
    config["stage_args"][1]["engine_args"]["dtype"] = "half"
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: purelib)
    monkeypatch.setattr(runtime_module, "_detected_total_gpu_memory_bytes", lambda: 16 * 1024**3)

    output = runtime_module._write_stage_config(Path("/tmp/fake-python"))
    config = yaml.safe_load(output.read_text(encoding="utf-8"))

    generation_args = config["stage_args"][0]["engine_args"]
    tokenizer_args = config["stage_args"][1]["engine_args"]

    assert "dtype" not in generation_args
    assert "dtype" not in tokenizer_args


def test_write_stage_config_keeps_existing_small_gpu_defaults_on_24gb(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    purelib = tmp_path / "purelib"
    _write_source_stage_config(purelib)

    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: purelib)
    monkeypatch.setattr(runtime_module, "_detected_total_gpu_memory_bytes", lambda: 24 * 1024**3)

    output = runtime_module._write_stage_config(Path("/tmp/fake-python"))
    config = yaml.safe_load(output.read_text(encoding="utf-8"))

    generation_args = config["stage_args"][0]["engine_args"]
    tokenizer_args = config["stage_args"][1]["engine_args"]

    assert generation_args["gpu_memory_utilization"] == 0.4
    assert generation_args["max_model_len"] == 1536
    assert "attention_backend" not in generation_args
    assert "dtype" not in generation_args
    assert "kv_cache_dtype" not in generation_args
    assert tokenizer_args["gpu_memory_utilization"] == 0.05
    assert tokenizer_args["max_num_batched_tokens"] == 4096
    assert tokenizer_args["max_model_len"] == 4096
    assert "attention_backend" not in tokenizer_args


def test_write_stage_config_honors_stage_specific_env_overrides(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    purelib = tmp_path / "purelib"
    _write_source_stage_config(purelib)

    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    monkeypatch.setenv("VOX_VOXTRAL_GENERATION_GPU_MEMORY_UTILIZATION", "0.35")
    monkeypatch.setenv("VOX_VOXTRAL_TOKENIZER_GPU_MEMORY_UTILIZATION", "0.2")
    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: purelib)

    output = runtime_module._write_stage_config(Path("/tmp/fake-python"))
    config = yaml.safe_load(output.read_text(encoding="utf-8"))

    generation_args = config["stage_args"][0]["engine_args"]
    tokenizer_args = config["stage_args"][1]["engine_args"]

    assert generation_args["gpu_memory_utilization"] == 0.35
    assert tokenizer_args["gpu_memory_utilization"] == 0.2


def test_build_env_prepends_runtime_and_app_torch_lib_dirs(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    runtime_purelib = tmp_path / "runtime-purelib"
    app_purelib = tmp_path / "app-purelib"
    runtime_torch_lib = runtime_purelib / "torch" / "lib"
    app_torch_lib = app_purelib / "torch" / "lib"
    runtime_torch_lib.mkdir(parents=True)
    app_torch_lib.mkdir(parents=True)

    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: runtime_purelib)
    monkeypatch.setattr(runtime_module, "_app_purelib", lambda: app_purelib)
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/ld")

    env = runtime_module._build_env(
        Path("/tmp/fake-python"),
        extra_pythonpaths=["/extra/pythonpath"],
    )

    assert env["PYTHONPATH"] == "/extra/pythonpath:/existing/pythonpath"
    assert env["LD_LIBRARY_PATH"] == f"{runtime_torch_lib}:{app_torch_lib}:/existing/ld"


def test_build_env_includes_system_cuda_paths_when_present(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    runtime_purelib = tmp_path / "runtime-purelib"
    app_purelib = tmp_path / "app-purelib"
    runtime_torch_lib = runtime_purelib / "torch" / "lib"
    app_torch_lib = app_purelib / "torch" / "lib"
    runtime_torch_lib.mkdir(parents=True)
    app_torch_lib.mkdir(parents=True)

    real_isdir = runtime_module.os.path.isdir

    def fake_isdir(path: str) -> bool:
        if path in {"/usr/local/cuda/lib64", "/usr/local/cuda/targets/aarch64-linux/lib"}:
            return True
        return real_isdir(path)

    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: runtime_purelib)
    monkeypatch.setattr(runtime_module, "_app_purelib", lambda: app_purelib)
    monkeypatch.setattr(runtime_module.os.path, "isdir", fake_isdir)

    env = runtime_module._build_env(Path("/tmp/fake-python"))

    assert env["LD_LIBRARY_PATH"] == (
        f"{runtime_torch_lib}:{app_torch_lib}:/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib"
    )


def test_default_torchvision_version_handles_2_8():
    from vox_voxtral import runtime as runtime_module

    assert runtime_module._default_torchvision_version("2.8.0") == "0.23.0"


def test_gpu_torch_index_url_prefers_official_cu129_on_arm64(monkeypatch):
    from vox_voxtral import runtime as runtime_module

    class TorchVersion:
        cuda = "12.9"

    class Torch:
        version = TorchVersion()

    monkeypatch.delenv("VOX_VOXTRAL_TORCH_INDEX_URL", raising=False)
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setitem(__import__("sys").modules, "torch", Torch())

    assert runtime_module._gpu_torch_index_url() == "https://download.pytorch.org/whl/cu129"


def test_resolved_torch_runtime_version_adds_cu129_suffix_for_arm64(monkeypatch):
    from vox_voxtral import runtime as runtime_module

    class TorchVersion:
        cuda = "12.9"

    class Torch:
        version = TorchVersion()

    monkeypatch.delenv("VOX_VOXTRAL_TORCH_VERSION", raising=False)
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setitem(__import__("sys").modules, "torch", Torch())

    assert runtime_module._resolved_torch_runtime_version() == "2.10.0+cu129"
    assert runtime_module._resolved_torchaudio_runtime_version("2.10.0+cu129") == "2.10.0+cu129"
    assert runtime_module._resolved_torchvision_runtime_version("2.10.0+cu129") == "0.25.0+cu129"


def test_has_gpu_torch_accepts_runtime_local_cuda_wheel_even_if_arch_list_is_older(monkeypatch, tmp_path: Path):
    from vox_voxtral import runtime as runtime_module

    runtime_purelib = tmp_path / "runtime-purelib"
    (runtime_purelib / "torch" / "lib").mkdir(parents=True)
    payload = {
        "version": "2.8.0",
        "cuda": "12.9",
        "available": True,
        "file": str(runtime_purelib / "torch" / "__init__.py"),
    }

    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: runtime_purelib)
    monkeypatch.setattr(
        runtime_module,
        "_run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": yaml.safe_dump(payload)})(),
    )

    assert runtime_module._has_gpu_torch(tmp_path / "python") is True
