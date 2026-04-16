from __future__ import annotations

from pathlib import Path

import yaml


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


def test_write_stage_config_uses_stage_specific_spark_defaults(tmp_path: Path, monkeypatch):
    from vox_voxtral import runtime as runtime_module

    purelib = tmp_path / "purelib"
    _write_source_stage_config(purelib)

    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))
    monkeypatch.setattr(runtime_module, "_purelib", lambda python_bin: purelib)

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
