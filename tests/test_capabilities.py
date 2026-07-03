from __future__ import annotations

from vox.core.capabilities import (
    RuntimeRequirement,
    check_runtime_requirement,
    incompatible_pull_allowed,
    infer_model_requirements,
    missing_capabilities_for,
)
from vox.core.runtime import RuntimeCapabilities, detect_runtime_capabilities


def _entry(fmt: str, adapter: str = "some-adapter") -> dict:
    return {"format": fmt, "adapter": adapter}


def _caps(**overrides):
    values = {
        "system": "linux",
        "machine": "x86_64",
        "python_version": "3.12.0",
        "torch_installed": False,
        "torch_version": None,
        "torch_cuda": False,
        "torch_cuda_version": None,
        "torch_device_count": None,
        "torch_device_names": (),
        "torch_compute_capability": None,
        "torch_mps_available": False,
        "onnxruntime_installed": False,
        "onnxruntime_version": None,
        "onnxruntime_providers": (),
        "onnx_cuda": False,
        "onnx_coreml": False,
        "mps": False,
        "ram_gb": None,
        "vram_gb": None,
        "nvidia_device": False,
        "nvidia_smi_available": False,
        "nvidia_driver_version": None,
        "driver_cuda_version": None,
    }
    values.update(overrides)
    return RuntimeCapabilities(**values)


def test_infer_requirements_from_format():
    assert infer_model_requirements(_entry("pytorch")).runtime == "torch"
    assert infer_model_requirements(_entry("onnx")).runtime == "onnxruntime"
    assert infer_model_requirements(_entry("ct2")).runtime is None
    assert infer_model_requirements(_entry("gguf")).runtime is None


def test_vllm_adapter_requires_cuda():
    reqs = infer_model_requirements(_entry("pytorch", "voxtral-tts-vllm"))
    assert reqs.runtime == "torch"
    assert reqs.accelerator == "cuda"


def test_missing_torch_blocks_pytorch_model(monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "1")
    monkeypatch.delenv("VOX_RUNTIME_OVERRIDE", raising=False)
    missing = missing_capabilities_for(
        _entry("pytorch", "qwen3-tts-torch"),
        snapshot=_caps(torch_installed=False),
    )
    assert len(missing) == 1
    assert "PyTorch" in missing[0]


def test_torch_present_allows_pytorch_model(monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "0")
    monkeypatch.delenv("VOX_RUNTIME_OVERRIDE", raising=False)
    assert missing_capabilities_for(
        _entry("pytorch", "qwen3-tts-torch"),
        snapshot=_caps(torch_installed=True),
    ) == []


def test_torch_env_override_requires_explicit_override_mode(monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "1")
    monkeypatch.delenv("VOX_RUNTIME_OVERRIDE", raising=False)
    missing = missing_capabilities_for(
        _entry("pytorch", "qwen3-tts-torch"),
        snapshot=_caps(torch_installed=False),
    )
    assert missing

    monkeypatch.setenv("VOX_RUNTIME_OVERRIDE", "1")
    assert missing_capabilities_for(
        _entry("pytorch", "qwen3-tts-torch"),
        snapshot=_caps(torch_installed=False),
    ) == []


def test_onnx_model_allowed_when_onnxruntime_present(monkeypatch):
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "0")
    monkeypatch.delenv("VOX_RUNTIME_OVERRIDE", raising=False)
    assert missing_capabilities_for(
        _entry("onnx", "kokoro-tts-onnx"),
        snapshot=_caps(onnxruntime_installed=True),
    ) == []


def test_bare_onnx_model_blocked_when_onnxruntime_missing(monkeypatch):
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "1")
    monkeypatch.delenv("VOX_RUNTIME_OVERRIDE", raising=False)
    missing = missing_capabilities_for(
        _entry("onnx", "kokoro-tts-onnx"),
        snapshot=_caps(onnxruntime_installed=False),
    )
    assert missing and "onnxruntime" in missing[0]


def test_adapter_backed_onnx_model_does_not_require_base_onnxruntime(monkeypatch):
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "0")
    monkeypatch.delenv("VOX_RUNTIME_OVERRIDE", raising=False)

    entry = {
        "format": "onnx",
        "adapter": "kokoro-tts-onnx",
        "adapter_package": "vox-kokoro",
        "runtime": {"required": {"python_modules": ["onnxruntime"]}},
    }

    assert missing_capabilities_for(
        entry,
        snapshot=_caps(onnxruntime_installed=False),
    ) == []


def test_ct2_model_never_requires_base_runtime(monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "0")
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "0")
    assert missing_capabilities_for(
        _entry("ct2", "whisper-stt-ct2"),
        snapshot=_caps(),
    ) == []


def test_vllm_model_blocked_without_gpu(monkeypatch):
    missing = missing_capabilities_for(
        _entry("pytorch", "voxtral-tts-vllm"),
        snapshot=_caps(torch_installed=True, torch_cuda=False, nvidia_device=True),
    )
    assert any("cuda" in m for m in missing)


def test_vllm_model_allows_torch_cuda():
    assert missing_capabilities_for(
        _entry("pytorch", "voxtral-tts-vllm"),
        snapshot=_caps(torch_installed=True, torch_cuda=True),
    ) == []


def test_min_versions_block_old_torch():
    check = check_runtime_requirement(
        RuntimeRequirement(python_modules=("torch",), min_versions={"torch": "2.5.1"}),
        snapshot=_caps(torch_installed=True, torch_version="2.4.0"),
    )
    assert not check.runnable
    assert any("torch>=2.5.1" in reason for reason in check.missing)


def test_unknown_compute_capability_fails_ordered_requirement():
    check = check_runtime_requirement(
        RuntimeRequirement(accelerators=("cuda",), min_compute_capability=80),
        snapshot=_caps(torch_cuda=True, torch_compute_capability=None),
    )
    assert not check.runnable
    assert any("compute capability" in reason and "UNKNOWN" in reason for reason in check.missing)


def test_unknown_vram_fails_ordered_requirement():
    check = check_runtime_requirement(
        RuntimeRequirement(accelerators=("cuda",), min_vram_gb=8),
        snapshot=_caps(torch_cuda=True, vram_gb=None),
    )
    assert not check.runnable
    assert any("VRAM" in reason and "UNKNOWN" in reason for reason in check.missing)


def test_runtime_snapshot_handles_missing_optional_backends(monkeypatch):
    monkeypatch.setattr("vox.core.runtime._torch_probe", lambda: type("T", (), {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": None,
        "device_names": (),
        "compute_capability": None,
        "mps_available": False,
        "vram_gb": None,
    })())
    monkeypatch.setattr("vox.core.runtime._onnx_probe", lambda: type("O", (), {
        "installed": False,
        "version": None,
        "providers": (),
    })())
    monkeypatch.setattr("vox.core.runtime._nvidia_smi_probe", lambda: type("N", (), {
        "available": False,
        "driver_version": None,
        "cuda_version": None,
        "vram_gb": None,
    })())
    caps = detect_runtime_capabilities()
    assert caps.torch_installed is False
    assert caps.torch_cuda is False
    assert caps.onnxruntime_installed is False


def test_runtime_snapshot_captures_cuda_details(monkeypatch):
    monkeypatch.setattr("vox.core.runtime._torch_probe", lambda: type("T", (), {
        "installed": True,
        "version": "2.8.0",
        "cuda_available": True,
        "cuda_version": "12.8",
        "device_count": 1,
        "device_names": ("RTX Test",),
        "compute_capability": 89,
        "mps_available": False,
        "vram_gb": 24.0,
    })())
    monkeypatch.setattr("vox.core.runtime._onnx_probe", lambda: type("O", (), {
        "installed": True,
        "version": "1.23.0",
        "providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    })())
    monkeypatch.setattr("vox.core.runtime._nvidia_smi_probe", lambda: type("N", (), {
        "available": True,
        "driver_version": "575.1",
        "cuda_version": "12.8",
        "vram_gb": 24.0,
    })())
    caps = detect_runtime_capabilities()
    assert caps.torch_installed is True
    assert caps.torch_cuda is True
    assert caps.torch_version == "2.8.0"
    assert caps.torch_cuda_version == "12.8"
    assert caps.torch_device_count == 1
    assert caps.torch_device_names == ("RTX Test",)
    assert caps.torch_compute_capability == 89
    assert caps.onnx_cuda is True
    assert caps.vram_gb == 24.0


def test_incompatible_pull_allowed_override(monkeypatch):
    monkeypatch.delenv("VOX_ALLOW_INCOMPATIBLE", raising=False)
    assert incompatible_pull_allowed() is False
    monkeypatch.setenv("VOX_ALLOW_INCOMPATIBLE", "1")
    assert incompatible_pull_allowed() is True
    monkeypatch.setenv("VOX_ALLOW_INCOMPATIBLE", "no")
    assert incompatible_pull_allowed() is False
