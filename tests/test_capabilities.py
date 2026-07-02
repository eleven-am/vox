from __future__ import annotations

from vox.core.capabilities import (
    incompatible_pull_allowed,
    infer_model_requirements,
    missing_capabilities_for,
)


def _entry(fmt: str, adapter: str = "some-adapter") -> dict:
    return {"format": fmt, "adapter": adapter}


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
    monkeypatch.setenv("VOX_HAS_TORCH", "0")
    missing = missing_capabilities_for(_entry("pytorch", "qwen3-tts-torch"))
    assert len(missing) == 1
    assert "PyTorch" in missing[0]


def test_torch_present_allows_pytorch_model(monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "1")
    assert missing_capabilities_for(_entry("pytorch", "qwen3-tts-torch")) == []


def test_onnx_model_allowed_when_onnxruntime_present(monkeypatch):
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "1")
    assert missing_capabilities_for(_entry("onnx", "kokoro-tts-onnx")) == []


def test_onnx_model_blocked_when_onnxruntime_missing(monkeypatch):
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "0")
    missing = missing_capabilities_for(_entry("onnx", "kokoro-tts-onnx"))
    assert missing and "onnxruntime" in missing[0]


def test_ct2_model_never_requires_base_runtime(monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "0")
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "0")
    assert missing_capabilities_for(_entry("ct2", "whisper-stt-ct2")) == []


def test_vllm_model_blocked_without_gpu(monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "1")
    monkeypatch.setattr(
        "vox.core.capabilities.detect_runtime_capabilities",
        lambda: type("C", (), {"has_gpu_accelerator": False})(),
    )
    missing = missing_capabilities_for(_entry("pytorch", "voxtral-tts-vllm"))
    assert any("CUDA GPU" in m for m in missing)


def test_incompatible_pull_allowed_override(monkeypatch):
    monkeypatch.delenv("VOX_ALLOW_INCOMPATIBLE", raising=False)
    assert incompatible_pull_allowed() is False
    monkeypatch.setenv("VOX_ALLOW_INCOMPATIBLE", "1")
    assert incompatible_pull_allowed() is True
    monkeypatch.setenv("VOX_ALLOW_INCOMPATIBLE", "no")
    assert incompatible_pull_allowed() is False
