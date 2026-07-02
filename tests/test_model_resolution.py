from __future__ import annotations

from vox.core.model_resolution import parse_model_variant_ref, resolve_catalog_entry
from vox.core.runtime import RuntimeCapabilities


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
        "onnxruntime_installed": True,
        "onnxruntime_version": "1.27.0",
        "onnxruntime_providers": ("CPUExecutionProvider",),
        "onnx_cuda": False,
        "onnx_coreml": False,
        "mps": False,
        "ram_gb": 16.0,
        "vram_gb": None,
        "nvidia_device": False,
        "nvidia_smi_available": False,
        "nvidia_driver_version": None,
        "driver_cuda_version": None,
    }
    values.update(overrides)
    return RuntimeCapabilities(**values)


def _logical_kokoro_entry():
    return {
        "type": "tts",
        "architecture": "kokoro",
        "description": "logical kokoro",
        "variants": [
            {
                "id": "torch",
                "aliases": ["cuda"],
                "priority": 100,
                "requires": {
                    "python_modules": ["torch"],
                    "accelerators": ["cuda"],
                    "min_compute_capability": 70,
                },
                "adapter": "kokoro-tts-torch",
                "adapter_package": "vox-kokoro",
                "format": "pytorch",
                "source": "hexgrad/Kokoro-82M",
            },
            {
                "id": "onnx",
                "aliases": ["cpu"],
                "priority": 0,
                "fallback": True,
                "requires": {"python_modules": ["onnxruntime"]},
                "adapter": "kokoro-tts-onnx",
                "adapter_package": "vox-kokoro",
                "format": "onnx",
                "source": "onnx-community/Kokoro-82M-v1.0-ONNX",
            },
        ],
    }


def test_parse_model_variant_ref_keeps_concrete_names_untouched():
    parsed = parse_model_variant_ref("parakeet-stt-onnx:tdt-0.6b-v3")

    assert parsed.name == "parakeet-stt-onnx"
    assert parsed.tag == "tdt-0.6b-v3"
    assert parsed.explicit_tag is True


def test_parse_model_variant_ref_supports_explicit_tag():
    parsed = parse_model_variant_ref("kokoro-tts:v1.0")

    assert parsed.name == "kokoro-tts"
    assert parsed.tag == "v1.0"
    assert parsed.explicit_tag is True


def test_parse_model_variant_ref_keeps_double_hyphen_as_name_text():
    parsed = parse_model_variant_ref("kokoro-tts--onnx:v1.0")

    assert parsed.name == "kokoro-tts--onnx"
    assert parsed.tag == "v1.0"


def test_concrete_catalog_entry_resolves_as_single_entry():
    entry = {
        "architecture": "kokoro",
        "type": "tts",
        "adapter": "kokoro-tts-onnx",
        "format": "onnx",
        "source": "onnx-community/Kokoro-82M-v1.0-ONNX",
    }

    resolved = resolve_catalog_entry(entry, snapshot=_caps())

    assert resolved.entry == entry
    assert resolved.variant_id is None
    assert resolved.missing == ()


def test_logical_entry_resolves_cpu_variant_on_torch_free_runtime():
    resolved = resolve_catalog_entry(_logical_kokoro_entry(), snapshot=_caps())

    assert resolved.variant_id == "onnx"
    assert resolved.entry["adapter"] == "kokoro-tts-onnx"
    assert resolved.entry["format"] == "onnx"
    assert resolved.missing == ()


def test_logical_entry_resolves_highest_priority_cuda_variant_when_available():
    resolved = resolve_catalog_entry(
        _logical_kokoro_entry(),
        snapshot=_caps(
            torch_installed=True,
            torch_cuda=True,
            torch_compute_capability=89,
            vram_gb=24.0,
        ),
    )

    assert resolved.variant_id == "torch"
    assert resolved.entry["adapter"] == "kokoro-tts-torch"
    assert resolved.entry["format"] == "pytorch"


def test_forced_variant_returns_its_missing_reasons():
    resolved = resolve_catalog_entry(
        _logical_kokoro_entry(),
        snapshot=_caps(),
        forced_variant="torch",
    )

    assert resolved.variant_id == "torch"
    assert resolved.entry["adapter"] == "kokoro-tts-torch"
    assert resolved.missing
    assert any("PyTorch" in reason for reason in resolved.missing)


def test_no_compatible_variant_reports_every_variant_failure():
    entry = _logical_kokoro_entry()
    resolved = resolve_catalog_entry(
        entry,
        snapshot=_caps(onnxruntime_installed=False),
    )

    assert resolved.variant_id == "onnx"
    assert resolved.entry["adapter"] == "kokoro-tts-onnx"
    assert any("variant torch:" in reason for reason in resolved.missing)
    assert any("variant onnx:" in reason for reason in resolved.missing)


def test_forced_variant_alias_can_select_onnx_on_cuda_runtime():
    resolved = resolve_catalog_entry(
        _logical_kokoro_entry(),
        snapshot=_caps(
            torch_installed=True,
            torch_cuda=True,
            torch_compute_capability=89,
            vram_gb=24.0,
        ),
        forced_variant="onnx",
    )

    assert resolved.variant_id == "onnx"
    assert resolved.entry["adapter"] == "kokoro-tts-onnx"
    assert resolved.missing == ()


def test_legacy_forced_variant_alias_still_maps_to_torch_variant():
    resolved = resolve_catalog_entry(
        _logical_kokoro_entry(),
        snapshot=_caps(
            torch_installed=True,
            torch_cuda=True,
            torch_compute_capability=89,
            vram_gb=24.0,
        ),
        forced_variant="cuda",
    )

    assert resolved.variant_id == "torch"
    assert resolved.entry["adapter"] == "kokoro-tts-torch"


def test_variant_records_preferred_backend_fallback_warning():
    entry = {
        "type": "tts",
        "architecture": "qwen3-tts",
        "variants": [
            {
                "id": "torch",
                "priority": 0,
                "fallback": True,
                "requires": {"python_modules": ["torch"]},
                "adapter": "qwen3-tts-torch",
                "adapter_package": "vox-qwen",
                "format": "pytorch",
                "source": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "backends": {
                    "preferred": [
                        {
                            "name": "faster-qwen3-tts",
                            "requires": {
                                "python_modules": ["torch", "faster_qwen3_tts"],
                                "accelerators": ["cuda"],
                            },
                        }
                    ],
                    "fallback": {
                        "name": "qwen-tts",
                        "requires": {"python_modules": ["torch"]},
                    },
                },
            }
        ],
    }

    resolved = resolve_catalog_entry(
        entry,
        snapshot=_caps(torch_installed=True, torch_cuda=False),
    )

    assert resolved.variant_id == "torch"
    assert resolved.preferred_backend == "qwen-tts"
    assert resolved.missing == ()
    assert resolved.warnings
    assert "faster-qwen3-tts" in resolved.warnings[0]
