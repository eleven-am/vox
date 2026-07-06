from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vox.core.runtime import RuntimeCapabilities
from vox.core.store import BlobStore
from vox.core.types import ModelFormat, ModelInfo, ModelType
from vox.operations.errors import (
    CatalogEntryNotFoundError,
    ModelIncompatibleError,
    ModelInUseError,
    StoredModelNotFoundError,
)
from vox.operations.models import (
    ModelLayer,
    PullEvent,
    ShowResult,
    delete_model,
    delete_model_payload,
    list_models,
    list_models_payload,
    model_reference_request_from_fields,
    pull_event_payload,
    pull_model,
    resolve_model_reference,
    show_model,
    show_model_payload,
)


def _registry_mock() -> MagicMock:
    reg = MagicMock()
    reg.resolve_model_ref.side_effect = lambda n, t, explicit_tag=False: (n, t)
    return reg


def _runtime_caps(**overrides) -> RuntimeCapabilities:
    values = {
        "system": "linux",
        "machine": "x86_64",
        "python_version": "3.12.0",
        "torch_installed": True,
        "torch_version": "2.8.0",
        "torch_cuda": True,
        "torch_cuda_version": "12.8",
        "torch_device_count": 1,
        "torch_device_names": ("RTX Test",),
        "torch_compute_capability": 89,
        "torch_mps_available": False,
        "onnxruntime_installed": True,
        "onnxruntime_version": "1.27.0",
        "onnxruntime_providers": ("CPUExecutionProvider",),
        "onnx_cuda": False,
        "onnx_coreml": False,
        "mps": False,
        "ram_gb": 32.0,
        "vram_gb": 24.0,
        "nvidia_device": True,
        "nvidia_smi_available": True,
        "nvidia_driver_version": "575.1",
        "driver_cuda_version": "12.8",
    }
    values.update(overrides)
    return RuntimeCapabilities(**values)


def test_list_models_returns_store_models(tmp_path: Path):
    store = MagicMock()
    fake = MagicMock()
    store.list_models.return_value = [fake]
    assert list_models(store=store) == [fake]


def test_delete_model_payload_preserves_http_contract_shape():
    assert delete_model_payload() == {"status": "success"}


def test_model_reference_request_from_fields_preserves_transport_name():
    request = model_reference_request_from_fields(name="parakeet:tdt-0.6b-v3")

    assert request.name == "parakeet:tdt-0.6b-v3"


def test_resolve_model_reference_preserves_explicit_tag_policy():
    registry = _registry_mock()

    resolved = resolve_model_reference(
        registry=registry,
        request=model_reference_request_from_fields(name="parakeet:tdt-0.6b-v3"),
    )

    registry.resolve_model_ref.assert_called_once_with(
        "parakeet",
        "tdt-0.6b-v3",
        explicit_tag=True,
    )
    assert resolved.requested_name == "parakeet:tdt-0.6b-v3"
    assert resolved.parsed_name == "parakeet"
    assert resolved.parsed_tag == "tdt-0.6b-v3"
    assert resolved.requested_variant is None
    assert resolved.explicit_tag is True


def test_resolve_model_reference_keeps_variant_out_of_registry_aliases():
    registry = _registry_mock()

    resolved = resolve_model_reference(
        registry=registry,
        request=model_reference_request_from_fields(name="kokoro-tts:v1.0", variant="cuda"),
    )

    registry.resolve_model_ref.assert_called_once_with(
        "kokoro-tts",
        "v1.0",
        explicit_tag=True,
    )
    assert resolved.parsed_name == "kokoro-tts"
    assert resolved.parsed_tag == "v1.0"
    assert resolved.requested_variant == "cuda"


def test_show_model_raises_when_missing(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    with pytest.raises(StoredModelNotFoundError):
        show_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="missing:latest"),
        )


def test_show_model_returns_layers_and_config(tmp_path: Path):
    from vox.core.store import Manifest, ManifestLayer

    store = BlobStore(root=tmp_path)
    layer = ManifestLayer(media_type="application/x", digest="sha256-x", size=1, filename="x.bin")
    manifest = Manifest(
        layers=[layer],
        config={"architecture": "fake", "type": "stt", "adapter": "fake", "format": "onnx"},
    )
    store.save_manifest("foo", "latest", manifest)
    registry = _registry_mock()
    result = show_model(
        store=store,
        registry=registry,
        request=model_reference_request_from_fields(name="foo:latest"),
    )
    assert result.name == "foo:latest"
    assert result.config["architecture"] == "fake"
    assert result.layers[0].digest == "sha256-x"


@pytest.mark.asyncio
async def test_delete_model_in_use_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    scheduler = MagicMock()
    scheduler.unload = AsyncMock(return_value=False)
    registry = _registry_mock()
    with pytest.raises(ModelInUseError):
        await delete_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
        )


@pytest.mark.asyncio
async def test_delete_model_missing_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    scheduler = MagicMock()
    scheduler.unload = AsyncMock(return_value=True)
    registry = _registry_mock()
    with pytest.raises(StoredModelNotFoundError):
        await delete_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="missing:latest"),
        )


@pytest.mark.asyncio
async def test_delete_model_success_removes_manifest(tmp_path: Path):
    from vox.core.store import Manifest, ManifestLayer

    store = BlobStore(root=tmp_path)
    layer = ManifestLayer(media_type="application/x", digest="sha256-x", size=1, filename="x.bin")
    manifest = Manifest(
        layers=[layer],
        config={"architecture": "fake", "type": "stt", "adapter": "fake", "format": "onnx"},
    )
    store.save_manifest("foo", "latest", manifest)
    scheduler = MagicMock()
    scheduler.unload = AsyncMock(return_value=True)
    registry = _registry_mock()
    await delete_model(
        store=store,
        scheduler=scheduler,
        registry=registry,
        request=model_reference_request_from_fields(name="foo:latest"),
    )
    assert store.resolve_model("foo", "latest") is None


def test_pull_model_unknown_catalog_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = None
    scheduler = MagicMock()
    with pytest.raises(CatalogEntryNotFoundError):
        pull_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="missing:latest"),
        )


def test_pull_model_blocks_incompatible_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "0")
    monkeypatch.setenv("VOX_RUNTIME_OVERRIDE", "1")
    monkeypatch.delenv("VOX_ALLOW_INCOMPATIBLE", raising=False)
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake", "type": "tts", "adapter": "qwen3-tts-torch",
        "format": "pytorch", "source": "owner/repo", "parameters": {}, "adapter_package": "vox-qwen",
    }
    with pytest.raises(ModelIncompatibleError, match="PyTorch"):
        pull_model(
            store=store,
            scheduler=MagicMock(),
            registry=registry,
            request=model_reference_request_from_fields(name="qwen3-tts:0.6b"),
        )


def test_pull_model_override_allows_incompatible_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "0")
    monkeypatch.setenv("VOX_RUNTIME_OVERRIDE", "1")
    monkeypatch.setenv("VOX_ALLOW_INCOMPATIBLE", "1")
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake", "type": "tts", "adapter": "qwen3-tts-torch",
        "format": "pytorch", "source": "owner/repo", "parameters": {}, "adapter_package": "vox-qwen",
    }
    # gate is bypassed -> pull_model returns the async generator without raising
    events = pull_model(
        store=store,
        scheduler=MagicMock(),
        registry=registry,
        request=model_reference_request_from_fields(name="qwen3-tts:0.6b"),
    )
    assert events is not None


@pytest.mark.asyncio
async def test_pull_model_override_emits_missing_capability_warning(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VOX_HAS_TORCH", "0")
    monkeypatch.setenv("VOX_RUNTIME_OVERRIDE", "1")
    monkeypatch.setenv("VOX_ALLOW_INCOMPATIBLE", "1")
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "qwen3-tts-torch",
        "format": "pytorch",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }
    scheduler = MagicMock()

    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)):
        events = pull_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="qwen3-tts:0.6b"),
        )
        collected = [event async for event in events]

    warning = next(event for event in collected if event.status == "warning")
    assert "VOX_ALLOW_INCOMPATIBLE=1" in warning.error
    assert "PyTorch" in warning.error
    assert collected[-1].status == "success"


@pytest.mark.asyncio
async def test_pull_model_yields_progress_and_success(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VOX_RUNTIME_OVERRIDE", "1")
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "1")
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "stt",
        "adapter": "fake",
        "format": "onnx",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "",
    }
    scheduler = MagicMock()

    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("huggingface_hub.HfApi") as mock_api_cls,
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
    ):
        mock_api_cls.return_value.repo_info.return_value = MagicMock(
            siblings=[MagicMock(rfilename="model.bin")]
        )
        events = pull_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
        )
        collected = [event async for event in events]

    assert collected[-1].status == "success"
    assert any(e.status.startswith("downloading") for e in collected)
    manifest = store.resolve_model("foo", "latest")
    assert manifest is not None
    assert manifest.config["runtime"]["checked_at_pull"] is True
    assert manifest.config["runtime"]["resolved_variant"] == ""


@pytest.mark.asyncio
async def test_pull_model_prepares_adapter_runtime_after_download(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VOX_RUNTIME_OVERRIDE", "1")
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "1")
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "stt",
        "adapter": "fake",
        "format": "onnx",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "vox-fake",
        "files": ["model.bin"],
    }
    scheduler = MagicMock()
    prepared: list[str] = []

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            prepared.append("runtime")

    registry.ensure_adapter.return_value = True
    registry.get_adapter_class.return_value = FakeAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)):
        events = pull_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
        )
        collected = [event async for event in events]

    assert prepared == ["runtime"]
    assert any(event.status == "preparing adapter runtime fake" for event in collected)
    assert any(event.status == "adapter runtime fake ready" for event in collected)
    assert collected[-1].status == "success"


@pytest.mark.asyncio
async def test_pull_model_does_not_save_manifest_when_runtime_prepare_fails(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("VOX_RUNTIME_OVERRIDE", "1")
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "1")
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "fake",
        "format": "onnx",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "vox-fake",
        "files": ["model.bin"],
    }
    scheduler = MagicMock()

    class BrokenAdapter:
        def prepare_runtime(self) -> None:
            raise RuntimeError("runtime bootstrap failed")

    registry.ensure_adapter.return_value = True
    registry.get_adapter_class.return_value = BrokenAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)):
        events = pull_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
        )
        collected = [event async for event in events]

    assert any(event.status == "preparing adapter runtime fake" for event in collected)
    assert collected[-1].status == "error"
    assert collected[-1].error == "runtime bootstrap failed"
    assert store.resolve_model("foo", "latest") is None


@pytest.mark.asyncio
async def test_pull_model_resolves_logical_variant_and_records_runtime_metadata(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("VOX_RUNTIME_OVERRIDE", "1")
    monkeypatch.setenv("VOX_HAS_ONNXRUNTIME", "1")
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "type": "tts",
        "architecture": "kokoro",
        "description": "logical kokoro",
        "variants": [
            {
                "id": "torch",
                "aliases": ["cuda"],
                "priority": 100,
                "requires": {"python_modules": ["torch"], "accelerators": ["cuda"]},
                "adapter": "kokoro-tts-torch",
                "format": "pytorch",
                "source": "hexgrad/Kokoro-82M",
                "adapter_package": "",
                "files": ["model.bin"],
            },
            {
                "id": "onnx",
                "aliases": ["cpu"],
                "priority": 0,
                "fallback": True,
                "requires": {"python_modules": ["onnxruntime"]},
                "adapter": "kokoro-tts-onnx",
                "format": "onnx",
                "source": "onnx-community/Kokoro-82M-v1.0-ONNX",
                "adapter_package": "",
                "files": ["model.bin"],
                "backends": {
                    "preferred": [
                        {
                            "name": "kokoro-onnx-cuda",
                            "requires": {"accelerators": ["onnx_cuda"]},
                        }
                    ],
                    "fallback": {"name": "kokoro-onnx-cpu"},
                },
            },
        ],
    }
    scheduler = MagicMock()

    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)):
        events = pull_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="kokoro-tts:latest"),
        )
        collected = [event async for event in events]

    assert collected[-1].status == "success"
    warning = next(event for event in collected if event.status == "warning")
    assert "kokoro-onnx-cuda" in warning.error
    manifest = store.resolve_model("kokoro-tts", "latest")
    assert manifest is not None
    assert manifest.config["adapter"] == "kokoro-tts-onnx"
    assert manifest.config["format"] == "onnx"
    assert manifest.config["source"] == "onnx-community/Kokoro-82M-v1.0-ONNX"
    assert manifest.config["runtime"]["resolved_variant"] == "onnx"
    assert manifest.config["runtime"]["preferred_backend"] == "kokoro-onnx-cpu"
    assert "kokoro-onnx-cuda" in manifest.config["runtime"]["warnings"][0]
    assert manifest.config["runtime"]["detected"]["onnxruntime_installed"] is True


@pytest.mark.asyncio
async def test_pull_model_can_force_onnx_variant_on_cuda_runtime(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "type": "tts",
        "architecture": "kokoro",
        "variants": [
            {
                "id": "torch",
                "aliases": ["cuda"],
                "priority": 100,
                "requires": {"python_modules": ["torch"], "accelerators": ["cuda"]},
                "adapter": "kokoro-tts-torch",
                "format": "pytorch",
                "source": "hexgrad/Kokoro-82M",
                "adapter_package": "",
                "files": ["model.bin"],
            },
            {
                "id": "onnx",
                "aliases": ["cpu"],
                "priority": 0,
                "fallback": True,
                "requires": {"python_modules": ["onnxruntime"]},
                "adapter": "kokoro-tts-onnx",
                "format": "onnx",
                "source": "onnx-community/Kokoro-82M-v1.0-ONNX",
                "adapter_package": "",
                "files": ["model.bin"],
            },
        ],
    }
    scheduler = MagicMock()

    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
    ):
        events = pull_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="kokoro-tts:v1.0", variant="onnx"),
        )
        collected = [event async for event in events]

    assert collected[-1].status == "success"
    manifest = store.resolve_model("kokoro-tts", "v1.0")
    assert manifest is not None
    assert manifest.config["adapter"] == "kokoro-tts-onnx"
    assert manifest.config["format"] == "onnx"
    assert manifest.config["runtime"]["resolved_variant"] == "onnx"
    assert manifest.config["runtime"]["detected"]["torch_cuda"] is True


def test_pull_model_forced_missing_variant_fails_clearly(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "type": "tts",
        "architecture": "kokoro",
        "variants": [
            {
                "id": "onnx",
                "aliases": ["cpu"],
                "priority": 0,
                "fallback": True,
                "requires": {"python_modules": ["onnxruntime"]},
                "adapter": "kokoro-tts-onnx",
                "format": "onnx",
                "source": "onnx-community/Kokoro-82M-v1.0-ONNX",
                "adapter_package": "",
            }
        ],
    }

    with pytest.raises(ModelIncompatibleError, match="variant 'mlx' is not defined"):
        pull_model(
            store=store,
            scheduler=MagicMock(),
            registry=registry,
            request=model_reference_request_from_fields(name="kokoro-tts:v1.0", variant="mlx"),
        )


@pytest.mark.asyncio
async def test_pull_model_voxtral_emits_preload_events(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VOX_ALLOW_INCOMPATIBLE", "1")
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "voxtral-tts-vllm",
        "type": "tts",
        "adapter": "voxtral-tts-vllm",
        "format": "pytorch",
        "source": "owner/voxtral",
        "parameters": {},
        "adapter_package": "",
    }
    scheduler = MagicMock()
    scheduler.preload = AsyncMock()

    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("huggingface_hub.HfApi") as mock_api_cls,
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
    ):
        mock_api_cls.return_value.repo_info.return_value = MagicMock(
            siblings=[MagicMock(rfilename="model.bin")]
        )
        events = pull_model(
            store=store,
            scheduler=scheduler,
            registry=registry,
            request=model_reference_request_from_fields(name="voxtral-tts-vllm:4b"),
        )
        collected = [event async for event in events]

    assert any("preloading" in e.status for e in collected)
    assert any(e.status.endswith("ready") for e in collected)
    assert collected[-1].status == "success"
    scheduler.preload.assert_awaited_once()


def test_pull_event_default_fields():
    event = PullEvent(status="x")
    assert event.completed == 0
    assert event.total == 0
    assert event.error == ""


def test_list_models_payload_preserves_http_contract_shape():
    model = ModelInfo(
        name="parakeet-stt-onnx",
        tag="tdt-0.6b-v3",
        type=ModelType.STT,
        format=ModelFormat.ONNX,
        architecture="parakeet",
        adapter="parakeet",
        size_bytes=123,
        description="fast stt",
    )

    assert list_models_payload([model]) == {
        "models": [
            {
                "name": "parakeet-stt-onnx:tdt-0.6b-v3",
                "type": "stt",
                "format": "onnx",
                "architecture": "parakeet",
                "size_bytes": 123,
                "description": "fast stt",
            }
        ]
    }


def test_show_model_payload_preserves_layers_and_config_shape():
    result = ShowResult(
        name="foo:latest",
        config={"architecture": "fake"},
        layers=(
            ModelLayer(
                media_type="application/vox.model.bin",
                digest="sha256-x",
                size=12,
                filename="model.bin",
            ),
        ),
    )

    assert show_model_payload(result) == {
        "name": "foo:latest",
        "config": {"architecture": "fake"},
        "layers": [
            {
                "media_type": "application/vox.model.bin",
                "digest": "sha256-x",
                "size": 12,
                "filename": "model.bin",
            }
        ],
    }


def test_pull_event_payload_omits_zero_progress_and_empty_error():
    assert pull_event_payload(PullEvent(status="checking")) == {"status": "checking"}


def test_pull_event_payload_includes_progress_and_error_when_present():
    assert pull_event_payload(PullEvent(status="error", completed=2, total=5, error="boom")) == {
        "status": "error",
        "completed": 2,
        "total": 5,
        "error": "boom",
    }
