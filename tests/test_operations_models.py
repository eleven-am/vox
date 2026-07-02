from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vox.core.store import BlobStore
from vox.core.types import ModelFormat, ModelInfo, ModelType
from vox.operations.errors import (
    CatalogEntryNotFoundError,
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
    assert resolved.explicit_tag is True


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


@pytest.mark.asyncio
async def test_pull_model_yields_progress_and_success(tmp_path: Path):
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


@pytest.mark.asyncio
async def test_pull_model_voxtral_emits_preload_events(tmp_path: Path):
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
