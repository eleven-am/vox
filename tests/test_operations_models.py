from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vox.core.store import BlobStore
from vox.operations.errors import (
    CatalogEntryNotFoundError,
    ModelInUseError,
    StoredModelNotFoundError,
)
from vox.operations.models import (
    PullEvent,
    delete_model,
    list_models,
    pull_model,
    show_model,
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


def test_show_model_raises_when_missing(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    with pytest.raises(StoredModelNotFoundError):
        show_model(store=store, registry=registry, name="missing:latest")


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
    result = show_model(store=store, registry=registry, name="foo:latest")
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
            store=store, scheduler=scheduler, registry=registry, name="foo:latest",
        )


@pytest.mark.asyncio
async def test_delete_model_missing_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    scheduler = MagicMock()
    scheduler.unload = AsyncMock(return_value=True)
    registry = _registry_mock()
    with pytest.raises(StoredModelNotFoundError):
        await delete_model(
            store=store, scheduler=scheduler, registry=registry, name="missing:latest",
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
    await delete_model(store=store, scheduler=scheduler, registry=registry, name="foo:latest")
    assert store.resolve_model("foo", "latest") is None


def test_pull_model_unknown_catalog_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = None
    scheduler = MagicMock()
    with pytest.raises(CatalogEntryNotFoundError):
        pull_model(store=store, scheduler=scheduler, registry=registry, name="missing:latest")


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
            store=store, scheduler=scheduler, registry=registry, name="foo:latest",
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
            store=store, scheduler=scheduler, registry=registry, name="voxtral-tts-vllm:4b",
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
