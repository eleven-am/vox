"""Tests for ModelRegistry: catalog lookup, adapter discovery, and model resolution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox.core.errors import AdapterNotFoundError, ModelNotFoundError
from vox.core.registry import CATALOG, ModelRegistry, discover_adapters
from vox.core.store import BlobStore, Manifest, ManifestLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> BlobStore:
    """Create a BlobStore rooted at tmp_path."""
    return BlobStore(root=tmp_path)


def _write_manifest(
    store: BlobStore,
    name: str,
    tag: str,
    *,
    adapter: str = "fake",
    model_type: str = "stt",
    fmt: str = "onnx",
    source: str | None = None,
    layers: list[ManifestLayer] | None = None,
) -> Manifest:
    """Create a manifest on disk and return it."""
    if layers is None:
        # Create a default layer with a real blob file
        digest = "sha256-" + "ab" * 32
        blob_path = store.blobs_dir / digest
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"fake-model-data")
        layers = [
            ManifestLayer(
                media_type="application/vox.model.onnx",
                digest=digest,
                size=15,
                filename="model.onnx",
            )
        ]

    config: dict = {
        "architecture": "test-arch",
        "type": model_type,
        "adapter": adapter,
        "format": fmt,
        "parameters": {"sample_rate": 16000},
    }
    if source is not None:
        config["source"] = source

    manifest = Manifest(layers=layers, config=config)
    store.save_manifest(name, tag, manifest)
    return manifest


def _make_registry(store: BlobStore, adapters: dict | None = None) -> ModelRegistry:
    """Create a ModelRegistry with mocked adapter discovery."""
    with patch("vox.core.registry.discover_adapters", return_value=adapters or {}):
        return ModelRegistry(store)


# ---------------------------------------------------------------------------
# Catalog lookup tests
# ---------------------------------------------------------------------------

class TestLookup:
    def test_lookup_existing_model(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        entry = registry.lookup("whisper", "large-v3")
        assert entry is not None
        assert entry["source"] == "Systran/faster-whisper-large-v3"
        assert entry["type"] == "stt"
        assert entry["adapter"] == "whisper"

    def test_lookup_missing_model(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        assert registry.lookup("nonexistent-model") is None

    def test_lookup_missing_tag(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        # "whisper" exists but "no-such-tag" does not
        assert registry.lookup("whisper", "no-such-tag") is None


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestGetAdapterClass:
    def test_get_adapter_class_raises_when_missing(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store, adapters={})

        with pytest.raises(AdapterNotFoundError):
            registry.get_adapter_class("nonexistent")

    def test_get_adapter_class_returns_class(self, tmp_path: Path):
        store = _make_store(tmp_path)

        class FakeAdapter:
            pass

        registry = _make_registry(store, adapters={"fake": FakeAdapter})
        assert registry.get_adapter_class("fake") is FakeAdapter


# ---------------------------------------------------------------------------
# Resolve tests
# ---------------------------------------------------------------------------

class TestResolve:
    def test_resolve_raises_model_not_found_when_no_manifest(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        with pytest.raises(ModelNotFoundError):
            registry.resolve("nothing", "latest")

    def test_resolve_creates_symlinks(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(store, "mymodel", "v1")
        registry = _make_registry(store)

        info, model_dir = registry.resolve("mymodel", "v1")

        assert info.name == "mymodel"
        assert info.tag == "v1"
        assert model_dir.is_dir()

        link = model_dir / "model.onnx"
        assert link.is_symlink()
        assert link.resolve().exists()

    def test_resolve_handles_stale_symlinks(self, tmp_path: Path):
        store = _make_store(tmp_path)
        manifest = _write_manifest(store, "mymodel", "v1")
        registry = _make_registry(store)

        # Pre-create a stale symlink at the expected location
        model_dir = store.root / "models" / "links" / "mymodel" / "v1"
        model_dir.mkdir(parents=True, exist_ok=True)
        stale_link = model_dir / "model.onnx"
        stale_link.symlink_to("/nonexistent/path/that/does/not/exist")
        assert stale_link.is_symlink()
        assert not stale_link.exists()  # stale — target missing

        # resolve() should remove the stale symlink and recreate it
        info, resolved_dir = registry.resolve("mymodel", "v1")

        link = resolved_dir / "model.onnx"
        assert link.is_symlink()
        assert link.exists()  # points to a real blob now

    def test_resolve_injects_source_into_parameters(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(
            store, "mymodel", "v1", source="huggingface/some-repo"
        )
        registry = _make_registry(store)

        info, _ = registry.resolve("mymodel", "v1")
        assert info.parameters.get("_source") == "huggingface/some-repo"

    def test_resolve_no_source_means_no_injection(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(store, "mymodel", "v1")  # no source
        registry = _make_registry(store)

        info, _ = registry.resolve("mymodel", "v1")
        assert "_source" not in info.parameters


# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

class TestAvailableModels:
    def test_available_models_returns_catalog(self, tmp_path: Path):
        store = _make_store(tmp_path)
        registry = _make_registry(store)

        catalog = registry.available_models()
        assert catalog is CATALOG
        assert "whisper" in catalog
        assert "kokoro" in catalog


# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------

class TestDiscoverAdapters:
    def test_discover_adapters_skips_broken_plugins(self):
        """Broken entry points are logged and skipped rather than crashing."""
        good_ep = MagicMock()
        good_ep.name = "good"
        good_ep.load.return_value = type("GoodAdapter", (), {})

        bad_ep = MagicMock()
        bad_ep.name = "broken"
        bad_ep.load.side_effect = ImportError("missing dependency")

        with patch("vox.core.registry.entry_points", return_value=[good_ep, bad_ep]):
            adapters = discover_adapters()

        assert "good" in adapters
        assert "broken" not in adapters
