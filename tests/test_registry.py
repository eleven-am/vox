"""Tests for ModelRegistry: catalog lookup, adapter discovery, and model resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox.core.errors import AdapterNotFoundError, ModelNotFoundError
from vox.core.registry import (
    BUNDLED_ADAPTERS_ENV,
    BUNDLED_ADAPTERS_NO_DEPS_ENV,
    CATALOG,
    ModelRegistry,
    _find_bundled_adapter_source,
    discover_adapters,
    install_adapter_package,
)
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

    def test_resolve_creates_parent_dirs_for_nested_filenames(self, tmp_path: Path):
        store = _make_store(tmp_path)
        digest = "sha256-" + "cd" * 32
        blob_path = store.blobs_dir / digest
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"nested-model")
        manifest = Manifest(
            layers=[
                ManifestLayer(
                    media_type="application/vox.model.onnx",
                    digest=digest,
                    size=12,
                    filename="onnx/model.onnx",
                )
            ],
            config={
                "architecture": "test-arch",
                "type": "tts",
                "adapter": "kokoro",
                "format": "onnx",
                "parameters": {"sample_rate": 24000},
            },
        )
        store.save_manifest("kokoro", "v1", manifest)
        registry = _make_registry(store)

        _, model_dir = registry.resolve("kokoro", "v1")

        link = model_dir / "onnx" / "model.onnx"
        assert link.is_symlink()
        assert link.resolve().exists()

    def test_resolve_handles_stale_symlinks(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _write_manifest(store, "mymodel", "v1")
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

    def test_whisper_catalog_uses_ct2_and_whisper_adapter_package(self):
        whisper = CATALOG["whisper"]

        assert set(whisper) == {"large-v3", "large-v3-turbo", "base.en"}
        for _tag, entry in whisper.items():
            assert entry["adapter_package"] == "vox-whisper"
            assert entry["format"] == "ct2"
            assert entry["adapter"] == "whisper"
            assert entry["type"] == "stt"
            assert entry["parameters"]["sample_rate"] == 16000

    def test_sesame_catalog_entry_has_default_voice(self):
        sesame = CATALOG["sesame"]["csm-1b"]

        assert sesame["adapter_package"] == "vox-sesame"
        assert sesame["parameters"]["sample_rate"] == 24_000
        assert sesame["parameters"]["default_voice"] == "0"

    def test_parakeet_nemo_catalog_entry_is_explicit_and_pytorch(self):
        parakeet_nemo = CATALOG["parakeet"]["tdt-0.6b-v3-nemo"]

        assert parakeet_nemo["adapter_package"] == "vox-parakeet"
        assert parakeet_nemo["adapter"] == "parakeet-nemo"
        assert parakeet_nemo["format"] == "pytorch"
        assert parakeet_nemo["files"] == ["parakeet-tdt-0.6b-v3.nemo"]
        assert parakeet_nemo["parameters"]["sample_rate"] == 16_000

    def test_parakeet_cuda_alias_points_to_nemo_backend(self):
        parakeet_cuda = CATALOG["parakeet"]["tdt-0.6b-v3-cuda"]

        assert parakeet_cuda["adapter_package"] == "vox-parakeet"
        assert parakeet_cuda["adapter"] == "parakeet-nemo"
        assert parakeet_cuda["format"] == "pytorch"
        assert parakeet_cuda["files"] == ["parakeet-tdt-0.6b-v3.nemo"]

    def test_kokoro_torch_catalog_entry_is_explicit_and_pytorch(self):
        kokoro_torch = CATALOG["kokoro"]["v1.0-torch"]

        assert kokoro_torch["adapter_package"] == "vox-kokoro"
        assert kokoro_torch["adapter"] == "kokoro-torch"
        assert kokoro_torch["format"] == "pytorch"
        assert kokoro_torch["files"] == ["kokoro-v1_0.pth"]
        assert kokoro_torch["parameters"]["default_voice"] == "af_heart"

    def test_openvoice_catalog_entry_has_checkpoint_files(self):
        openvoice = CATALOG["openvoice"]["v1"]

        assert openvoice["adapter_package"] == "vox-openvoice"
        assert openvoice["parameters"]["sample_rate"] == 22_050
        assert openvoice["parameters"]["default_voice"] == "en/default"
        assert "checkpoints/base_speakers/EN/config.json" in openvoice["files"]

    def test_xtts_catalog_entry_uses_huggingface_repo_id(self):
        xtts = CATALOG["xtts"]["v2"]

        assert xtts["source"] == "coqui/XTTS-v2"
        assert xtts["adapter_package"] == "vox-xtts"


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


class TestBundledAdapters:
    def test_find_bundled_adapter_source_uses_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        bundled_root = tmp_path / "bundled"
        adapter_dir = bundled_root / "vox-kokoro"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "pyproject.toml").write_text("[project]\nname='vox-kokoro'\n", encoding="utf-8")

        monkeypatch.setenv(BUNDLED_ADAPTERS_ENV, str(bundled_root))

        assert _find_bundled_adapter_source("vox-kokoro") == adapter_dir

    def test_install_adapter_package_prefers_bundled_source(self, tmp_path: Path):
        store = _make_store(tmp_path)
        bundled_adapter = tmp_path / "adapters" / "vox-kokoro"
        bundled_adapter.mkdir(parents=True)
        (bundled_adapter / "pyproject.toml").write_text("[project]\nname='vox-kokoro'\n", encoding="utf-8")

        calls: list[list[str]] = []

        def _fake_run(cmd: list[str], **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with (
            patch("vox.core.registry._find_bundled_adapter_source", return_value=bundled_adapter),
            patch("vox.core.registry.subprocess.run", side_effect=_fake_run),
        ):
            assert install_adapter_package("vox-kokoro", store.root) is True

        assert calls == [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(store.root / "adapters"),
                str(bundled_adapter),
            ]
        ]

    def test_install_adapter_package_can_skip_bundled_dependencies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        store = _make_store(tmp_path)
        bundled_adapter = tmp_path / "adapters" / "vox-kokoro"
        bundled_adapter.mkdir(parents=True)
        (bundled_adapter / "pyproject.toml").write_text("[project]\nname='vox-kokoro'\n", encoding="utf-8")
        monkeypatch.setenv(BUNDLED_ADAPTERS_NO_DEPS_ENV, "1")

        calls: list[list[str]] = []

        def _fake_run(cmd: list[str], **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with (
            patch("vox.core.registry._find_bundled_adapter_source", return_value=bundled_adapter),
            patch("vox.core.registry.subprocess.run", side_effect=_fake_run),
        ):
            assert install_adapter_package("vox-kokoro", store.root) is True

        assert calls == [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(store.root / "adapters"),
                "--no-deps",
                str(bundled_adapter),
            ]
        ]
