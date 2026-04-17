"""Tests for vox.core.store — BlobStore, Manifest, and ManifestLayer."""

from __future__ import annotations

import io
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from vox.core.store import BlobStore, Manifest, ManifestLayer, _manifest_to_dict






def _make_store(tmp_path: Path) -> BlobStore:
    """Create a BlobStore rooted at a temp directory."""
    return BlobStore(root=tmp_path)


def _sha256(data: bytes) -> str:
    return f"sha256-{hashlib.sha256(data).hexdigest()}"


def _save_minimal_manifest(store: BlobStore, name: str, tag: str, digest: str, size: int) -> Manifest:
    """Persist a one-layer manifest with the required config keys."""
    manifest = Manifest(
        layers=[ManifestLayer(media_type="application/vox.model.onnx", digest=digest, size=size, filename="model.onnx")],
        config={"type": "stt", "format": "onnx", "adapter": "whisper", "architecture": "whisper"},
    )
    store.save_manifest(name, tag, manifest)
    return manifest






class TestWriteBlob:
    def test_write_blob_computes_correct_sha256(self, tmp_path: Path):
        store = _make_store(tmp_path)
        data = b"hello world"
        digest = store.write_blob(io.BytesIO(data))
        assert digest == _sha256(data)
        assert store.get_blob_path(digest).read_bytes() == data

    def test_write_blob_idempotent_same_content(self, tmp_path: Path):
        store = _make_store(tmp_path)
        data = b"same content"
        d1 = store.write_blob(io.BytesIO(data))
        d2 = store.write_blob(io.BytesIO(data))
        assert d1 == d2

        blob_files = list(store.blobs_dir.iterdir())
        assert len(blob_files) == 1

    def test_write_blob_cleans_up_temp_on_exception(self, tmp_path: Path):
        store = _make_store(tmp_path)
        store.blobs_dir.mkdir(parents=True, exist_ok=True)

        class ExplodingIO(io.BytesIO):
            """Raises after the first read."""
            _first = True

            def read(self, n=-1):
                if self._first:
                    self._first = False
                    return b"partial"
                raise OSError("disk on fire")

        with pytest.raises(OSError, match="disk on fire"):
            store.write_blob(ExplodingIO())


        remaining = list(store.blobs_dir.iterdir())
        assert remaining == []






class TestManifestOperations:
    def test_save_manifest_and_resolve_roundtrip(self, tmp_path: Path):
        store = _make_store(tmp_path)
        original = Manifest(
            layers=[
                ManifestLayer(media_type="application/vox.model.onnx", digest="sha256-aaa", size=100, filename="m.onnx"),
                ManifestLayer(media_type="application/vox.voices", digest="sha256-bbb", size=50, filename="voices.bin"),
            ],
            config={"type": "tts", "format": "onnx", "adapter": "kokoro"},
        )
        store.save_manifest("kokoro", "latest", original)
        loaded = store.resolve_model("kokoro", "latest")

        assert loaded is not None
        assert loaded.schema_version == original.schema_version
        assert len(loaded.layers) == 2
        assert loaded.layers[0].digest == "sha256-aaa"
        assert loaded.layers[1].size == 50
        assert loaded.config["adapter"] == "kokoro"

    def test_resolve_model_returns_none_for_missing(self, tmp_path: Path):
        store = _make_store(tmp_path)
        assert store.resolve_model("nonexistent") is None

    def test_list_models_returns_all_stored(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _save_minimal_manifest(store, "whisper", "large-v3", "sha256-aaa", 100)
        _save_minimal_manifest(store, "whisper", "tiny", "sha256-bbb", 20)
        _save_minimal_manifest(store, "kokoro", "latest", "sha256-ccc", 50)

        models = store.list_models()
        full_names = {m.full_name for m in models}
        assert full_names == {"whisper:large-v3", "whisper:tiny", "kokoro:latest"}

    def test_list_models_empty_when_no_dir(self, tmp_path: Path):
        store = _make_store(tmp_path)

        assert store.list_models() == []

    def test_list_models_skips_corrupted_manifests_with_warning(self, tmp_path: Path, caplog):
        store = _make_store(tmp_path)

        _save_minimal_manifest(store, "whisper", "good", "sha256-aaa", 100)

        bad_dir = store.manifests_dir / "whisper"
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "bad").write_text("{not json!!!")

        with caplog.at_level(logging.WARNING, logger="vox.core.store"):
            models = store.list_models()

        assert len(models) == 1
        assert models[0].tag == "good"
        assert any("Skipping corrupted manifest" in msg for msg in caplog.messages)






class TestDeleteModel:
    def test_delete_model_removes_manifest(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _save_minimal_manifest(store, "whisper", "latest", "sha256-aaa", 100)
        assert store.resolve_model("whisper", "latest") is not None

        store.delete_model("whisper", "latest")
        assert store.resolve_model("whisper", "latest") is None

    def test_delete_model_removes_empty_parent_dir(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _save_minimal_manifest(store, "whisper", "latest", "sha256-aaa", 100)
        parent = store.manifests_dir / "whisper"
        assert parent.is_dir()

        store.delete_model("whisper", "latest")
        assert not parent.exists(), "Empty parent directory should be removed"






class TestGcBlobs:
    def test_gc_blobs_removes_unreferenced(self, tmp_path: Path):
        store = _make_store(tmp_path)

        d1 = store.write_blob(io.BytesIO(b"referenced"))
        d2 = store.write_blob(io.BytesIO(b"orphan"))


        _save_minimal_manifest(store, "whisper", "latest", d1, 10)

        removed = store.gc_blobs()
        assert removed == 1
        assert store.has_blob(d1)
        assert not store.has_blob(d2)

    def test_gc_blobs_keeps_referenced(self, tmp_path: Path):
        store = _make_store(tmp_path)
        d1 = store.write_blob(io.BytesIO(b"keep me"))
        _save_minimal_manifest(store, "whisper", "latest", d1, 7)

        removed = store.gc_blobs()
        assert removed == 0
        assert store.has_blob(d1)

    def test_gc_blobs_removes_old_temp_files(self, tmp_path: Path):
        store = _make_store(tmp_path)
        store.blobs_dir.mkdir(parents=True, exist_ok=True)

        old_tmp = store.blobs_dir / "something.tmp"
        old_tmp.write_bytes(b"stale")

        old_mtime = time.time() - 7200
        os.utime(old_tmp, (old_mtime, old_mtime))

        removed = store.gc_blobs()
        assert removed == 1
        assert not old_tmp.exists()

    def test_gc_blobs_keeps_recent_temp_files(self, tmp_path: Path):
        store = _make_store(tmp_path)
        store.blobs_dir.mkdir(parents=True, exist_ok=True)

        recent_tmp = store.blobs_dir / "inprogress.tmp"
        recent_tmp.write_bytes(b"still writing")


        removed = store.gc_blobs()
        assert removed == 0
        assert recent_tmp.exists()






class TestManifestLayerValidation:
    def test_manifest_layer_rejects_invalid_digest(self):
        with pytest.raises(ValueError, match="Invalid digest format"):
            ManifestLayer(media_type="application/octet-stream", digest="md5-abc", size=10, filename="f")

    def test_manifest_layer_rejects_negative_size(self):
        with pytest.raises(ValueError, match="Invalid layer size"):
            ManifestLayer(media_type="application/octet-stream", digest="sha256-abc", size=-1, filename="f")
