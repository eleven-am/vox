"""Tests for vox.core.store — BlobStore, Manifest, and ManifestLayer."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vox.core.store import BlobStore, Manifest, ManifestLayer


def _make_store(tmp_path: Path) -> BlobStore:
    """Create a BlobStore rooted at a temp directory."""
    return BlobStore(root=tmp_path)


def _sha256(data: bytes) -> str:
    return f"sha256-{hashlib.sha256(data).hexdigest()}"


def _save_minimal_manifest(store: BlobStore, name: str, tag: str, digest: str, size: int) -> Manifest:
    """Persist a one-layer manifest with the required config keys."""
    manifest = Manifest(
        layers=[
            ManifestLayer(media_type="application/vox.model.onnx", digest=digest, size=size, filename="model.onnx"),
        ],
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


class TestStoreWriter:
    def test_writer_lease_excludes_another_process(self, tmp_path: Path):
        store = _make_store(tmp_path)
        code = "\n".join(
            (
                "import sys",
                "from pathlib import Path",
                "from vox.core.store import BlobStore, StoreWriterBusyError",
                "store = BlobStore(root=Path(sys.argv[1]))",
                "try:",
                "    store.acquire_writer_lease(timeout=0.1)",
                "except StoreWriterBusyError:",
                "    raise SystemExit(0)",
                "raise SystemExit(1)",
            )
        )

        with store.writer_lease():
            result = subprocess.run(
                [sys.executable, "-c", code, str(tmp_path)],
                cwd=Path(__file__).parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )

        assert result.returncode == 0, result.stderr

    def test_fenced_writer_cannot_publish_a_manifest(self, tmp_path: Path):
        store = _make_store(tmp_path)
        manifest = Manifest(config={"type": "stt", "format": "onnx"})

        with store.writer_lease() as writer:
            epoch_path = store.transaction_root / "store-writer.epoch"
            epoch_path.write_text(str(writer.epoch + 1))

            with pytest.raises(RuntimeError, match="fenced"):
                store.save_manifest("model", "latest", manifest)

        assert store.resolve_model("model", "latest") is None


class TestManifestOperations:
    def test_save_manifest_and_resolve_roundtrip(self, tmp_path: Path):
        store = _make_store(tmp_path)
        original = Manifest(
            layers=[
                ManifestLayer(
                    media_type="application/vox.model.onnx",
                    digest="sha256-aaa",
                    size=100,
                    filename="m.onnx",
                ),
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

    def test_save_manifest_dotted_tags_do_not_collide(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _save_minimal_manifest(store, "parakeet", "tdt-0.6b", "sha256-aaa", 100)
        _save_minimal_manifest(store, "parakeet", "tdt-0.6b-v3", "sha256-bbb", 200)

        first = store.resolve_model("parakeet", "tdt-0.6b")
        second = store.resolve_model("parakeet", "tdt-0.6b-v3")
        assert first is not None and first.layers[0].digest == "sha256-aaa"
        assert second is not None and second.layers[0].digest == "sha256-bbb"
        leftover = list((store.manifests_dir / "parakeet").glob("*.tmp"))
        assert leftover == []

    def test_manifest_staging_files_are_not_visible_as_models(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _save_minimal_manifest(store, "model", "latest", "sha256-aaa", 100)
        staging = store.manifest_staging_dir
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "latest.crashed.tmp").write_text("{}")

        models = store.list_models()

        assert [model.full_name for model in models] == ["model:latest"]

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

    def test_list_models_skips_manifest_with_malformed_layer(self, tmp_path: Path):
        store = _make_store(tmp_path)
        _save_minimal_manifest(store, "whisper", "good", "sha256-aaa", 100)

        bad_dir = store.manifests_dir / "whisper"
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "bad").write_text(
            json.dumps({"schema_version": 1, "layers": [{"digest": "sha256-x", "size": 1}], "config": {}})
        )

        models = store.list_models()
        assert [m.tag for m in models] == ["good"]


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
    def test_gc_blobs_reads_layer_roots_without_model_info_construction(
        self,
        tmp_path: Path,
    ):
        store = _make_store(tmp_path)
        referenced = store.write_blob(io.BytesIO(b"referenced"))
        orphan = store.write_blob(io.BytesIO(b"orphan"))
        path = store.manifests_dir / "broken" / "latest"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "layers": [
                        {
                            "media_type": "application/vox.model.bin",
                            "digest": referenced,
                            "size": 10,
                            "filename": "model.bin",
                        }
                    ],
                    "config": {},
                }
            )
        )
        old_mtime = time.time() - 7200
        os.utime(store.get_blob_path(referenced), (old_mtime, old_mtime))
        os.utime(store.get_blob_path(orphan), (old_mtime, old_mtime))

        assert store.list_models() == []
        assert store.gc_blobs(grace_seconds=0) == 1
        assert store.has_blob(referenced)
        assert not store.has_blob(orphan)

    def test_gc_blobs_deletes_nothing_when_a_manifest_is_unreadable(
        self,
        tmp_path: Path,
    ):
        store = _make_store(tmp_path)
        orphan = store.write_blob(io.BytesIO(b"orphan"))
        path = store.manifests_dir / "broken" / "latest"
        path.parent.mkdir(parents=True)
        path.write_text("{")
        old_mtime = time.time() - 7200
        os.utime(store.get_blob_path(orphan), (old_mtime, old_mtime))

        assert store.gc_blobs(grace_seconds=0) == 0
        assert store.has_blob(orphan)

    @pytest.mark.parametrize("broken", (False, True))
    def test_gc_blobs_deletes_nothing_when_manifest_root_is_a_symlink(
        self,
        tmp_path: Path,
        broken: bool,
    ):
        store = _make_store(tmp_path)
        orphan = store.write_blob(io.BytesIO(b"orphan"))
        target = tmp_path / "external-manifests"
        if not broken:
            target.mkdir()
        store.manifests_dir.parent.mkdir(parents=True)
        store.manifests_dir.symlink_to(target, target_is_directory=True)
        old_mtime = time.time() - 7200
        os.utime(store.get_blob_path(orphan), (old_mtime, old_mtime))

        assert store.gc_blobs(grace_seconds=0) == 0
        assert store.has_blob(orphan)

    def test_gc_blobs_removes_unreferenced(self, tmp_path: Path):
        store = _make_store(tmp_path)

        d1 = store.write_blob(io.BytesIO(b"referenced"))
        d2 = store.write_blob(io.BytesIO(b"orphan"))

        _save_minimal_manifest(store, "whisper", "latest", d1, 10)

        old_mtime = time.time() - 7200
        os.utime(store.get_blob_path(d2), (old_mtime, old_mtime))

        removed = store.gc_blobs()
        assert removed == 1
        assert store.has_blob(d1)
        assert not store.has_blob(d2)

    def test_gc_blobs_keeps_recent_unreferenced_blob(self, tmp_path: Path):
        store = _make_store(tmp_path)

        d1 = store.write_blob(io.BytesIO(b"referenced"))
        d2 = store.write_blob(io.BytesIO(b"just-downloaded-mid-pull"))
        _save_minimal_manifest(store, "whisper", "latest", d1, 10)

        removed = store.gc_blobs()
        assert removed == 0
        assert store.has_blob(d2)

    def test_gc_blobs_keeps_old_deduplicated_blob_until_manifest_publication(self, tmp_path: Path):
        store = _make_store(tmp_path)
        data = b"deduplicated-model"
        digest = store.write_blob(io.BytesIO(data))
        old_mtime = time.time() - 7200
        os.utime(store.get_blob_path(digest), (old_mtime, old_mtime))

        with store.blob_lease() as lease:
            leased_digest = lease.write_blob(io.BytesIO(data))

            assert leased_digest == digest
            assert store.gc_blobs() == 0
            assert store.has_blob(digest)
            _save_minimal_manifest(store, "whisper", "latest", digest, len(data))

        assert store.gc_blobs() == 0
        assert store.has_blob(digest)

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
