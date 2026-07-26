from __future__ import annotations

import json
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest

from vox.core.atomic_install import (
    DirectorySwap,
    bind_install_transaction,
    publish_staged_directory,
    staged_directory,
)
from vox.core.pull_transaction import PullTransaction, recover_pull_transactions
from vox.core.store import BlobStore, Manifest, ManifestLayer


def _manifest(store: BlobStore, content: bytes, source: str, transaction_id: str | None = None) -> Manifest:
    digest = store.write_blob(BytesIO(content))
    return Manifest(
        layers=[
            ManifestLayer(
                media_type="application/vox.model.bin",
                digest=digest,
                size=len(content),
                filename="model.bin",
            )
        ],
        config={
            "architecture": "fake",
            "type": "tts",
            "adapter": "fake",
            "format": "onnx",
            "source": source,
        },
        transaction_id=transaction_id,
    )


def _publish_replacement(target: Path, value: str) -> DirectorySwap:
    with staged_directory(target) as stage:
        (stage / "version.txt").write_text(value)
        return publish_staged_directory(
            stage,
            target,
            preserve_existing=False,
            retain_backup=True,
        )


def test_preparing_pull_recovery_reverses_repeated_swaps_and_manifest(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    previous = _manifest(store, b"stable", "owner/stable")
    replacement = _manifest(store, b"replacement", "owner/replacement")
    store.save_manifest("model", "latest", previous)
    runtime = tmp_path / "runtime" / "fake"
    runtime.mkdir(parents=True)
    (runtime / "version.txt").write_text("stable")
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=previous,
            candidate_digests=tuple(layer.digest for layer in replacement.layers),
        )
        with bind_install_transaction(transaction):
            _publish_replacement(runtime, "intermediate")
            _publish_replacement(runtime, "replacement")

    assert recover_pull_transactions(store) == 1
    assert store.resolve_model("model", "latest") == previous
    assert (runtime / "version.txt").read_text() == "stable"
    assert store.has_blob(previous.layers[0].digest)
    assert not store.has_blob(replacement.layers[0].digest)
    assert not transaction.path.exists()


def test_committed_pull_recovery_keeps_newest_swap_and_manifest(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    previous = _manifest(store, b"stable", "owner/stable")
    replacement = _manifest(store, b"replacement", "owner/replacement")
    store.save_manifest("model", "latest", previous)
    runtime = tmp_path / "runtime" / "fake"
    runtime.mkdir(parents=True)
    (runtime / "version.txt").write_text("stable")
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=previous,
            candidate_digests=tuple(layer.digest for layer in replacement.layers),
        )
        with bind_install_transaction(transaction):
            _publish_replacement(runtime, "intermediate")
            _publish_replacement(runtime, "replacement")
        replacement.transaction_id = transaction.id
        store.save_manifest("model", "latest", replacement)
        transaction.mark_committed()

    assert recover_pull_transactions(store) == 1
    assert store.resolve_model("model", "latest") == replacement
    assert (runtime / "version.txt").read_text() == "replacement"
    assert not transaction.path.exists()


def test_preparing_clean_install_recovery_removes_uncommitted_target(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    replacement = _manifest(store, b"replacement", "owner/replacement")
    runtime = tmp_path / "runtime" / "fake"
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=None,
            candidate_digests=tuple(layer.digest for layer in replacement.layers),
        )
        with bind_install_transaction(transaction):
            _publish_replacement(runtime, "replacement")

    assert recover_pull_transactions(store) == 1
    assert store.resolve_model("model", "latest") is None
    assert not runtime.exists()
    assert not transaction.path.exists()


def test_committed_recovery_completes_remaining_swap_cleanup(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    previous = _manifest(store, b"stable", "owner/stable")
    replacement = _manifest(store, b"replacement", "owner/replacement")
    store.save_manifest("model", "latest", previous)
    first = tmp_path / "runtime" / "first"
    second = tmp_path / "runtime" / "second"
    for target in (first, second):
        target.mkdir(parents=True)
        (target / "version.txt").write_text("stable")
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=previous,
            candidate_digests=tuple(layer.digest for layer in replacement.layers),
        )
        with bind_install_transaction(transaction):
            first_swap = _publish_replacement(first, "replacement")
            _publish_replacement(second, "replacement")
        replacement.transaction_id = transaction.id
        store.save_manifest("model", "latest", replacement)
        transaction.mark_committed()
        first_swap.commit()

    assert recover_pull_transactions(store) == 1
    assert store.resolve_model("model", "latest") == replacement
    assert (first / "version.txt").read_text() == "replacement"
    assert (second / "version.txt").read_text() == "replacement"
    assert tuple((tmp_path / "runtime").glob(".*.previous-*")) == ()
    assert tuple((tmp_path / "runtime").glob(".*.committed-*")) == ()


def test_recovery_removes_unpublished_transaction_temp_files(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    transaction_root = tmp_path / ".transactions" / "pulls"
    transaction_root.mkdir(parents=True)
    temporary = transaction_root / ".abandoned.json.123.tmp"
    temporary.write_text('{"state":"preparing"}')

    assert recover_pull_transactions(store) == 0
    assert not temporary.exists()
    assert not transaction_root.exists()


def test_second_pull_transaction_cannot_begin_while_first_is_preparing(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    with store.writer_lease():
        first = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=None,
            candidate_digests=(),
        )

        with pytest.raises(RuntimeError, match="active pull transaction"):
            PullTransaction.begin(
                store=store,
                name="model",
                tag="latest",
                previous_manifest=None,
                candidate_digests=(),
            )

        first.rollback()


def test_recovery_cannot_enter_while_another_process_owns_the_pull(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    code = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from vox.core.pull_transaction import recover_pull_transactions",
            "from vox.core.store import BlobStore, StoreWriterBusyError",
            "store = BlobStore(root=Path(sys.argv[1]))",
            "try:",
            "    recover_pull_transactions(store, timeout=0.1)",
            "except StoreWriterBusyError:",
            "    raise SystemExit(0)",
            "raise SystemExit(1)",
        )
    )

    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=None,
            candidate_digests=(),
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            cwd=Path(__file__).parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        assert transaction.path.exists()
        transaction.rollback()

    assert result.returncode == 0, result.stderr


def test_pull_journal_roots_candidate_blobs_during_preparation(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    candidate = _manifest(store, b"candidate", "owner/candidate")
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=None,
            candidate_digests=tuple(layer.digest for layer in candidate.layers),
        )

        assert (
            store.gc_blobs(
                candidates=tuple(layer.digest for layer in candidate.layers),
                grace_seconds=0,
            )
            == 0
        )
        assert store.has_blob(candidate.layers[0].digest)

        transaction.rollback()


def test_pull_journal_roots_previous_blobs_after_manifest_commit(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    previous = _manifest(store, b"previous", "owner/previous")
    candidate = _manifest(store, b"candidate", "owner/candidate")
    store.save_manifest("model", "latest", previous)
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=previous,
            candidate_digests=tuple(layer.digest for layer in candidate.layers),
        )
        candidate.transaction_id = transaction.id
        store.save_manifest("model", "latest", candidate)

        assert (
            store.gc_blobs(
                candidates=tuple(layer.digest for layer in previous.layers),
                grace_seconds=0,
            )
            == 0
        )
        assert store.has_blob(previous.layers[0].digest)

        transaction.mark_committed()
        transaction.finish()


def test_committed_recovery_promotes_a_durable_stage_when_target_is_missing(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    stage = tmp_path / "runtime" / ".fake.installing-test"
    target = tmp_path / "runtime" / "fake"
    stage.mkdir(parents=True)
    (stage / "version.txt").write_text("candidate")
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=None,
            candidate_digests=(),
        )
        transaction.record_swap(stage=stage, target=target, backup=None)
        candidate = Manifest(
            layers=[],
            config={
                "architecture": "fake",
                "type": "tts",
                "adapter": "fake",
                "format": "onnx",
            },
            transaction_id=transaction.id,
        )
        transaction.record_candidate_manifest(candidate)
        store.save_manifest("model", "latest", candidate)
        transaction.mark_committed()

    assert recover_pull_transactions(store) == 1
    assert (target / "version.txt").read_text() == "candidate"


def test_pull_transaction_rejects_malformed_candidate_digest(tmp_path: Path):
    store = BlobStore(root=tmp_path)

    with store.writer_lease(), pytest.raises(ValueError, match="digest"):
        PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=None,
            candidate_digests=("sha256-../../outside",),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"version": 1}, "version"),
        ({"state": "finished"}, "state"),
        ({"name": "../escape"}, "model name"),
        ({"tag": "bad/tag"}, "model tag"),
        ({"unexpected": True}, "schema"),
        (
            {
                "swaps": [
                    {
                        "stage": "runtime/.fake.installing-test",
                        "target": "voices/fake",
                        "backup": None,
                    }
                ]
            },
            "path",
        ),
    ),
)
def test_recovery_rejects_malformed_journal_schema(
    tmp_path: Path,
    changes: dict,
    message: str,
):
    store = BlobStore(root=tmp_path)
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=None,
            candidate_digests=(),
        )

    payload = json.loads(transaction.path.read_text())
    payload.update(changes)
    transaction.path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        recover_pull_transactions(store)


def test_recovery_rejects_a_swap_through_a_symlinked_runtime_root(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    stage_name = ".fake.installing-test"
    local_stage = runtime / stage_name
    local_stage.mkdir()
    with store.writer_lease():
        transaction = PullTransaction.begin(
            store=store,
            name="model",
            tag="latest",
            previous_manifest=None,
            candidate_digests=(),
        )
        transaction.record_swap(
            stage=local_stage,
            target=runtime / "fake",
            backup=None,
        )

    external = tmp_path.parent / f"{tmp_path.name}-external"
    retained = tmp_path / "runtime-retained"
    runtime.rename(retained)
    external.mkdir()
    stage = external / stage_name
    stage.mkdir()
    marker = stage / "marker.txt"
    marker.write_text("keep")
    runtime.symlink_to(external, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="path"):
            recover_pull_transactions(store)
        assert marker.read_text() == "keep"
    finally:
        shutil.rmtree(external)
