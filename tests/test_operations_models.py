from __future__ import annotations

import asyncio
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vox.core import adapter_runtime
from vox.core.pull_transaction import PullTransaction, recover_pull_transactions
from vox.core.runtime import RuntimeCapabilities
from vox.core.store import BlobLease, BlobStore, Manifest, ManifestLayer
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
    PullTaskRegistry,
    ShowResult,
    _PullPublication,
    _stage_adapter_mutation_owned,
    _stage_runtime_mutation_owned,
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


async def _collect_events(events):
    return [event async for event in events]


@pytest.mark.asyncio
async def test_pull_progress_is_bounded_when_the_consumer_is_slow():
    tasks = PullTaskRegistry()

    async def operation(emit):
        for index in range(200):
            emit(PullEvent(status=str(index)))

    events = tasks.start(operation)
    collected = [event async for event in events]
    await asyncio.sleep(0)

    assert len(collected) <= 63
    assert collected[-1].status == "199"
    assert tasks.active_count == 0


def test_pull_publication_does_not_report_cleanup_failure_after_logical_commit(
    tmp_path: Path,
    caplog,
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
        runtime_mutation = MagicMock()
        runtime_mutation.commit.side_effect = RuntimeError("runtime commit failed")
        adapter_mutation = MagicMock()
        blob_lease = MagicMock()
        publication = _PullPublication(
            adapter_mutation=adapter_mutation,
            runtime_mutation=runtime_mutation,
            blob_lease=blob_lease,
            transaction=transaction,
        )

        publication.commit()

    assert publication.committed is True
    runtime_mutation.commit.assert_called_once_with()
    adapter_mutation.commit.assert_called_once_with()
    blob_lease.close.assert_called_once_with()
    assert not transaction.path.exists()
    assert "pull publication cleanup deferred" in caplog.text


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
    store.save_manifest(
        "foo",
        "latest",
        Manifest(
            layers=[],
            config={
                "architecture": "fake",
                "type": "stt",
                "adapter": "fake",
                "format": "onnx",
            },
        ),
    )
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


@pytest.mark.asyncio
async def test_delete_waits_for_the_store_writer_before_unloading(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    manifest = Manifest(
        layers=[],
        config={
            "architecture": "fake",
            "type": "stt",
            "adapter": "fake",
            "format": "onnx",
        },
    )
    store.save_manifest("foo", "latest", manifest)
    scheduler = MagicMock()
    scheduler.unload = AsyncMock(return_value=True)
    writer = store.acquire_writer_lease()
    task = asyncio.create_task(
        delete_model(
            store=store,
            scheduler=scheduler,
            registry=_registry_mock(),
            request=model_reference_request_from_fields(name="foo:latest"),
        )
    )
    try:
        await asyncio.sleep(0.02)
        scheduler.unload.assert_not_awaited()
    finally:
        writer.close()

    await asyncio.wait_for(task, timeout=1)
    scheduler.unload.assert_awaited_once_with("foo:latest")


@pytest.mark.asyncio
async def test_cancelled_delete_closes_writer_acquired_after_cancellation(
    tmp_path: Path,
    monkeypatch,
):
    store = BlobStore(root=tmp_path)
    store.save_manifest(
        "foo",
        "latest",
        Manifest(
            layers=[],
            config={
                "architecture": "fake",
                "type": "stt",
                "adapter": "fake",
                "format": "onnx",
            },
        ),
    )
    holder = store.acquire_writer_lease()
    acquired = []
    original_acquire = store.acquire_writer_lease

    def capture_acquire(*, timeout: float = 30.0):
        lease = original_acquire(timeout=timeout)
        acquired.append(lease)
        return lease

    monkeypatch.setattr(store, "acquire_writer_lease", capture_acquire)
    task = asyncio.create_task(
        delete_model(
            store=store,
            scheduler=MagicMock(unload=AsyncMock(return_value=True)),
            registry=_registry_mock(),
            request=model_reference_request_from_fields(name="foo:latest"),
        )
    )
    await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    holder.close()
    with pytest.raises(asyncio.CancelledError):
        await task

    probe = None
    try:
        probe = original_acquire(timeout=0.05)
    finally:
        if probe is not None:
            probe.close()
        for lease in acquired:
            lease.close()

    assert probe is not None


def test_pull_model_unknown_catalog_raises(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = None
    with pytest.raises(CatalogEntryNotFoundError):
        pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="missing:latest"),
            tasks=PullTaskRegistry(),
        )


def test_pull_model_blocks_incompatible_model(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VOX_ALLOW_INCOMPATIBLE", raising=False)
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "qwen3-tts-torch",
        "format": "pytorch",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "vox-qwen",
    }
    with (
        patch(
            "vox.core.model_resolution.detect_runtime_capabilities",
            return_value=_runtime_caps(torch_installed=False, torch_cuda=False, torch_mps_available=False, mps=False),
        ),
        pytest.raises(ModelIncompatibleError, match="PyTorch"),
    ):
        pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="qwen3-tts:0.6b"),
            tasks=PullTaskRegistry(),
        )


@pytest.mark.asyncio
async def test_pull_model_override_allows_incompatible_model(tmp_path: Path, monkeypatch):
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
        "adapter_package": "vox-qwen",
    }
    # gate is bypassed -> pull_model returns the async generator without raising
    tasks = PullTaskRegistry()
    with patch(
        "vox.core.model_resolution.detect_runtime_capabilities",
        return_value=_runtime_caps(torch_installed=False, torch_cuda=False, torch_mps_available=False, mps=False),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="qwen3-tts:0.6b"),
            tasks=tasks,
        )
    assert events is not None
    await tasks.close(deadline=asyncio.get_running_loop().time() + 1)


@pytest.mark.asyncio
async def test_pull_model_override_emits_missing_capability_warning(tmp_path: Path, monkeypatch):
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
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch(
            "vox.core.model_resolution.detect_runtime_capabilities",
            return_value=_runtime_caps(torch_installed=False, torch_cuda=False, torch_mps_available=False, mps=False),
        ),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="qwen3-tts:0.6b"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    warning = next(event for event in collected if event.status == "warning")
    assert "VOX_ALLOW_INCOMPATIBLE=1" in warning.error
    assert "PyTorch" in warning.error
    assert collected[-1].status == "success"


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
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("huggingface_hub.HfApi") as mock_api_cls,
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        mock_api_cls.return_value.repo_info.return_value = MagicMock(siblings=[MagicMock(rfilename="model.bin")])
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    assert collected[-1].status == "success"
    assert any(e.status.startswith("downloading") for e in collected)
    manifest = store.resolve_model("foo", "latest")
    assert manifest is not None
    assert manifest.config["runtime"]["checked_at_pull"] is True
    assert manifest.config["runtime"]["resolved_variant"] == ""


@pytest.mark.asyncio
async def test_pull_model_prepares_adapter_runtime_after_download(tmp_path: Path):
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
    prepared: list[str] = []

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            prepared.append("runtime")

    registry.ensure_adapter.return_value = True
    registry.get_adapter_class.return_value = FakeAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    assert prepared == ["runtime"]
    assert any(event.status == "preparing adapter runtime fake" for event in collected)
    assert any(event.status == "adapter runtime fake ready" for event in collected)
    assert collected[-1].status == "success"


@pytest.mark.asyncio
async def test_pull_model_checks_adapter_off_event_loop(tmp_path: Path):
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
    event_loop_thread = threading.get_ident()
    install_threads: list[int] = []

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            return None

    def stage_adapter(_adapter_name: str, _package_name: str):
        install_threads.append(threading.get_ident())
        return MagicMock(ready=True)

    registry.stage_adapter.side_effect = stage_adapter
    registry.get_adapter_class.return_value = FakeAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    assert collected[-1].status == "success"
    assert install_threads
    assert set(install_threads) != {event_loop_thread}


@pytest.mark.asyncio
async def test_pull_model_does_not_save_manifest_when_runtime_prepare_fails(
    tmp_path: Path,
):
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

    class BrokenAdapter:
        def prepare_runtime(self) -> None:
            raise RuntimeError("runtime bootstrap failed")

    registry.ensure_adapter.return_value = True
    adapter_mutation = MagicMock(ready=True)
    registry.stage_adapter.return_value = adapter_mutation
    registry.get_adapter_class.return_value = BrokenAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    assert any(event.status == "preparing adapter runtime fake" for event in collected)
    assert collected[-1].status == "error"
    assert collected[-1].error == "runtime bootstrap failed"
    assert store.resolve_model("foo", "latest") is None
    assert tuple(store.blobs_dir.glob("sha256-*")) == ()
    adapter_mutation.rollback.assert_called_once_with()


@pytest.mark.asyncio
async def test_detached_progress_stream_does_not_cancel_pull(tmp_path: Path):
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

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            return None

    registry.ensure_adapter.return_value = True
    registry.get_adapter_class.return_value = FakeAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")
    tasks = PullTaskRegistry()

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=tasks,
        )
        async for event in events:
            if event.status == "preparing adapter runtime fake":
                break
        await events.aclose()
        while tasks.active_count:
            await asyncio.sleep(0)

    manifest = store.resolve_model("foo", "latest")
    assert manifest is not None
    assert store.has_blob(manifest.layers[0].digest)


@pytest.mark.asyncio
async def test_cancelled_runtime_preparation_waits_and_restores_previous_runtime(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "fake",
        "format": "onnx",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }
    runtime = tmp_path / "runtime" / "fake"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_text("stable")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingAdapter:
        def prepare_runtime(self) -> None:
            try:
                with adapter_runtime.staged_target_runtime(runtime) as stage:
                    (stage / "replacement.txt").write_text("replacement")
                    started.set()
                    assert release.wait(timeout=2)
            finally:
                finished.set()

    registry.get_adapter_class.return_value = BlockingAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")
    tasks = PullTaskRegistry()

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=tasks,
        )
        async for event in events:
            if event.status == "preparing adapter runtime fake":
                break
        while not started.is_set():
            await asyncio.sleep(0)
        close_task = asyncio.create_task(tasks.close(deadline=asyncio.get_running_loop().time() + 2))
        try:
            await asyncio.sleep(0.02)
            assert close_task.done() is False
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 2)

        await close_task

    assert (runtime / "stable.txt").read_text() == "stable"
    assert not (runtime / "replacement.txt").exists()
    assert store.resolve_model("foo", "latest") is None


@pytest.mark.asyncio
async def test_repeated_cancellation_rolls_back_late_runtime_mutation():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    mutation = MagicMock()

    def stage(*_args):
        started.set()
        try:
            assert release.wait(timeout=2)
            return mutation
        finally:
            finished.set()

    with patch(
        "vox.operations.models.stage_adapter_runtime_mutation",
        side_effect=stage,
    ):
        task = asyncio.create_task(_stage_runtime_mutation_owned(MagicMock()))
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        asyncio.get_running_loop().call_later(0.01, release.set)
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(finished.wait, 2)

    mutation.rollback.assert_called_once_with()


@pytest.mark.asyncio
async def test_repeated_cancellation_rolls_back_late_adapter_mutation():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    mutation = MagicMock()

    def stage():
        started.set()
        try:
            assert release.wait(timeout=2)
            return mutation
        finally:
            finished.set()

    task = asyncio.create_task(_stage_adapter_mutation_owned(stage))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    asyncio.get_running_loop().call_later(0.01, release.set)
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.to_thread(finished.wait, 2)

    mutation.rollback.assert_called_once_with()


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_blob_write_before_lease_abort(
    tmp_path: Path,
):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "",
        "format": "onnx",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    abort_started = threading.Event()
    original_abort = BlobLease.abort

    class BlockingReader:
        def __init__(self):
            self._complete = False

        def read(self, _size):
            if self._complete:
                return b""
            started.set()
            assert release.wait(timeout=2)
            self._complete = True
            return b"candidate"

    reader = BlockingReader()

    def write_blob(lease, _path):
        try:
            return lease.write_blob(reader)
        finally:
            finished.set()

    def track_abort(lease):
        abort_started.set()
        return original_abort(lease)

    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"candidate")
    tasks = PullTaskRegistry()
    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch(
            "vox.core.model_resolution.detect_runtime_capabilities",
            return_value=_runtime_caps(),
        ),
        patch("vox.operations.models._write_blob_from_path", side_effect=write_blob),
        patch.object(BlobLease, "abort", track_abort),
    ):
        pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=tasks,
        )
        await asyncio.to_thread(started.wait, 2)
        task = next(iter(tasks._tasks))
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        aborted_before_release = await asyncio.to_thread(
            abort_started.wait,
            0.05,
        )
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(finished.wait, 2)

    assert aborted_before_release is False
    assert tuple(store.blobs_dir.glob("sha256-*")) == ()
    assert store._leased_blobs == {}


@pytest.mark.asyncio
async def test_pull_shutdown_closes_writer_acquired_after_cancellation(
    tmp_path: Path,
    monkeypatch,
):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "",
        "format": "onnx",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }
    holder = store.acquire_writer_lease()
    acquired = []
    original_acquire = store.acquire_writer_lease

    def capture_acquire(*, timeout: float = 30.0):
        lease = original_acquire(timeout=timeout)
        acquired.append(lease)
        return lease

    monkeypatch.setattr(store, "acquire_writer_lease", capture_acquire)
    tasks = PullTaskRegistry()
    pull_model(
        store=store,
        registry=registry,
        request=model_reference_request_from_fields(name="foo:latest"),
        tasks=tasks,
    )
    await asyncio.sleep(0.02)
    close_task = asyncio.create_task(tasks.close(deadline=asyncio.get_running_loop().time() + 1))
    await asyncio.sleep(0.02)
    holder.close()
    await close_task

    probe = None
    try:
        probe = original_acquire(timeout=0.05)
    finally:
        if probe is not None:
            probe.close()
        for lease in acquired:
            lease.close()

    assert probe is not None


@pytest.mark.asyncio
async def test_progress_consumer_cannot_retain_the_runtime_mutation_lock(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "fake",
        "format": "onnx",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }
    runtime = tmp_path / "runtime" / "fake"

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            with adapter_runtime.staged_target_runtime(runtime) as stage:
                (stage / "runtime.txt").write_text("ready")

    registry.get_adapter_class.return_value = FakeAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"replacement")
    events = None
    acquired = threading.Event()
    tasks = PullTaskRegistry()

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=tasks,
        )
        async for event in events:
            if event.status == "adapter runtime fake ready":
                break

        waiter = asyncio.create_task(
            asyncio.to_thread(
                adapter_runtime.run_with_adapter_runtime_lock,
                acquired.set,
            )
        )
        try:
            assert await asyncio.to_thread(acquired.wait, 0.2)
        finally:
            await events.aclose()
            await asyncio.wait_for(waiter, timeout=1)
            while tasks.active_count:
                await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancellation_during_publication_leaves_fully_committed_pull(
    tmp_path: Path,
    monkeypatch,
):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "fake",
        "format": "onnx",
        "source": "owner/repo",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }
    runtime = tmp_path / "runtime" / "fake"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_text("stable")
    commit_started = threading.Event()
    release_commit = threading.Event()

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            with adapter_runtime.staged_target_runtime(runtime) as stage:
                (stage / "replacement.txt").write_text("replacement")

    original_commit = adapter_runtime.RuntimeMutation.commit

    def blocking_commit(mutation) -> None:
        commit_started.set()
        assert release_commit.wait(timeout=2)
        original_commit(mutation)

    monkeypatch.setattr(adapter_runtime.RuntimeMutation, "commit", blocking_commit)
    registry.get_adapter_class.return_value = FakeAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"replacement")
    tasks = PullTaskRegistry()

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=tasks,
        )
        async for event in events:
            if event.status == "adapter runtime fake ready":
                break
        await asyncio.to_thread(commit_started.wait, 2)
        close_task = asyncio.create_task(tasks.close(deadline=asyncio.get_running_loop().time() + 2))
        await asyncio.sleep(0)
        assert close_task.done() is False
        release_commit.set()
        await close_task

    manifest = store.resolve_model("foo", "latest")
    assert manifest is not None
    assert manifest.layers[0].filename == "model.bin"
    assert store.has_blob(manifest.layers[0].digest)
    assert (runtime / "replacement.txt").read_text() == "replacement"
    assert not (runtime / "stable.txt").exists()


@pytest.mark.asyncio
async def test_manifest_post_replace_failure_rolls_runtime_forward(
    tmp_path: Path,
    monkeypatch,
):
    import vox.core.store as store_module

    store = BlobStore(root=tmp_path)
    previous = Manifest(
        layers=[],
        config={
            "architecture": "fake",
            "type": "tts",
            "adapter": "fake",
            "format": "onnx",
            "source": "owner/stable",
        },
    )
    store.save_manifest("foo", "latest", previous)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "fake",
        "type": "tts",
        "adapter": "fake",
        "format": "onnx",
        "source": "owner/replacement",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }
    runtime = tmp_path / "runtime" / "fake"
    runtime.mkdir(parents=True)
    (runtime / "version.txt").write_text("stable")

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            with adapter_runtime.staged_target_runtime(runtime) as stage:
                (stage / "version.txt").write_text("replacement")

    registry.get_adapter_class.return_value = FakeAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"replacement")
    original_sync = store_module._sync_directory
    manifest_parent = store.manifests_dir / "foo"

    def fail_after_manifest_replace(path: Path) -> None:
        if path == manifest_parent:
            raise OSError("manifest directory sync failed")
        original_sync(path)

    monkeypatch.setattr(
        store_module,
        "_sync_directory",
        fail_after_manifest_replace,
    )

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch(
            "vox.core.model_resolution.detect_runtime_capabilities",
            return_value=_runtime_caps(),
        ),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="foo:latest"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    manifest = store.resolve_model("foo", "latest")
    assert collected[-1].status == "success"
    assert any(event.status == "warning" and "manifest directory sync failed" in event.error for event in collected)
    assert manifest is not None
    assert manifest.config["source"] == "owner/replacement"
    assert (runtime / "version.txt").read_text() == "replacement"
    journals = tuple((tmp_path / ".transactions" / "pulls").glob("*.json"))
    assert len(journals) == 1

    monkeypatch.setattr(store_module, "_sync_directory", original_sync)
    store.save_manifest("foo", "latest", previous)

    assert recover_pull_transactions(store) == 1
    recovered = store.resolve_model("foo", "latest")
    assert recovered is not None
    assert recovered.config["source"] == "owner/replacement"
    assert (runtime / "version.txt").read_text() == "replacement"
    assert not journals[0].exists()


@pytest.mark.asyncio
async def test_pull_does_not_publish_a_temporary_manifest_for_voxtral_preload(
    tmp_path: Path,
):
    store = BlobStore(root=tmp_path)
    previous_blob = store.write_blob(BytesIO(b"stable"))
    previous = Manifest(
        layers=[
            ManifestLayer(
                media_type="application/vox.model.bin",
                digest=previous_blob,
                size=6,
                filename="model.bin",
            )
        ],
        config={
            "architecture": "voxtral",
            "type": "tts",
            "adapter": "voxtral-tts-vllm",
            "format": "pytorch",
            "source": "owner/stable",
        },
    )
    store.save_manifest("voxtral-tts", "latest", previous)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "voxtral",
        "type": "tts",
        "adapter": "voxtral-tts-vllm",
        "format": "pytorch",
        "source": "owner/replacement",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            return None

    registry.get_adapter_class.return_value = FakeAdapter
    scheduler = MagicMock()
    scheduler.preload = AsyncMock()
    downloaded = tmp_path / "replacement.bin"
    downloaded.write_bytes(b"replacement")

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="voxtral-tts:latest"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    assert collected[-1].status == "success"
    assert store.resolve_model("voxtral-tts", "latest") != previous
    scheduler.preload.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_pull_attempts_every_rollback_owner(tmp_path: Path):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "voxtral",
        "type": "tts",
        "adapter": "voxtral-tts-vllm",
        "format": "pytorch",
        "source": "owner/replacement",
        "parameters": {},
        "adapter_package": "vox-fake",
        "files": ["model.bin"],
    }
    adapter_mutation = MagicMock(ready=True)
    runtime_mutation = MagicMock()
    runtime_mutation.rollback.side_effect = RuntimeError("runtime rollback failed")
    registry.stage_adapter.return_value = adapter_mutation
    downloaded = tmp_path / "replacement.bin"
    downloaded.write_bytes(b"replacement")

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
        patch(
            "vox.operations.models._stage_runtime_mutation_owned",
            AsyncMock(return_value=runtime_mutation),
        ),
        patch.object(
            store,
            "save_manifest",
            side_effect=RuntimeError("manifest publish failed"),
        ),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="voxtral-tts:latest"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    assert collected[-1].status == "error"
    assert "manifest publish failed" in collected[-1].error
    assert "runtime rollback failed" in collected[-1].error
    runtime_mutation.rollback.assert_called_once_with()
    adapter_mutation.rollback.assert_called_once_with()
    assert store.resolve_model("voxtral-tts", "latest") is None
    assert tuple(store.blobs_dir.glob("sha256-*")) == ()
    assert tuple((tmp_path / ".transactions" / "pulls").glob("*.json")) == ()


@pytest.mark.asyncio
async def test_voxtral_pull_publishes_its_staged_runtime_without_preload(
    tmp_path: Path,
):
    store = BlobStore(root=tmp_path)
    registry = _registry_mock()
    registry.lookup.return_value = {
        "architecture": "voxtral",
        "type": "tts",
        "adapter": "voxtral-tts-vllm",
        "format": "pytorch",
        "source": "owner/replacement",
        "parameters": {},
        "adapter_package": "",
        "files": ["model.bin"],
    }
    runtime = tmp_path / "runtime" / "voxtral"

    class FakeAdapter:
        def prepare_runtime(self) -> None:
            with adapter_runtime.staged_target_runtime(runtime) as stage:
                (stage / "runtime.txt").write_text("ready")

    registry.get_adapter_class.return_value = FakeAdapter
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"replacement")
    scheduler = MagicMock()
    scheduler.preload = AsyncMock()

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="voxtral-tts:latest"),
            tasks=PullTaskRegistry(),
        )
        collected = await asyncio.wait_for(
            _collect_events(events),
            timeout=1,
        )

    assert collected[-1].status == "success"
    assert (runtime / "runtime.txt").read_text() == "ready"
    scheduler.preload.assert_not_awaited()


@pytest.mark.asyncio
async def test_pull_model_resolves_logical_variant_and_records_runtime_metadata(
    tmp_path: Path,
):
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
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        patch(
            "vox.core.model_resolution.detect_runtime_capabilities",
            return_value=_runtime_caps(torch_cuda=False, nvidia_device=False),
        ),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="kokoro-tts:latest"),
            tasks=PullTaskRegistry(),
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
    downloaded = tmp_path / "model.bin"
    downloaded.write_bytes(b"hello")

    with (
        patch("vox.core.model_resolution.detect_runtime_capabilities", return_value=_runtime_caps()),
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
    ):
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="kokoro-tts:v1.0", variant="onnx"),
            tasks=PullTaskRegistry(),
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
            registry=registry,
            request=model_reference_request_from_fields(name="kokoro-tts:v1.0", variant="mlx"),
            tasks=PullTaskRegistry(),
        )


@pytest.mark.asyncio
async def test_pull_model_voxtral_does_not_load_the_model(tmp_path: Path, monkeypatch):
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
        mock_api_cls.return_value.repo_info.return_value = MagicMock(siblings=[MagicMock(rfilename="model.bin")])
        events = pull_model(
            store=store,
            registry=registry,
            request=model_reference_request_from_fields(name="voxtral-tts-vllm:4b"),
            tasks=PullTaskRegistry(),
        )
        collected = [event async for event in events]

    assert not any("preloading" in e.status for e in collected)
    assert collected[-1].status == "success"
    scheduler.preload.assert_not_awaited()


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
