from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from vox.core.adapter_resolution import (
    AdapterMutation,
    bind_adapter_mutation,
)
from vox.core.adapter_runtime import (
    RuntimeMutation,
    bind_runtime_mutation,
    stage_adapter_runtime_mutation,
)
from vox.core.atomic_install import bind_install_transaction
from vox.core.capabilities import incompatible_pull_allowed
from vox.core.hf_runtime import configure_hf_runtime
from vox.core.model_resolution import parse_model_variant_ref, resolve_catalog_entry
from vox.core.pull_transaction import PullTransaction, recover_pull_transactions
from vox.core.store import Manifest, ManifestLayer
from vox.core.types import ModelInfo
from vox.operations.errors import (
    CatalogEntryNotFoundError,
    ModelIncompatibleError,
    ModelInUseError,
    StoredModelNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelLayer:
    media_type: str
    digest: str
    size: int
    filename: str


@dataclass(frozen=True)
class ShowResult:
    name: str
    config: dict[str, Any]
    layers: tuple[ModelLayer, ...]


@dataclass(frozen=True)
class PullEvent:
    status: str
    completed: int = 0
    total: int = 0
    error: str = ""


@dataclass(frozen=True)
class ArtifactSource:
    source: str
    prefix: str = ""
    files: tuple[str, ...] | None = None


_PULL_EVENT_LIMIT = 64
_PULL_EVENT_END = object()


class PullEventStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[PullEvent | object] = asyncio.Queue(maxsize=_PULL_EVENT_LIMIT)
        self._detached = False

    def emit(self, event: PullEvent) -> None:
        if self._detached:
            return
        while self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(event)

    def finish(self) -> None:
        if self._detached:
            return
        while self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(_PULL_EVENT_END)

    def __aiter__(self) -> PullEventStream:
        return self

    async def __anext__(self) -> PullEvent:
        item = await self._queue.get()
        if item is _PULL_EVENT_END:
            raise StopAsyncIteration
        if not isinstance(item, PullEvent):
            raise RuntimeError("invalid pull progress event")
        return item

    async def aclose(self) -> None:
        self._detached = True
        while not self._queue.empty():
            self._queue.get_nowait()


class PullTaskRegistry:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    def start(
        self,
        operation: Callable[[Callable[[PullEvent], None]], Awaitable[None]],
    ) -> PullEventStream:
        if self._closed:
            raise RuntimeError("model pull service is closed")
        stream = PullEventStream()

        async def run() -> None:
            try:
                await operation(stream.emit)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("unhandled model pull failure")
                stream.emit(PullEvent(status="error", error=str(exc)))
            finally:
                stream.finish()

        task = asyncio.create_task(run())
        self._tasks.add(task)

        def completed(done: asyncio.Task[None]) -> None:
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)
        return stream

    async def close(self, *, deadline: float) -> None:
        self._closed = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        _done, pending = await asyncio.wait(
            tasks,
            timeout=max(0.0, deadline - time.monotonic()),
        )
        if pending:
            raise TimeoutError(f"model pull shutdown timed out with {len(pending)} active operation(s)")


@dataclass(frozen=True)
class ModelReferenceRequest:
    name: str
    variant: str | None = None


@dataclass(frozen=True)
class ResolvedModelReference:
    requested_name: str
    parsed_name: str
    parsed_tag: str
    requested_variant: str | None
    resolved_name: str
    resolved_tag: str
    explicit_tag: bool


@dataclass
class _PullPublication:
    adapter_mutation: AdapterMutation | None
    runtime_mutation: RuntimeMutation | None
    blob_lease: Any
    transaction: PullTransaction
    committed: bool = False

    def commit(self) -> None:
        self.committed = True
        operations: list[tuple[str, Callable[[], Any]]] = [
            ("transaction state", self.transaction.mark_committed),
            *[
                (label, mutation.commit)
                for label, mutation in (
                    ("runtime mutation", self.runtime_mutation),
                    ("adapter mutation", self.adapter_mutation),
                )
                if mutation is not None
            ],
            ("blob lease", self.blob_lease.close),
            ("transaction cleanup", self.transaction.finish),
        ]
        for label, operation in operations:
            try:
                operation()
            except BaseException as exc:
                logger.warning(
                    "pull publication cleanup deferred owner=%s error=%s",
                    label,
                    exc,
                )


def model_reference_request_from_fields(
    *,
    name: str,
    variant: str | None = None,
) -> ModelReferenceRequest:
    return ModelReferenceRequest(name=name, variant=variant or None)


def model_info_payload(model: ModelInfo) -> dict[str, Any]:
    return {
        "name": model.full_name,
        "type": model.type.value,
        "format": model.format.value,
        "architecture": model.architecture,
        "size_bytes": model.size_bytes,
        "description": model.description,
    }


def list_models_payload(models: list[ModelInfo]) -> dict[str, Any]:
    return {"models": [model_info_payload(model) for model in models]}


def model_layer_payload(layer: ModelLayer) -> dict[str, Any]:
    return {
        "media_type": layer.media_type,
        "digest": layer.digest,
        "size": layer.size,
        "filename": layer.filename,
    }


def show_model_payload(result: ShowResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "config": result.config,
        "layers": [model_layer_payload(layer) for layer in result.layers],
    }


def pull_event_payload(event: PullEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": event.status}
    if event.total > 0:
        payload["completed"] = event.completed
        payload["total"] = event.total
    if event.error:
        payload["error"] = event.error
    return payload


def delete_model_payload() -> dict[str, str]:
    return {"status": "success"}


def list_models(*, store: Any) -> list[ModelInfo]:
    return list(store.list_models())


def show_model(*, store: Any, registry: Any, request: ModelReferenceRequest) -> ShowResult:
    resolved = resolve_model_reference(registry=registry, request=request)
    manifest = store.resolve_model(resolved.resolved_name, resolved.resolved_tag)
    if not manifest:
        raise StoredModelNotFoundError(request.name)
    return ShowResult(
        name=request.name,
        config=dict(manifest.config),
        layers=tuple(
            ModelLayer(
                media_type=layer.media_type,
                digest=layer.digest,
                size=layer.size,
                filename=layer.filename,
            )
            for layer in manifest.layers
        ),
    )


async def delete_model(
    *,
    store: Any,
    scheduler: Any,
    registry: Any,
    request: ModelReferenceRequest,
) -> None:
    resolved = resolve_model_reference(registry=registry, request=request)
    writer = await _acquire_writer_owned(store)
    try:
        with store.bind_writer_lease(writer):
            recover_pull_transactions(store)
            manifest = store.resolve_model(
                resolved.resolved_name,
                resolved.resolved_tag,
            )
            if not manifest:
                raise StoredModelNotFoundError(request.name)

            unloaded = await scheduler.unload(f"{resolved.resolved_name}:{resolved.resolved_tag}")
            if not unloaded:
                raise ModelInUseError(request.name)

            store.delete_model(resolved.resolved_name, resolved.resolved_tag)
            store.gc_blobs()
            logger.info(
                "model deleted: %s:%s",
                resolved.resolved_name,
                resolved.resolved_tag,
            )
    finally:
        await _run_blocking_owned(writer.close)


def pull_model(
    *,
    store: Any,
    registry: Any,
    request: ModelReferenceRequest,
    tasks: PullTaskRegistry,
) -> PullEventStream:
    resolved = resolve_model_reference(registry=registry, request=request)
    catalog_entry = registry.lookup(
        resolved.parsed_name,
        resolved.parsed_tag,
        explicit_tag=resolved.explicit_tag,
    )

    if not catalog_entry:
        raise CatalogEntryNotFoundError(request.name)

    variant_resolution = resolve_catalog_entry(
        catalog_entry,
        forced_variant=resolved.requested_variant,
    )
    catalog_entry = variant_resolution.entry
    missing = list(variant_resolution.missing)
    if not incompatible_pull_allowed() and missing:
        raise ModelIncompatibleError(request.name, missing)
    if not catalog_entry:
        raise ModelIncompatibleError(request.name, missing or ["no compatible catalog entry resolved"])

    logger.info(
        "pull requested: %s -> %s:%s (variant=%s, adapter=%s, source=%s)",
        request.name,
        resolved.resolved_name,
        resolved.resolved_tag,
        variant_resolution.variant_id or "-",
        catalog_entry.get("adapter", "?"),
        catalog_entry.get("source", "?"),
    )

    async def execute(emit: Callable[[PullEvent], None]) -> None:
        emit(PullEvent(status=f"pulling {request.name}"))
        if missing:
            emit(
                PullEvent(
                    status="warning",
                    error=("pull compatibility bypassed by VOX_ALLOW_INCOMPATIBLE=1: " + "; ".join(missing)),
                )
            )
        for warning in variant_resolution.warnings:
            emit(PullEvent(status="warning", error=warning))

        writer = await _acquire_writer_owned(store)
        try:
            with store.bind_writer_lease(writer):
                recover_pull_transactions(store)
                await _execute_pull(
                    store=store,
                    registry=registry,
                    request=request,
                    resolved=resolved,
                    catalog_entry=catalog_entry,
                    variant_resolution=variant_resolution,
                    emit=emit,
                )
        finally:
            await _run_blocking_owned(writer.close)

    return tasks.start(execute)


def _artifact_sources(catalog_entry: dict[str, Any]) -> tuple[ArtifactSource, ...]:
    primary_source = str(catalog_entry.get("source") or "").strip()
    if not primary_source:
        raise ValueError("catalog entry requires a non-empty source")
    sources = [
        ArtifactSource(
            source=primary_source,
            files=_artifact_files(catalog_entry.get("files"), label="catalog files"),
        )
    ]
    raw_artifacts = catalog_entry.get("artifacts")
    if raw_artifacts is not None and not isinstance(raw_artifacts, (list, tuple)):
        raise ValueError("catalog artifacts must be a list")
    for raw in raw_artifacts or ():
        if not isinstance(raw, dict):
            raise ValueError("catalog artifacts entries must be objects")
        artifact_source = str(raw.get("source") or "").strip()
        if not artifact_source:
            raise ValueError("catalog artifacts entries require a non-empty source")
        prefix = str(raw.get("prefix") or "").strip("/")
        raw_files = raw.get("files")
        sources.append(
            ArtifactSource(
                source=artifact_source,
                prefix=prefix,
                files=_artifact_files(raw_files, label=f"artifact files for {artifact_source}"),
            )
        )
    return tuple(sources)


def _artifact_files(value: Any, *, label: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    files: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{label} must contain non-empty strings")
        files.append(raw)
    return tuple(files)


def _artifact_target_filename(prefix: str, filename: str) -> str:
    normalized = filename.replace("\\", "/")
    if not normalized or normalized in (".", ".."):
        raise ValueError(f"unsafe model artifact filename: {filename!r}")
    source_path = PurePosixPath(normalized)
    if source_path.is_absolute() or PureWindowsPath(filename).is_absolute() or ".." in source_path.parts:
        raise ValueError(f"unsafe model artifact filename: {filename!r}")

    prefix_path = PurePosixPath(prefix.replace("\\", "/")) if prefix else PurePosixPath()
    if prefix_path.is_absolute() or PureWindowsPath(prefix).is_absolute() or ".." in prefix_path.parts:
        raise ValueError(f"unsafe model artifact prefix: {prefix!r}")
    return str(prefix_path / source_path)


async def _execute_pull(
    *,
    store: Any,
    registry: Any,
    request: ModelReferenceRequest,
    resolved: ResolvedModelReference,
    catalog_entry: dict[str, Any],
    variant_resolution: Any,
    emit: Callable[[PullEvent], None],
) -> None:
    adapter_name = catalog_entry.get("adapter", "")
    adapter_package = catalog_entry.get("adapter_package", "")
    source = catalog_entry["source"]
    blob_lease = store.acquire_blob_lease()
    previous_manifest = store.resolve_model(
        resolved.resolved_name,
        resolved.resolved_tag,
    )
    adapter_mutation: AdapterMutation | None = None
    runtime_mutation: RuntimeMutation | None = None
    transaction: PullTransaction | None = None
    publication: _PullPublication | None = None

    try:
        configure_hf_runtime()
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        artifact_sources = _artifact_sources(catalog_entry)
        downloads: list[tuple[ArtifactSource, str, str]] = []
        target_filenames: set[str] = set()
        for artifact_source in artifact_sources:
            source_files = artifact_source.files
            if source_files is None:
                repo_info = await asyncio.to_thread(api.repo_info, artifact_source.source)
                source_files = tuple(
                    sibling.rfilename
                    for sibling in repo_info.siblings or ()
                    if not sibling.rfilename.startswith(".")
                )
            for filename in source_files:
                target_filename = _artifact_target_filename(artifact_source.prefix, filename)
                if target_filename in target_filenames:
                    raise ValueError(f"duplicate model artifact target filename: {target_filename}")
                target_filenames.add(target_filename)
                downloads.append((artifact_source, filename, target_filename))

        layers: list[ManifestLayer] = []
        total_files = len(downloads)

        for i, (artifact_source, filename, target_filename) in enumerate(downloads):
            emit(
                PullEvent(
                    status=f"downloading {artifact_source.source}/{filename}",
                    completed=i,
                    total=total_files,
                )
            )

            local_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=artifact_source.source,
                filename=filename,
                cache_dir=None,
            )

            file_size = os.path.getsize(local_path)
            digest = await _run_blocking_owned(
                _write_blob_from_path,
                blob_lease,
                local_path,
            )

            ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
            media_type = f"application/vox.model.{ext}"

            layers.append(
                ManifestLayer(
                    media_type=media_type,
                    digest=digest,
                    size=file_size,
                    filename=target_filename,
                )
            )

        transaction = PullTransaction.begin(
            store=store,
            name=resolved.resolved_name,
            tag=resolved.resolved_tag,
            previous_manifest=previous_manifest,
            candidate_digests=tuple(layer.digest for layer in layers),
        )

        if adapter_package:
            emit(PullEvent(status=f"checking adapter {adapter_name}"))
            with bind_install_transaction(transaction):
                adapter_mutation = await _stage_adapter_mutation_owned(
                    registry.stage_adapter,
                    adapter_name,
                    adapter_package,
                )
            if not adapter_mutation.ready:
                raise RuntimeError(f"Failed to install adapter package: {adapter_package}")
            emit(PullEvent(status=f"adapter {adapter_name} ready"))

        if adapter_name:
            emit(PullEvent(status=f"preparing adapter runtime {adapter_name}"))
            with (
                bind_install_transaction(transaction),
                bind_adapter_mutation(adapter_mutation),
            ):
                runtime_mutation = await _stage_runtime_mutation_owned(
                    _prepare_adapter_runtime,
                    registry,
                    adapter_name,
                )
            emit(PullEvent(status=f"adapter runtime {adapter_name} ready"))

        manifest = Manifest(
            layers=layers,
            config={
                "architecture": catalog_entry["architecture"],
                "type": catalog_entry["type"],
                "adapter": catalog_entry["adapter"],
                "format": catalog_entry["format"],
                "source": source,
                "runtime_source": catalog_entry.get("runtime_source", ""),
                "parameters": catalog_entry.get("parameters", {}),
                "description": catalog_entry.get("description", ""),
                "license": catalog_entry.get("license", ""),
                "adapter_package": catalog_entry.get("adapter_package", ""),
                "runtime": _runtime_diagnostic_payload(
                    variant_id=variant_resolution.variant_id,
                    preferred_backend=variant_resolution.preferred_backend,
                    warnings=variant_resolution.warnings,
                    snapshot=variant_resolution.snapshot,
                ),
            },
            transaction_id=transaction.id,
        )
        transaction.record_candidate_manifest(manifest)
        publication = _PullPublication(
            adapter_mutation=adapter_mutation,
            runtime_mutation=runtime_mutation,
            blob_lease=blob_lease,
            transaction=transaction,
        )
        manifest_publication_error: BaseException | None = None
        try:
            with (
                bind_install_transaction(transaction),
                bind_adapter_mutation(adapter_mutation),
                bind_runtime_mutation(runtime_mutation),
            ):
                store.save_manifest(
                    resolved.resolved_name,
                    resolved.resolved_tag,
                    manifest,
                )
        except BaseException as exc:
            if not transaction.owns_canonical_manifest():
                raise
            manifest_publication_error = exc

        await _run_blocking_owned(publication.commit)
        adapter_mutation = None
        runtime_mutation = None
        transaction = None
        if manifest_publication_error is not None:
            if not isinstance(manifest_publication_error, Exception):
                raise manifest_publication_error
            logger.warning(
                "pull manifest durability confirmation failed after publication: %s",
                manifest_publication_error,
            )
            emit(
                PullEvent(
                    status="warning",
                    error=str(manifest_publication_error),
                )
            )

        total_bytes = sum(layer.size for layer in layers)
        logger.info(
            "pull complete: %s:%s (%d layers, %.1f MiB)",
            resolved.resolved_name,
            resolved.resolved_tag,
            len(layers),
            total_bytes / (1024 * 1024),
        )
        emit(PullEvent(status="success"))

    except BaseException as e:
        if publication is not None and publication.committed:
            adapter_mutation = None
            runtime_mutation = None
            transaction = None
            raise
        cleanup_errors = await _rollback_pull_owners(
            runtime_mutation=runtime_mutation,
            adapter_mutation=adapter_mutation,
            transaction=transaction,
            blob_lease=blob_lease,
        )
        if not isinstance(e, Exception):
            for error in cleanup_errors:
                e.add_note(f"pull rollback failed: {error}")
            raise
        logger.exception("pull failed: %s", request.name)
        error_message = str(e)
        if cleanup_errors:
            failures = "; ".join(str(error) for error in cleanup_errors)
            error_message = f"{error_message}; rollback failures: {failures}"
        emit(PullEvent(status="error", error=error_message))
    finally:
        blob_lease.close()


def _write_blob_from_path(blob_lease: Any, path: str) -> str:
    with open(path, "rb") as stream:
        return blob_lease.write_blob(stream)


def resolve_model_reference(
    *,
    registry: Any,
    request: ModelReferenceRequest,
) -> ResolvedModelReference:
    parsed = parse_model_variant_ref(request.name)
    resolved_name, resolved_tag = registry.resolve_model_ref(
        parsed.name,
        parsed.tag,
        explicit_tag=parsed.explicit_tag,
    )
    return ResolvedModelReference(
        requested_name=request.name,
        parsed_name=parsed.name,
        parsed_tag=parsed.tag,
        requested_variant=request.variant,
        resolved_name=resolved_name,
        resolved_tag=resolved_tag,
        explicit_tag=parsed.explicit_tag,
    )


def _runtime_diagnostic_payload(
    *,
    variant_id: str | None,
    preferred_backend: str | None,
    warnings: tuple[str, ...],
    snapshot: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checked_at_pull": True,
        "resolved_variant": variant_id or "",
        "preferred_backend": preferred_backend or "",
        "warnings": list(warnings),
    }
    if snapshot is not None:
        payload["detected"] = asdict(snapshot)
    return payload


def _prepare_adapter_runtime(registry: Any, adapter_name: str) -> None:
    adapter_cls = registry.get_adapter_class(adapter_name)
    adapter = adapter_cls()
    adapter.prepare_runtime()


async def _run_blocking_owned(
    operation: Callable[..., Any],
    /,
    *args: Any,
) -> Any:
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    result, cancellation = await _complete_owned_task(task)
    if cancellation is not None:
        raise cancellation
    return result


async def _acquire_writer_owned(store: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(store.acquire_writer_lease))
    writer, cancellation = await _complete_owned_task(task)
    if cancellation is None:
        return writer
    await _run_blocking_owned(writer.close)
    raise cancellation


async def _complete_owned_task(
    task: asyncio.Task[Any],
) -> tuple[Any, asyncio.CancelledError | None]:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    if cancellation is not None:
        with suppress(BaseException):
            return task.result(), cancellation
        raise cancellation
    return task.result(), None


async def _rollback_pull_owners(
    *,
    runtime_mutation: RuntimeMutation | None,
    adapter_mutation: AdapterMutation | None,
    transaction: PullTransaction | None,
    blob_lease: Any,
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    operations = [mutation.rollback for mutation in (runtime_mutation, adapter_mutation) if mutation is not None]
    if transaction is not None:
        operations.append(transaction.rollback)
    operations.append(blob_lease.abort)
    for operation in operations:
        try:
            await _run_blocking_owned(operation)
        except BaseException as exc:
            errors.append(exc)
    return tuple(errors)


async def _stage_runtime_mutation_owned(
    operation: Callable[..., Any],
    /,
    *args: Any,
) -> RuntimeMutation:
    task = asyncio.create_task(
        asyncio.to_thread(
            stage_adapter_runtime_mutation,
            operation,
            *args,
        )
    )
    mutation, cancellation = await _complete_owned_task(task)
    if cancellation is None:
        return mutation
    await _run_blocking_owned(mutation.rollback)
    raise cancellation


async def _stage_adapter_mutation_owned(
    operation: Callable[..., Any],
    /,
    *args: Any,
) -> AdapterMutation:
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    mutation, cancellation = await _complete_owned_task(task)
    if cancellation is None:
        return mutation
    await _run_blocking_owned(mutation.rollback)
    raise cancellation
