from __future__ import annotations

import pytest

from vox.core.types import (
    DeviceMemoryInfo,
    LoadedModelInfo,
    ModelType,
    ProcessMemoryInfo,
    VramSnapshot,
)
from vox.operations.errors import ModelInUseError
from vox.operations.system import (
    HealthStatusRequest,
    HealthStatusResult,
    ListLoadedModelsRequest,
    ListLoadedModelsResult,
    MemoryStatusRequest,
    TrimIdleMemoryRequest,
    TrimIdleResult,
    TrimModelRequest,
    TrimModelResult,
    UnloadIdleModelsRequest,
    UnloadIdleResult,
    get_health_status,
    get_memory_status,
    health_status_payload,
    health_status_request_from_fields,
    list_loaded_models,
    list_loaded_models_payload,
    list_loaded_models_request_from_fields,
    memory_snapshot_payload,
    memory_status_payload,
    memory_status_request_from_fields,
    trim_idle_memory,
    trim_idle_memory_request_from_fields,
    trim_idle_payload,
    trim_model,
    trim_model_payload,
    trim_model_request_from_fields,
    unload_idle_models,
    unload_idle_models_request_from_fields,
    unload_idle_payload,
)


class FakeSystemScheduler:
    def __init__(self) -> None:
        self.loaded = [
            LoadedModelInfo(
                name="parakeet-stt-onnx",
                tag="tdt-0.6b-v3",
                type=ModelType.STT,
                device="cuda",
                vram_bytes=1024,
                ref_count=0,
                is_evictable=True,
                is_trimmable=True,
                backend_memory={"workspace_bytes": 128},
            ),
            LoadedModelInfo(
                name="chatterbox-tts-turbo",
                tag="0.1.7",
                type=ModelType.TTS,
                device="cuda",
                vram_bytes=2048,
                ref_count=1,
            ),
        ]
        self.trim_idle_calls: list[int] = []
        self.trim_calls: list[str] = []
        self.unload_calls: list[str] = []
        self.trim_result = True

    def list_loaded(self):
        return self.loaded

    def memory_snapshot(self):
        return VramSnapshot(
            device=DeviceMemoryInfo(
                device="cuda",
                free_bytes=5_000,
                total_bytes=20_000,
                torch_allocated_bytes=3_000,
                torch_reserved_bytes=4_000,
            ),
            process=ProcessMemoryInfo(
                rss_bytes=6_000,
                peak_rss_bytes=7_000,
                cgroup_current_bytes=8_000,
                cgroup_peak_bytes=9_000,
                cgroup_limit_bytes=10_000,
            ),
            idle_trim_seconds=30,
            loaded_models=tuple(self.loaded),
            estimated_loaded_vram_bytes=3_072,
            active_model_count=1,
        )

    async def trim(self, model_name: str) -> bool:
        self.trim_calls.append(model_name)
        return self.trim_result

    async def trim_idle(self, *, min_idle_seconds: int = 0):
        self.trim_idle_calls.append(min_idle_seconds)
        return ["parakeet-stt-onnx:tdt-0.6b-v3"]

    async def unload(self, model_name: str) -> bool:
        self.unload_calls.append(model_name)
        return True


def test_memory_status_payload_preserves_http_contract_shape():
    scheduler = FakeSystemScheduler()

    result = get_memory_status(scheduler=scheduler, request=MemoryStatusRequest())
    payload = memory_status_payload(result)

    assert "policy" not in payload
    assert set(payload) == {
        "device",
        "process",
        "idle_trim_seconds",
        "estimated_loaded_vram_bytes",
        "active_model_count",
        "models",
    }
    assert payload["idle_trim_seconds"] == 30
    assert payload["device"]["torch_reserved_bytes"] == 4_000
    assert payload["process"] == {
        "rss_bytes": 6_000,
        "peak_rss_bytes": 7_000,
        "cgroup_current_bytes": 8_000,
        "cgroup_peak_bytes": 9_000,
        "cgroup_limit_bytes": 10_000,
    }
    assert payload["estimated_loaded_vram_bytes"] == 3_072
    assert payload["active_model_count"] == 1
    assert payload["models"][0]["backend_memory"] == {"workspace_bytes": 128}


def test_memory_snapshot_process_metrics_are_backward_compatible_for_python_callers():
    snapshot = VramSnapshot(
        device=DeviceMemoryInfo(device="cpu"),
        idle_trim_seconds=0,
        loaded_models=(),
        estimated_loaded_vram_bytes=0,
        active_model_count=0,
    )

    assert snapshot.process == ProcessMemoryInfo()


def test_health_status_payload_preserves_http_contract_shape():
    result = get_health_status(request=HealthStatusRequest())

    assert result == HealthStatusResult()
    assert health_status_payload(result) == {"status": "ok"}


def test_list_loaded_models_payload_preserves_existing_contract_shape():
    scheduler = FakeSystemScheduler()

    result = list_loaded_models(scheduler=scheduler, request=ListLoadedModelsRequest())
    payload = list_loaded_models_payload(result)

    assert result == ListLoadedModelsResult(models=scheduler.loaded)
    assert payload == {
        "models": [
            {
                "name": "parakeet-stt-onnx",
                "tag": "tdt-0.6b-v3",
                "type": "stt",
                "device": "cuda",
                "vram_bytes": 1024,
                "loaded_at": 0.0,
                "last_used": 0.0,
                "ref_count": 0,
            },
            {
                "name": "chatterbox-tts-turbo",
                "tag": "0.1.7",
                "type": "tts",
                "device": "cuda",
                "vram_bytes": 2048,
                "loaded_at": 0.0,
                "last_used": 0.0,
                "ref_count": 1,
            },
        ]
    }
    assert "backend_memory" not in payload["models"][0]
    assert "is_evictable" not in payload["models"][0]
    assert "is_trimmable" not in payload["models"][0]


def test_memory_snapshot_loaded_model_payload_extends_loaded_model_summary_shape():
    scheduler = FakeSystemScheduler()

    summary = list_loaded_models_payload(list_loaded_models(scheduler=scheduler, request=ListLoadedModelsRequest()))[
        "models"
    ][0]
    detailed = memory_snapshot_payload(scheduler.memory_snapshot())["models"][0]

    assert {key: detailed[key] for key in summary} == summary
    assert detailed["is_evictable"] is True
    assert detailed["is_trimmable"] is True
    assert detailed["backend_memory"] == {"workspace_bytes": 128}


def test_system_operation_request_builders_preserve_transport_values():
    assert health_status_request_from_fields() == HealthStatusRequest()
    assert memory_status_request_from_fields() == MemoryStatusRequest()
    assert list_loaded_models_request_from_fields() == ListLoadedModelsRequest()
    assert trim_idle_memory_request_from_fields(
        min_idle_seconds=-15,
    ) == TrimIdleMemoryRequest(min_idle_seconds=-15)
    assert trim_model_request_from_fields(
        model_name="kokoro-tts:v1.0",
    ) == TrimModelRequest(model_name="kokoro-tts:v1.0")
    assert unload_idle_models_request_from_fields() == UnloadIdleModelsRequest()


def test_system_action_payloads_preserve_http_contract_shapes():
    scheduler = FakeSystemScheduler()
    snapshot = scheduler.memory_snapshot()

    assert trim_idle_payload(TrimIdleResult(trimmed=["parakeet-stt-onnx:tdt-0.6b-v3"], snapshot=snapshot)) == {
        "trimmed": ["parakeet-stt-onnx:tdt-0.6b-v3"],
        "memory": memory_snapshot_payload(snapshot),
    }
    assert trim_model_payload(TrimModelResult()) == {"status": "success"}
    assert unload_idle_payload(UnloadIdleResult(unloaded=["parakeet-stt-onnx:tdt-0.6b-v3"], snapshot=snapshot)) == {
        "unloaded": ["parakeet-stt-onnx:tdt-0.6b-v3"],
        "memory": memory_snapshot_payload(snapshot),
    }


@pytest.mark.asyncio
async def test_trim_idle_clamps_negative_idle_seconds():
    scheduler = FakeSystemScheduler()

    result = await trim_idle_memory(
        scheduler=scheduler,
        request=TrimIdleMemoryRequest(min_idle_seconds=-15),
    )

    assert scheduler.trim_idle_calls == [0]
    assert result.trimmed == ["parakeet-stt-onnx:tdt-0.6b-v3"]


@pytest.mark.asyncio
async def test_trim_model_raises_operation_error_when_model_is_active():
    scheduler = FakeSystemScheduler()
    scheduler.trim_result = False

    with pytest.raises(ModelInUseError, match="currently in use"):
        await trim_model(
            scheduler=scheduler,
            request=TrimModelRequest(model_name="chatterbox-tts-turbo:0.1.7"),
        )


@pytest.mark.asyncio
async def test_unload_idle_models_unloads_only_inactive_models():
    scheduler = FakeSystemScheduler()

    result = await unload_idle_models(
        scheduler=scheduler,
        request=UnloadIdleModelsRequest(),
    )

    assert scheduler.unload_calls == ["parakeet-stt-onnx:tdt-0.6b-v3"]
    assert result.unloaded == ["parakeet-stt-onnx:tdt-0.6b-v3"]
