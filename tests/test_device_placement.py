"""Tests for the device_placement module — single source of truth for placement decisions."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest

from vox.core.device_placement import (
    DEVICE_MEMORY_HEADROOM_BYTES,
    LoadedModelView,
    Placement,
    PlacementTier,
    auto_device_for_model,
    decide_placement,
    runtime_profile_for_alias,
    select_tier,
)
from vox.core.runtime import RuntimeCapabilities
from vox.core.types import ModelFormat, ModelInfo, ModelType


def _make_info(*, fmt: ModelFormat = ModelFormat.PYTORCH, name: str = "m", tag: str = "v1") -> ModelInfo:
    return ModelInfo(
        name=name,
        tag=tag,
        type=ModelType.STT,
        format=fmt,
        architecture="fake",
        adapter="fake",
    )


def _caps(
    *,
    system: str = "linux",
    machine: str = "x86_64",
    torch_cuda: bool = False,
    onnx_cuda: bool = False,
    onnx_coreml: bool = False,
    mps: bool = False,
    nvidia_device: bool = False,
) -> RuntimeCapabilities:
    return RuntimeCapabilities(
        system=system,
        machine=machine,
        torch_cuda=torch_cuda,
        onnx_cuda=onnx_cuda,
        onnx_coreml=onnx_coreml,
        mps=mps,
        nvidia_device=nvidia_device,
    )


class TestAutoDeviceForModel:
    def test_onnx_picks_cuda_when_onnx_cuda_available(self):
        info = _make_info(fmt=ModelFormat.ONNX)
        assert auto_device_for_model(info, _caps(onnx_cuda=True)) == "cuda"

    def test_onnx_falls_back_to_auto_when_only_coreml(self):
        info = _make_info(fmt=ModelFormat.ONNX)
        assert auto_device_for_model(info, _caps(onnx_coreml=True)) == "auto"

    def test_onnx_falls_back_to_cpu_with_no_accelerator(self):
        info = _make_info(fmt=ModelFormat.ONNX)
        assert auto_device_for_model(info, _caps()) == "cpu"

    def test_pytorch_picks_cuda_when_torch_cuda_available(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        assert auto_device_for_model(info, _caps(torch_cuda=True)) == "cuda"

    def test_pytorch_picks_mps_when_mps_available_and_no_cuda(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        assert auto_device_for_model(info, _caps(mps=True)) == "mps"

    def test_pytorch_falls_back_to_cpu(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        assert auto_device_for_model(info, _caps()) == "cpu"

    def test_ct2_uses_pytorch_path(self):
        info = _make_info(fmt=ModelFormat.CT2)
        assert auto_device_for_model(info, _caps(torch_cuda=True)) == "cuda"
        assert auto_device_for_model(info, _caps(mps=True)) == "mps"
        assert auto_device_for_model(info, _caps()) == "cpu"

    def test_unknown_format_returns_cpu(self):
        info = _make_info(fmt=ModelFormat.GGUF)
        assert auto_device_for_model(info, _caps(torch_cuda=True)) == "cpu"


class TestRuntimeProfileForAlias:
    def test_returns_default_for_non_linux(self, monkeypatch):
        monkeypatch.setattr(
            "vox.core.device_placement.platform.machine", lambda: "arm64"
        )
        with patch(
            "vox.core.device_placement.infer_runtime_profile",
            return_value="default",
        ):
            assert runtime_profile_for_alias(device_hint="auto") == "default"

    def test_forces_spark_when_cuda_hint_on_arm64(self, monkeypatch):
        monkeypatch.setattr(
            "vox.core.device_placement.platform.machine", lambda: "arm64"
        )
        assert runtime_profile_for_alias(device_hint="cuda") == "spark"

    def test_forces_spark_when_cuda_hint_on_aarch64(self, monkeypatch):
        monkeypatch.setattr(
            "vox.core.device_placement.platform.machine", lambda: "aarch64"
        )
        assert runtime_profile_for_alias(device_hint="cuda") == "spark"

    def test_delegates_to_infer_runtime_profile_for_other_hints(self, monkeypatch):
        monkeypatch.setattr(
            "vox.core.device_placement.platform.machine", lambda: "x86_64"
        )
        with patch(
            "vox.core.device_placement.infer_runtime_profile",
            return_value="spark",
        ):
            assert runtime_profile_for_alias(device_hint="auto") == "spark"

    def test_reads_vox_device_env_when_no_hint(self, monkeypatch):
        monkeypatch.setenv("VOX_DEVICE", "cuda")
        monkeypatch.setattr(
            "vox.core.device_placement.platform.machine", lambda: "arm64"
        )
        assert runtime_profile_for_alias() == "spark"


class TestSelectTier:
    def test_returns_none_when_no_tiers(self):
        assert select_tier((), total_memory_bytes=16 * 1024**3) is None

    def test_picks_smallest_bound_that_fits(self):
        tiers = (
            PlacementTier(name="small", total_memory_max_bytes=16 * 1024**3),
            PlacementTier(name="medium", total_memory_max_bytes=24 * 1024**3),
            PlacementTier(name="default", total_memory_max_bytes=None),
        )
        assert select_tier(tiers, total_memory_bytes=10 * 1024**3).name == "small"
        assert select_tier(tiers, total_memory_bytes=20 * 1024**3).name == "medium"
        assert select_tier(tiers, total_memory_bytes=80 * 1024**3).name == "default"

    def test_picks_unbounded_when_total_unknown(self):
        tiers = (
            PlacementTier(name="small", total_memory_max_bytes=16 * 1024**3),
            PlacementTier(name="default", total_memory_max_bytes=None),
        )
        assert select_tier(tiers, total_memory_bytes=None).name == "default"

    def test_falls_back_to_largest_bounded_when_no_unbounded(self):
        tiers = (
            PlacementTier(name="small", total_memory_max_bytes=16 * 1024**3),
            PlacementTier(name="big", total_memory_max_bytes=24 * 1024**3),
        )
        assert select_tier(tiers, total_memory_bytes=80 * 1024**3).name == "big"


class TestDecidePlacementCpuOnly:
    def test_pytorch_with_no_accelerator_returns_cpu(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(),
        )
        assert placement.device == "cpu"
        assert placement.evict == []
        assert placement.tier is None


class TestDecidePlacementSingleCuda:
    def test_pytorch_with_cuda_no_estimate_returns_cuda(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
        )
        assert placement.device == "cuda"
        assert placement.evict == []

    def test_pytorch_fits_in_free_memory(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
            estimated_vram_bytes=2_000_000_000,
            free_memory_query=lambda d: 4_500_000_000,
        )
        assert placement.device == "cuda"
        assert placement.evict == []

    def test_oom_falls_back_to_cpu_when_no_evictable_models(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
            estimated_vram_bytes=4_000_000_000,
            free_memory_query=lambda d: 3_500_000_000,
        )
        assert placement.device == "cpu"
        assert placement.evict == []

    def test_oom_evicts_idle_cuda_model_to_make_room(self):
        info = _make_info(fmt=ModelFormat.PYTORCH, name="candidate")
        loaded = [
            LoadedModelView(
                full_name="existing:latest",
                device="cuda",
                vram_bytes=2_000_000_000,
                ref_count=0,
                last_used=time.time() - 120,
            )
        ]
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
            loaded_models=loaded,
            estimated_vram_bytes=4_000_000_000,
            free_memory_query=lambda d: 3_500_000_000,
        )
        assert placement.device == "cuda"
        assert placement.evict == ["existing:latest"]

    def test_in_use_models_are_not_evicted(self):
        info = _make_info(fmt=ModelFormat.PYTORCH, name="candidate")
        loaded = [
            LoadedModelView(
                full_name="busy:latest",
                device="cuda",
                vram_bytes=4_000_000_000,
                ref_count=1,
                last_used=time.time() - 120,
            )
        ]
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
            loaded_models=loaded,
            estimated_vram_bytes=4_000_000_000,
            free_memory_query=lambda d: 3_500_000_000,
        )
        assert placement.device == "cpu"
        assert placement.evict == []

    def test_evicts_lru_first(self):
        info = _make_info(fmt=ModelFormat.PYTORCH, name="candidate")
        now = time.time()
        loaded = [
            LoadedModelView(
                full_name="newer:latest",
                device="cuda",
                vram_bytes=2_000_000_000,
                ref_count=0,
                last_used=now - 10,
            ),
            LoadedModelView(
                full_name="older:latest",
                device="cuda",
                vram_bytes=2_000_000_000,
                ref_count=0,
                last_used=now - 200,
            ),
        ]
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
            loaded_models=loaded,
            estimated_vram_bytes=4_000_000_000,
            free_memory_query=lambda d: 3_500_000_000,
        )
        assert placement.device == "cuda"
        assert placement.evict[0] == "older:latest"


class TestDecidePlacementExplicitDevice:
    def test_explicit_cuda_is_returned_even_when_estimate_exceeds_free(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        placement = decide_placement(
            info,
            requested_device="cuda",
            capabilities=_caps(torch_cuda=True),
            estimated_vram_bytes=16_000_000_000,
            free_memory_query=lambda d: 3_500_000_000,
        )
        assert placement.device == "cuda"
        assert placement.evict == []

    def test_explicit_cpu_is_passed_through(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        placement = decide_placement(
            info,
            requested_device="cpu",
            capabilities=_caps(torch_cuda=True),
            estimated_vram_bytes=4_000_000_000,
        )
        assert placement.device == "cpu"


class TestDecidePlacementMps:
    def test_pytorch_picks_mps_in_auto_mode(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(mps=True),
        )
        assert placement.device == "mps"


class TestDecidePlacementSparkProfile:
    def test_arm64_with_torch_cuda_picks_cuda(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(machine="aarch64", torch_cuda=True, nvidia_device=True),
        )
        assert placement.device == "cuda"


class TestDecidePlacementWithTiers:
    def test_publishes_chosen_tier_in_placement(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        tiers = (
            PlacementTier(
                name="small-gpu",
                total_memory_max_bytes=16 * 1024**3,
                extras={"gpu_memory_utilization": 0.62, "vram_bytes": 12_000_000_000},
            ),
            PlacementTier(
                name="big-gpu",
                total_memory_max_bytes=None,
                extras={"gpu_memory_utilization": 0.4},
            ),
        )
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
            tiers=tiers,
            total_memory_query=lambda d: 16 * 1024**3,
        )
        assert placement.device == "cuda"
        assert placement.tier == "small-gpu"
        assert placement.notes["gpu_memory_utilization"] == 0.62

    def test_no_tier_when_device_is_not_cuda(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        tiers = (
            PlacementTier(
                name="any",
                total_memory_max_bytes=None,
                extras={"vram_bytes": 12_000_000_000},
            ),
        )
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(),
            tiers=tiers,
        )
        assert placement.device == "cpu"
        assert placement.tier is None
        assert placement.notes == {}

    def test_tier_passed_through_when_explicit_cuda(self):
        info = _make_info(fmt=ModelFormat.PYTORCH)
        tiers = (
            PlacementTier(
                name="default",
                total_memory_max_bytes=None,
                extras={"profile": "default"},
            ),
        )
        placement = decide_placement(
            info,
            requested_device="cuda",
            capabilities=_caps(torch_cuda=True),
            tiers=tiers,
            total_memory_query=lambda d: None,
        )
        assert placement.device == "cuda"
        assert placement.tier == "default"
        assert placement.notes["profile"] == "default"


class TestVoxtralTierTable:
    """Adapter publishes tier table, placement picks one — without vllm installed."""

    def test_voxtral_tts_tier_picked_for_16gb_gpu(self):
        from vox.core.adapter import BaseAdapter

        class FakeVoxtralAdapter(BaseAdapter):
            def info(self):
                raise NotImplementedError

            def load(self, model_path, device, **kwargs):
                raise NotImplementedError

            def unload(self):
                pass

            @property
            def is_loaded(self):
                return False

            def placement_tiers(self):
                return (
                    PlacementTier(
                        name="voxtral-16gb",
                        total_memory_max_bytes=16 * 1024**3,
                        extras={
                            "generation_gpu_memory_utilization": 0.62,
                            "kv_cache_dtype": "fp8",
                            "attention_backend": "triton_attn",
                        },
                    ),
                    PlacementTier(
                        name="voxtral-24gb",
                        total_memory_max_bytes=24 * 1024**3,
                        extras={"generation_gpu_memory_utilization": 0.4},
                    ),
                    PlacementTier(
                        name="voxtral-default",
                        total_memory_max_bytes=None,
                        extras={"generation_gpu_memory_utilization": 0.4},
                    ),
                )

        adapter = FakeVoxtralAdapter()
        tiers = adapter.placement_tiers()
        info = _make_info(fmt=ModelFormat.PYTORCH, name="voxtral-tts-vllm", tag="4b")
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
            tiers=tiers,
            total_memory_query=lambda d: 16 * 1024**3,
        )
        assert placement.tier == "voxtral-16gb"
        assert placement.notes["kv_cache_dtype"] == "fp8"
        assert placement.notes["attention_backend"] == "triton_attn"

    def test_voxtral_tts_tier_picked_for_24gb_gpu(self):
        from vox.core.adapter import BaseAdapter

        class FakeVoxtralAdapter(BaseAdapter):
            def info(self):
                raise NotImplementedError

            def load(self, model_path, device, **kwargs):
                raise NotImplementedError

            def unload(self):
                pass

            @property
            def is_loaded(self):
                return False

            def placement_tiers(self):
                return (
                    PlacementTier(
                        name="voxtral-16gb",
                        total_memory_max_bytes=16 * 1024**3,
                        extras={"generation_gpu_memory_utilization": 0.62},
                    ),
                    PlacementTier(
                        name="voxtral-24gb",
                        total_memory_max_bytes=24 * 1024**3,
                        extras={"generation_gpu_memory_utilization": 0.4},
                    ),
                    PlacementTier(
                        name="voxtral-default",
                        total_memory_max_bytes=None,
                        extras={"generation_gpu_memory_utilization": 0.4},
                    ),
                )

        info = _make_info(fmt=ModelFormat.PYTORCH, name="voxtral-tts-vllm", tag="4b")
        placement = decide_placement(
            info,
            requested_device="auto",
            capabilities=_caps(torch_cuda=True),
            tiers=FakeVoxtralAdapter().placement_tiers(),
            total_memory_query=lambda d: 24 * 1024**3,
        )
        assert placement.tier == "voxtral-24gb"


class TestPlacementDataclass:
    def test_default_evict_list_is_empty(self):
        placement = Placement(device="cuda")
        assert placement.evict == []
        assert placement.notes == {}
        assert placement.tier is None
        assert placement.precision is None
